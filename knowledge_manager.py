import asyncio
import logging
from typing import Type, Dict, List, Tuple, Union
import langchain
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import PGVector
from langchain.chat_models.base import BaseChatModel
from langchain.document_loaders import UnstructuredAPIFileLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

class KnowledgeManager:
    def __init__(
        self,
        embeddings: Embeddings,
        llm_cls: Type[BaseChatModel],
        llm_kwargs: Dict[str, str],
        unstructured_api_key: str,
        connection_string: str,
        chunk_size: int = 2000,
        conversattion_limit: int = 800,
        docs_limit: int = 3000
    ) -> None:
        self.embeddings = embeddings
        self.conversattion_limit = conversattion_limit
        self.docs_limit = docs_limit
        self.llm_cls = llm_cls
        self.llm_kwargs = llm_kwargs
        self.unstructured_api_key = unstructured_api_key
        self.chunk_size = chunk_size
        self.connection_string = connection_string

    def split_docs(self, docs: Document) -> List[Document]:
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size
        ).split_documents(docs)

    def load_data(self, file_path: str) -> Tuple[str, List[Document], bytes]:
        logging.info(f"Loading {file_path}")
        loader = UnstructuredAPIFileLoader(
            file_path=file_path,
            api_key=self.unstructured_api_key,
        )

        docs = loader.load()
        logging.info(f"Documents loaded {docs}")
        docs = self.split_docs(docs)
        contents = "\n\n".join([doc.page_content for doc in docs])

        with open(file_path, "rt") as f:
            logging.info(f"contents :  {f.read()}")

        with open(file_path, "rb") as f:
            file_bytes = f.read()

        return contents, docs, file_bytes

    def injest_data(
        self,
        documents: List[Document],
        ids: List[str] = None,
        collection_name: str = "data",
    ) -> List[str]:
        vectorstore = PGVector(self.connection_string, self.embeddings, collection_name)
        return vectorstore.add_documents(documents, ids=ids)

    def add_metadata_to_docs(self, metadata: Dict, docs: List[Document]):
        for document in docs:
            document.metadata.update(metadata)
        return docs

    def load_and_injest_file(
        self, collection_name: str, filepath: str, metadata: Dict
    ) -> Tuple[str, List[str], bytes]:
        contents, docs, file_bytes = self.load_data(filepath)
        docs = self.add_metadata_to_docs(metadata=metadata, docs=docs)
        ids = self.injest_data(collection_name=collection_name, documents=docs)
        return contents, ids, file_bytes

    def delete_collection(self, collection_name: str) -> bool:
        try:
            vectorstore = PGVector(
                self.connection_string, self.embeddings, collection_name
            )
            vectorstore.delete_collection()
            return True
        except Exception:
            return False

    def delete_ids(self, collection_name: str, ids: list[str]):
        vectorstore = PGVector(self.connection_string, self.embeddings, collection_name)
        return vectorstore.delete(ids)

    def query_data(
        self, query: str, collection_name: str, k: int = 5, metadata: Dict[str, str] = None
    ) -> str:
        vectorstore = PGVector(self.connection_string, self.embeddings, collection_name)
        return vectorstore.similarity_search(query, k, filter=metadata)

    def format_messages(
        self,
        chat_history: List[Tuple[str, str]],
        tokens_limit: int,
        llm: BaseChatModel,
        human_only: bool = False,
        ai_name: str = "AgritechAI",
    ) -> str:
        cleaned_msgs: List[Union[str, Tuple[str, str]]] = []
        tokens_used: int = 0

        for human_msg, ai_msg in chat_history:
            human_msg_formatted = f"Human: {human_msg}"
            ai_msg_formatted = f"{ai_name}: {ai_msg}"

            human_tokens = llm.get_num_tokens(human_msg_formatted)
            ai_tokens = llm.get_num_tokens(ai_msg_formatted)

            if not human_only:
                new_tokens_used = tokens_used + human_tokens + ai_tokens
            else:
                new_tokens_used = tokens_used + human_tokens

            if new_tokens_used > tokens_limit:
                break

            if human_only:
                cleaned_msgs.append(human_msg_formatted)
            else:
                cleaned_msgs.append((human_msg_formatted, ai_msg_formatted))

            tokens_used = new_tokens_used

        if human_only:
            return "\n\n".join(cleaned_msgs)
        else:
            return "\n\n".join(
                [f"{clean_msg[0]}\n\n{clean_msg[1]}" for clean_msg in cleaned_msgs]
            )

    def _reduce_tokens_below_limit(
        self, docs: list, docs_limit: int, llm: BaseChatModel
    ) -> list[Document]:
        num_docs = len(docs)
        tokens = [llm.get_num_tokens(doc.page_content) for doc in docs]
        token_count = sum(tokens[:num_docs])
        while token_count > docs_limit:
            num_docs -= 1
            token_count -= tokens[num_docs]

        return docs[:num_docs]

    async def chat(
        self,
        query: str,
        chat_history: list[tuple[str, str]],
        collection_name: str = "data",
        **kwargs,
    ) -> str:
        
        llm = self.llm_cls(**self.llm_kwargs)
        conversation = self.format_messages(chat_history, self.conversattion_limit, llm)
        combined = (
            self.format_messages(
                chat_history=chat_history,
                tokens_limit=self.conversattion_limit,
                human_only=True,
                llm=llm,
            )
            + "\n"
            + f"Human: {query}"
        )        
        docs = self.query_data(combined, collection_name, **kwargs)
        docs = self._reduce_tokens_below_limit(docs, self.docs_limit, llm)
        docs = "\n\n".join([doc.page_content for doc in docs])
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    """
YOu are an AI named AgriTechAI, You are designed to answer user questions on agriculture and many other topic.
You use the help data to answer user questions.
Use your own knowledge if needed.
Only return the next message content in language of the human. dont return anything else not even the name of AI.
You must answer the human in language of the human (important)"""
                ),
                HumanMessagePromptTemplate.from_template(
                    """
Help Data:
=========
{help_data}
=========

Let's think in a step by step, answer the humans question in language of the human.

{conversation}

Human: {question}

AgritechAI (answers language of the human):"""
                ),
            ],
            input_variables=[
                "help_data",
                "conversation",
                "question",
            ],
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        res = await chain.arun(question=query, conversation=conversation, help_data=docs)
        return res
        
        

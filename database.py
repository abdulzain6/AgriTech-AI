from peewee import *
from contextvars import ContextVar
from typing import List, Optional
from pydantic import BaseModel
from playhouse.sqlite_ext import *

import peewee


db_state_default = {"closed": None, "conn": None, "ctx": None, "transactions": None}
db_state = ContextVar("db_state", default=db_state_default.copy())


class PeeweeConnectionState(peewee._ConnectionState):
    def __init__(self, **kwargs):
        super().__setattr__("_state", db_state)
        super().__init__(**kwargs)

    def __setattr__(self, name, value):
        self._state.get()[name] = value

    def __getattr__(self, name):
        return self._state.get()[name]


class FileModel(BaseModel):
    filename: str
    description: str = None
    filetype: Optional[str] = None
    vector_ids: List[str]
    file_content: Optional[str] = None
    file_bytes: Optional[bytes] = None



class FileDBManager:
    def __init__(
        self, db: SqliteDatabase
    ) -> None:
        class File(Model):
            filename = CharField(null=False)
            filetype = CharField(null=True)
            vector_ids = JSONField(default=list)
            file_content = CharField(null=True)
            file_bytes = BlobField(null=True)
            description = CharField(null=True)

            class Meta:
                database = db

            def to_dict(self) -> dict:
                return {
                    "filename": self.filename,
                    "filetype": self.filetype,
                    "vector_ids": self.vector_ids,
                    "file_content": self.file_content,
                    "file_bytes": self.file_bytes,
                    "description" : self.description
                }

        self.model = File
        db.connect(reuse_if_open=True)
        db.create_tables([self.model], safe=True)
        self.db = db

    def file_to_pydantic(self, file: Model) -> FileModel:
        return FileModel(
            filename=file.filename,
            filetype=file.filetype,
            vector_ids=file.vector_ids,
            file_content=file.file_content,
            file_bytes=bytes(file.file_bytes),
            description=file.description
        )

    def pydantic_to_file(self, file_model: FileModel) -> Model:
        return self.model(
            filename=file_model.filename,
            filetype=file_model.filetype,
            vector_ids=file_model.vector_ids,
            file_content=file_model.file_content,
            file_bytes=file_model.file_bytes,
            description=file_model.description
        )

    def add_file(self, file_model: FileModel) -> FileModel:
        with self.db.connection_context():
            
            existing_file = (
                self.model.select().where(self.model.filename == file_model.filename).first()
            )
            if existing_file is not None:
                raise ValueError("File already exists")
                        
            file = self.pydantic_to_file(file_model)
            file.save(force_insert=True)
            return self.file_to_pydantic(file)

    def get_file_by_name(self, filename: str) -> Optional[FileModel]:
        with self.db.connection_context():
            try:
                file = self.model.get(self.model.filename == filename)
                return self.file_to_pydantic(file)
            except DoesNotExist:
                return None

    def delete_file(self, filename: str) -> int:
        with self.db.connection_context():
            # Get the collection
            file = self.get_file_by_name(filename)
            if not file:
                raise ValueError("Invalid collection/file")

            query = self.model.delete().where(self.model.filename == filename)
            return query.execute()

    def delete_all(self) -> int:
        with self.db.connection_context():
            query = self.model.delete()
            return query.execute()

    def get_all(self) -> List[FileModel]:
        with self.db.connection_context():
            return [self.file_to_pydantic(file) for file in self.model.select()]

    def insert_many(self, rows: List[FileModel]) -> None:
        with self.db.connection_context():
            data_to_insert = [self.pydantic_to_file(row) for row in rows]
            self.model.bulk_create(data_to_insert)

    def update_file(self, filename: str, **kwargs: dict) -> int:
        with self.db.connection_context():
            query = self.model.update(kwargs).where(self.model.filename == filename)
            return query.execute()

class ChatManager:
    def __init__(self, db: PostgresqlDatabase) -> None:
        
        class ChatMessage(Model):
            ai_message = TextField()
            human_message = TextField()
            sequence_number = IntegerField()
            namespace = CharField()

            class Meta:
                database = db
                
        self.model = ChatMessage
        db.connect(reuse_if_open=True)
        db.create_tables([ChatMessage], safe=True)
        self.db = db
        
    def add_message(self, namespace: str, ai_message: str, human_message: str) -> None:
        with self.db.connection_context():
            last_message = self.model.select().where(self.model.namespace == namespace).order_by(
                self.model.sequence_number.desc()).first()

            if last_message is None:
                sequence_number = 0
            else:
                sequence_number = last_message.sequence_number + 1

            self.model.create(namespace=namespace, ai_message=ai_message, human_message=human_message,
                                sequence_number=sequence_number)

    def retrieve_all_messages(self, namespace: str):
        with self.db.connection_context():
            query = self.model.select().where(self.model.namespace == namespace).order_by(self.model.sequence_number)
            return [(row.human_message, row.ai_message) for row in query]





def test_file_manager():
    db = SqliteDatabase("test.db")

    file_manager = FileDBManager(db)

    file_manager.delete_all()


    file_content = "This is a test file content."
    file_bytes = b"Some bytes for testing"
    file = FileModel(
        filename="Test File",
        filetype=".txt",
        vector_ids=["1", "2", "3"],
        file_content=file_content,
        file_bytes=file_bytes,
        description="Desctiption"
    )
    file_manager.add_file(file)

    file = file_manager.get_file_by_name("Test File")
    print(f"Retrieved File: {file}")

    file_manager.delete_file(file.filename)
    print(f"Deleted File: {file.filename}")

    file = file_manager.get_file_by_name(file.filename)
    print(f"File after deletion (should be None): {file}")

    print(file_manager.get_all())


if __name__ == "__main__":
    test_file_manager()

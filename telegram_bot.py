import os
import tempfile
import openai
import logging
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from typing import Optional, Dict, Any
from knowledge_manager import KnowledgeManager
from database import ChatManager



logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperTranscriber:
    def __init__(self, api_key: str, model_name: str = "whisper-1"):
        self.model_name = model_name
        self.api_key = api_key

    def transcribe(self, audio_file_path: str) -> Optional[Dict[str, Any]]:
        try:
            with open(audio_file_path, "rb") as audio_file:
                transcript = openai.Audio.transcribe(self.model_name, audio_file, api_key=self.api_key)
            return transcript
        except Exception as e:
            logger.error(f"An error occurred while transcribing voice: {e}")
            return None

class AIResponder:
    def __init__(self, knowledge_manager: KnowledgeManager, chatmanager: ChatManager) -> None:
        self.knowledge_manager = knowledge_manager
        self.chatmanager = chatmanager
        
    async def generate_response(self, text: str, user_id: str) -> str:
        # Detect the language of the incoming text        
        # Generate the AI response based on the detected language
        ai_response = await self.knowledge_manager.chat(text, self.chatmanager.retrieve_all_messages(user_id))
        
        # Store the message and response in the chat manager
        self.chatmanager.add_message(user_id, ai_response, text)
        
        return ai_response

class TelegramBot:
    def __init__(self, token: str, transcriber: WhisperTranscriber, responder: 'AIResponder'):
        self.application = Application.builder().token(token).build()
        self.transcriber = transcriber
        self.responder = responder
        self._setup_handlers()
        
    def _setup_handlers(self) -> None:
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_update))
        self.application.add_handler(MessageHandler(filters.VOICE, self.handle_voice))

    async def handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        audio_file_path = await self._download_voice(update)
        print(audio_file_path)
        if not audio_file_path:
            await update.message.reply_text("Sorry, I couldn't download the voice message.")
            return

        transcription = self.transcriber.transcribe(audio_file_path)
        
        if not transcription:
            await update.message.reply_text("Sorry, I couldn't understand the voice message.")
            return
        
        await self._generate_and_send_response(transcription.get("text", ""), update)

    async def handle_update(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message and update.message.text:
            await self._generate_and_send_response(update.message.text, update)

    async def _generate_and_send_response(self, user_message: str, update: Update) -> None:
        try:
            # Extract user_id from the Update object
            user_id = update.message.from_user.id
            # Generate a response using the AIResponder
            ai_response = await self.responder.generate_response(user_message, user_id)
            # Send the generated response back to the user
            await update.message.reply_text(ai_response)
        except Exception as e:
            logger.error(f"An error occurred while generating AI response: {e}")
            await update.message.reply_text("Sorry, I couldn't process that.")


    async def _download_voice(self, update: Update) -> str:
        try:
            voice = update.message.voice
            if not voice:
                return ""
            
            # Create a temporary file and get its path without deleting the file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.ogg')
            temp_file_path = temp_file.name

            # Download the voice message
            voice_file = await voice.get_file()
            await voice_file.download_to_drive(custom_path=temp_file_path)
            
            return temp_file_path
                
        except Exception as e:
            logger.error(f"An error occurred while downloading voice message: {e}")
            return ""

    def run(self) -> None:
        self.application.run_polling(allowed_updates=Update.ALL_TYPES, timeout=2)

        
        
if __name__ == '__main__':
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.chat_models import ChatOpenAI
    from peewee import PostgresqlDatabase
    import langchain
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    UNSTRUCTURED_API_KEY = os.getenv("UNSTRUCTURED_API_KEY")
    DB_NAME = os.getenv("DB_NAME")
    DATABASE_PORT = os.getenv("DATABASE_PORT", "5432")
    DATABASE_HOST = os.getenv("DATABASE_HOST")
    DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD")
    DATABASE_USER = os.getenv("DATABASE_USER")
    
    CON_STRING = F"postgresql://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DB_NAME}"
    langchain.verbose = True
    manager = KnowledgeManager(
        OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
        ChatOpenAI,
        {"openai_api_key" : OPENAI_API_KEY, "temperature" : 0},
        unstructured_api_key=UNSTRUCTURED_API_KEY,
        connection_string=CON_STRING
    )
    chat_manager = ChatManager(
        PostgresqlDatabase(
            DB_NAME,
            user=DATABASE_USER,
            password=DATABASE_PASSWORD,
            host=DATABASE_HOST,
            port=DATABASE_PORT,
            autorollback=True, 
            autoconnect=True
        )
    )
    whisper_transcriber = WhisperTranscriber(api_key=OPENAI_API_KEY)
    ai_responder = AIResponder(manager, chat_manager)
    telegram_bot = TelegramBot(token='', transcriber=whisper_transcriber, responder=ai_responder)
    telegram_bot.run()

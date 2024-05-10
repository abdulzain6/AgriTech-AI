# Agritechai Telegram Bot

The Agritechai Telegram Bot leverages advanced AI to answer agricultural-related questions via Telegram. Supporting both text and voice interactions, the bot provides accessible and interactive user experiences using state-of-the-art technologies.

## Features

- **Voice Interaction**: Engage with the bot using voice commands and receive voice responses.
- **Knowledge Base**: Powered by a comprehensive agricultural database for accurate answers.
- **Data Extraction**: Utilizes unstructured.io for data extraction from files.
- **Vector Database**: Uses pgvector for efficient vector data handling.
- **Database**: Managed with Supabase for robust backend services.
- **AI Powered**: Integrates OpenAI for enhanced understanding and response generation.

## Prerequisites

- Python 3.10+
- pip package manager
- Telegram bot token from BotFather

## Installation

Install necessary Python packages:

```bash
pip install -r requirements.txt

## Usage

Start the bot with:

```bash
python telegram_bot.py
```

## Configuration

Set the following environment variables in `telegram_bot.py`:

```python
OPENAI_API_KEY = ""  # Your OpenAI API key
UNSTRUCTURED_API_KEY = ""  # API key for unstructured.io
DB_NAME = ""  # Database name
DATABASE_PORT = 5432  # Database port (default for PostgreSQL)
DATABASE_HOST = ""  # Database host URL
DATABASE_PASSWORD = ""  # Database password
DATABASE_USER = ""  # Database username
TELEGRAM_TOKEN = ""  # Telegram bot token
```

## Contributing

Contributions are welcome. Please fork the repository and submit pull requests.

## Acknowledgments

- OpenAI for AI capabilities.
- LangChain, pgvector, and Supabase for data handling and database solutions.
- unstructured.io for file data extraction capabilities.
```

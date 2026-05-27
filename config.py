import os
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# API keys and endpoints
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Chroma / persistence
CHROMA_CLIENT_PATH = os.getenv("CHROMA_CLIENT_PATH", "Collections_new_2")
DEFAULT_COLLECTION_NAME = os.getenv("DEFAULT_COLLECTION_NAME", "text_Collection_3")

# Other defaults
RESULTS_FOLDER = os.getenv("RESULTS_FOLDER", "results")

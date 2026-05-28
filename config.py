import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Load .env if present. Search both the current working directory and the
# directory containing this file so Streamlit can start from a different path
# and still pick up project-local secrets.
load_dotenv(find_dotenv(usecwd=True), override=False)
load_dotenv(Path(__file__).resolve().with_name(".env"), override=False)

# API keys and endpoints
# Groq is the default runtime.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "llama-3.2-90b-vision-preview")

# Optional alternatives kept here as commented examples for future use:
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
# TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Chroma / persistence
CHROMA_CLIENT_PATH = os.getenv("CHROMA_CLIENT_PATH", "Collections_new_2")
DEFAULT_COLLECTION_NAME = os.getenv("DEFAULT_COLLECTION_NAME", "text_Collection_3")

# Other defaults
RESULTS_FOLDER = os.getenv("RESULTS_FOLDER", "results")

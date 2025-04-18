import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env file from project root (adjust path if needed)
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model/Service Configurations
WHISPER_MODEL_SIZE = "base.en" # or tiny.en, small.en
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8" # or float32

LLAMA_INDEX_DIR = "/app/data/slide_text_data" # Path inside docker container
LLAMA_INDEX_PERSIST_DIR = "/app/vector_store_local" # Path inside docker container
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

GEMINI_MODEL_NAME = "gemini-1.5-flash"

MLFLOW_TRACKING_URI = "file:/app/mlruns" # Path inside docker container
MLFLOW_EXPERIMENT_NAME = "Classroom Captioning API Calls"

# Basic check
if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY environment variable not set!")
"""Configuration management for the RAG pipeline."""
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent

# API Configuration
# Google API key is only needed for LLM (Gemini), not for embeddings
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in .env file for LLM usage.")

# Model Configuration
# Using SentenceTransformer for local embeddings (no API calls needed)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.85"))

# Data Configuration
DATA_PATH = os.getenv("DATA_PATH", str(BASE_DIR / "data" / "Ayurveda Dataset"))
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", str(BASE_DIR / "vector_db"))

# Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Retrieval Configuration
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))

# Prompt Template
LLM_PROMPT_TEMPLATE = """You are an assistant for question-answering tasks with advanced analytical and reasoning capabilities.
Use the following context to answer the question.
If you don't know the answer, try to think of it without context.

Question: {question}

Context: {context}

Answer:"""

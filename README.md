# Ayurveda RAG Chatbot

**Download dataset from [Kaggle - Ayurveda Texts (English)](https://www.kaggle.com/datasets/rcratos/ayurveda-texts-english/data)**

A Retrieval-Augmented Generation (RAG) chatbot for answering questions about Ayurveda using SentenceTransformer for embeddings, Google's Gemini models for generation, and FAISS vector database.

## Features

- 📚 **Document Processing**: Loads and processes PDF and text files from the Ayurveda dataset
- 🔍 **Semantic Search**: Uses SentenceTransformer (all-MiniLM-L6-v2) for local, efficient semantic document retrieval
- 🤖 **Advanced RAG**: Implements advanced RAG chain with configurable prompts
- 💰 **Cost-Effective**: Local embeddings (no API costs for embeddings, only for LLM generation)
- 🌐 **Multiple Interfaces**: 
  - Streamlit web UI for interactive queries
  - FastAPI REST API for programmatic access
- ⚙️ **Configurable**: Centralized configuration management via environment variables

## Architecture

```
Data Sources (PDF/TXT) → Load Documents → Chunk Documents → Build Vector DB
                                                                    ↓
User Query → Retrieve Relevant Docs → Generate Answer with LLM → Response
```

## Project Structure

```
ayur-rag-chatbot/
├── app/
│   ├── fastapi_app.py      # FastAPI REST API
│   └── streamlit_app.py    # Streamlit web interface
├── chatbot/
│   ├── chain.py            # RAG chain orchestration
│   └── generate_answer.py  # Answer generation with LLM
├── data/
│   └── Ayurveda Dataset/   # Dataset directory
│       ├── ayurveda_books/
│       └── ayurveda_texts/
├── ingest/
│   ├── load_documents.py   # Document loading
│   ├── chunk_documents.py  # Document chunking
│   ├── build_vector_db.py  # Vector database creation
│   └── main.py             # Main ingestion script
├── retriever/
│   └── query_retriever.py  # Document retrieval
├── config.py               # Configuration management
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Google API Configuration
GOOGLE_API_KEY=your_google_api_key_here

# Model Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=gemini-2.5-flash
LLM_TEMPERATURE=0.3
LLM_TOP_P=0.85

# Data Configuration
DATA_PATH=data/Ayurveda Dataset
VECTOR_DB_PATH=vector_db

# Chunking Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Retrieval Configuration
RETRIEVAL_K=5
```

### 3. Build Vector Database

Run the ingestion pipeline to process documents and create the vector database:

```bash
python ingest/main.py
```

This will:
1. Load all PDF and text files from the data directory
2. Split them into chunks
3. Create embeddings using Google's embedding model
4. Save the vector database to `vector_db/`

**Note**: This process may take some time depending on the size of your dataset.

## Usage

### Streamlit Web Interface

Launch the interactive web interface:

```bash
streamlit run app/streamlit_app.py
```

Features:
- Interactive chat interface
- Source citations for answers
- Configurable model and retrieval parameters
- Chat history

### FastAPI REST API

Start the API server:

```bash
uvicorn app.fastapi_app:app --reload
```

Query the API:

```bash
curl "http://localhost:8000/ask?q=What%20are%20the%20three%20doshas?"
```

Response:
```json
{
  "answer": "The three doshas are Vata, Pitta, and Kapha..."
}
```

### Python API

```python
from chatbot.chain import rag

# Ask a question
response = rag("What are the three doshas in Ayurveda?")
print(response)
```

## Configuration

All configuration is managed through environment variables (see `.env` file):

- **GOOGLE_API_KEY**: Your Google API key for Gemini LLM models (not needed for embeddings)
- **EMBEDDING_MODEL**: SentenceTransformer model to use (default: `sentence-transformers/all-MiniLM-L6-v2`)
- **LLM_MODEL**: Gemini model to use (e.g., `gemini-2.5-flash`, `gemini-2.5-pro`)
- **LLM_TEMPERATURE**: Temperature for LLM responses (0.0-1.0)
- **LLM_TOP_P**: Top-p sampling parameter
- **CHUNK_SIZE**: Size of document chunks (characters)
- **CHUNK_OVERLAP**: Overlap between chunks (characters)
- **RETRIEVAL_K**: Number of documents to retrieve for each query

## Components

### Document Loading (`ingest/load_documents.py`)
- Supports PDF and text files
- Recursively searches data directory
- Handles encoding and errors gracefully

### Document Chunking (`ingest/chunk_documents.py`)
- Uses RecursiveCharacterTextSplitter
- Configurable chunk size and overlap
- Preserves document metadata

### Vector Database (`ingest/build_vector_db.py`)
- Creates FAISS vector database
- Uses SentenceTransformer (all-MiniLM-L6-v2) for local embeddings
- No API calls needed for embeddings (runs locally)
- Saves database for reuse

### Retrieval (`retriever/query_retriever.py`)
- Semantic search using FAISS
- Returns top-k most relevant documents
- Includes error handling and validation

### Answer Generation (`chatbot/generate_answer.py`)
- Advanced prompt template
- Configurable model parameters
- Context-aware responses

## Troubleshooting

### Vector Database Not Found

If you see an error about the vector database not being found:

```bash
python ingest/main.py
```

### API Key Issues

Ensure your `GOOGLE_API_KEY` is set correctly in the `.env` file.

### Import Errors

Make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```

## Example Questions

Here are some example questions you can ask the RAG chatbot:

- What are the 3 secondary supports of life?
- Reduced Therapy
- How to deal with heart disease
- What asanas should I do if I have imbalance in strength of my left and right arm?
- How can I make my tea better?
- Teach me in detail about dosas and how do I find mine?
- I have a Vatta-Pita imbalance. Help me

## License

This project is for educational purposes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

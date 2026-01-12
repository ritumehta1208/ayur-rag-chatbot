from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from pathlib import Path
import config

def retrieve(query, k=None):
    """
    Retrieve relevant documents from the vector database.
    
    Args:
        query: The search query
        k: Number of documents to retrieve (defaults to config.RETRIEVAL_K)
    
    Returns:
        List of relevant documents with metadata
    """
    if k is None:
        k = config.RETRIEVAL_K
    
    # Check if vector database exists
    vector_db_path = Path(config.VECTOR_DB_PATH)
    if not vector_db_path.exists():
        raise FileNotFoundError(
            f"Vector database not found at {vector_db_path}. "
            "Please run the ingestion pipeline first (python ingest/main.py)"
        )
    
    # Initialize embeddings using SentenceTransformer
    embedder = SentenceTransformerEmbeddings(
        model_name=config.EMBEDDING_MODEL
    )
    
    # Load vector database
    try:
        db = FAISS.load_local(
            str(vector_db_path),
            embedder,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load vector database: {str(e)}")
    
    # Perform similarity search
    try:
        results = db.similarity_search(query, k=k)
        return results
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve documents: {str(e)}")

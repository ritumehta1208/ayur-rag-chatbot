from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from pathlib import Path
import config

def build_db(chunks):
    """
    Build and save FAISS vector database from document chunks.
    
    Args:
        chunks: List of document chunks to embed and store
    """
    if not chunks:
        raise ValueError("No chunks provided to build vector database")
    
    print(f"Building vector database from {len(chunks)} chunks...")
    
    # Initialize embeddings using SentenceTransformer

    embedder = SentenceTransformerEmbeddings(
        model_name=config.EMBEDDING_MODEL
    )
    
    # Create vector database
    print("Creating embeddings (this may take a while)...")
    db = FAISS.from_documents(chunks, embedder)
    
    # Ensure directory exists
    vector_db_path = Path(config.VECTOR_DB_PATH)
    vector_db_path.mkdir(parents=True, exist_ok=True)
    
    # Save vector database
    print(f"Saving vector database to {vector_db_path}...")
    db.save_local(str(vector_db_path))
    print(f"Vector database saved successfully!")
    
    return db

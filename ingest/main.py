"""Main ingestion script to orchestrate the RAG pipeline."""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest.load_documents import load_docs
from ingest.chunk_documents import chunk_docs
from ingest.build_vector_db import build_db
import config

def main():
    """Orchestrate the full ingestion pipeline."""
    print("=" * 60)
    print("Starting RAG Pipeline Ingestion")
    print("=" * 60)
    
    try:
        # Step 1: Load documents
        print("\n[Step 1/3] Loading documents...")
        docs = load_docs()
        
        if not docs:
            print("No documents found. Please check your data path.")
            return
        
        # Step 2: Chunk documents
        print("\n[Step 2/3] Chunking documents...")
        chunks = chunk_docs(docs)
        
        if not chunks:
            print("No chunks created. Please check your documents.")
            return
        
        # Step 3: Build vector database
        print("\n[Step 3/3] Building vector database...")
        build_db(chunks)
        
        print("\n" + "=" * 60)
        print("Ingestion pipeline completed successfully!")
        print("=" * 60)
        print(f"\nVector database saved to: {config.VECTOR_DB_PATH}")
        print(f"You can now use the RAG pipeline to answer questions.")
        
    except Exception as e:
        print(f"\nError during ingestion: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

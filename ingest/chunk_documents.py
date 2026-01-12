from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

def chunk_docs(docs):
    """
    Split documents into chunks for embedding.
    
    Args:
        docs: List of documents to chunk
    
    Returns:
        List of document chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"Split {len(docs)} documents into {len(chunks)} chunks")
    return chunks

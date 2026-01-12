"""Streamlit web interface for the RAG chatbot."""
import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from chatbot.chain import rag
from retriever.query_retriever import retrieve
import config

# Page configuration
st.set_page_config(
    page_title="Ayurveda RAG Chatbot",
    page_icon="🌿",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model" not in st.session_state:
    st.session_state.model = config.LLM_MODEL

def main():
    """Main Streamlit application."""
    # Title and description
    st.title("🌿 Ayurveda RAG Chatbot")
    st.markdown("Ask questions about Ayurveda and get answers based on the knowledge base.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_options = ["gemini-2.5-flash", "gemini-2.5-flash-pro", "gemini-2.5-flash-lite"]
        selected_model = st.selectbox(
            "LLM Model",
            model_options,
            index=model_options.index(config.LLM_MODEL) if config.LLM_MODEL in model_options else 0
        )
        
        if selected_model != st.session_state.model:
            st.session_state.model = selected_model
            # Update config temporarily
            config.LLM_MODEL = selected_model
        
        # Retrieval parameters
        st.subheader("Retrieval Settings")
        k_value = st.slider(
            "Number of documents to retrieve",
            min_value=1,
            max_value=10,
            value=config.RETRIEVAL_K,
            help="More documents provide more context but may include irrelevant information"
        )
        
        # Display vector DB status
        st.subheader("System Status")
        from pathlib import Path
        vector_db_path = Path(config.VECTOR_DB_PATH)
        if vector_db_path.exists():
            st.success("✅ Vector database loaded")
        else:
            st.error("❌ Vector database not found")
            st.info("Please run: `python ingest/main.py` to build the vector database")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.text(f"Source {i}: {source[:200]}...")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about Ayurveda..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Retrieve relevant documents
                    retrieved_docs = retrieve(prompt, k=k_value)
                    
                    # Generate answer
                    response = rag(prompt)
                    
                    # Display response
                    st.markdown(response)
                    
                    # Show sources
                    if retrieved_docs:
                        with st.expander("📚 Sources"):
                            for i, doc in enumerate(retrieved_docs[:3], 1):
                                source_text = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                                st.text(f"Source {i}:")
                                st.text(source_text)
                                st.divider()
                        
                        # Store sources in message
                        sources = [doc.page_content[:200] for doc in retrieved_docs[:3]]
                    else:
                        sources = []
                    
                    # Add assistant response to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources
                    })
                    
                except FileNotFoundError as e:
                    error_msg = f"❌ {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

if __name__ == "__main__":
    main()

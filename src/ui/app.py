"""
Streamlit UI for Local RAG Chatbot
"""

import streamlit as st
from pathlib import Path
import sys

# CRITICAL: Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.main import RAGPipeline
from src.config import settings
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Page configuration
st.set_page_config(
    page_title="Local RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        font-size: 0.9rem;
    }
    .chat-message {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .bot-message {
        background-color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    """Get or create RAG pipeline (cached)"""
    return RAGPipeline()


def main():
    """Main application"""
    
    # Header
    st.markdown('<p class="main-header">🤖 Local RAG Chatbot</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = get_pipeline()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model info
        st.info(f"**Model:** {settings.llm_model}")
        st.info(f"**Embedding:** {settings.embedding_model}")
        
        # Document count
        doc_count = st.session_state.pipeline.get_document_count()
        st.metric("Documents in Store", doc_count)
        
        st.markdown("---")
        
        # Document upload
        st.header("📁 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF, TXT, MD, or DOCX files",
            type=["pdf", "txt", "md", "docx"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            save_dir = Path("data/sample_documents")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            for file in uploaded_files:
                file_path = save_dir / file.name
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                
                with st.spinner(f"Ingesting {file.name}..."):
                    chunks = st.session_state.pipeline.ingest_document(str(file_path))
                    st.success(f"✅ {file.name}: {chunks} chunks added")
        
        st.markdown("---")
        
        # Clear store
        if st.button("🗑️ Clear Document Store", use_container_width=True):
            st.session_state.pipeline.clear_store()
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("**Supported Formats:** PDF, TXT, MD, DOCX")
    
    # Chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if "sources" in message and message["sources"]:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>Source {i}:</strong> {source['source']}<br>
                            <em>{source['content']}</em>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.pipeline.query(prompt)
                
                if response["success"]:
                    st.markdown(response["answer"])
                    
                    if response["sources"]:
                        with st.expander("📚 View Sources"):
                            for i, source in enumerate(response["sources"], 1):
                                st.markdown(f"""
                                <div class="source-box">
                                    <strong>Source {i}:</strong> {source['source']}<br>
                                    <em>{source['content']}</em>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response["sources"]
                    })
                else:
                    st.error(response["answer"])
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"]
                    })


if __name__ == "__main__":
    main()
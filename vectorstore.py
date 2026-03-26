import os
from typing import List, Optional
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "rag_docs"

def get_embeddings():
    """Returns Ollama embeddings with nomic-embed-text model."""
    return OllamaEmbeddings(model="nomic-embed-text")

def create_vectorstore(chunks: List[Document]) -> Chroma:
    """Builds and persists the vectorstore from chunks."""
    try:
        print(f"Embedding {len(chunks)} chunks...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=get_embeddings(),
            persist_directory=CHROMA_PATH,
            collection_name=COLLECTION_NAME
        )
        # Persistent storage is automatic in newer Chroma versions, 
        # but Chroma.from_documents handles it.
        print("Vectorstore saved.")
        return vectorstore
    except Exception as e:
        print(f"Error creating vectorstore: {e}")
        raise

def load_vectorstore() -> Optional[Chroma]:
    """Loads the existing ChromaDB from disk."""
    if not os.path.exists(CHROMA_PATH):
        print(f"Warning: ChromaDB path '{CHROMA_PATH}' does not exist.")
        return None
        
    try:
        print(f"Loading vectorstore from {CHROMA_PATH}...")
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embeddings(),
            collection_name=COLLECTION_NAME
        )
        return vectorstore
    except Exception as e:
        print(f"Error loading vectorstore: {e}")
        return None

if __name__ == "__main__":
    # Test block
    db = load_vectorstore()
    if db:
        print("Successfully loaded existing DB.")

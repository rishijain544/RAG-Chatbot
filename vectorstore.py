from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import os
import shutil
from typing import List, Optional

# Constants
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "neural_docs"  # New name to force fresh start

def create_vectorstore(chunks: List[Document], path: str = CHROMA_PATH, collection: str = COLLECTION_NAME):
    """Creates a vectorstore from document chunks using HuggingFace embeddings."""
    try:
        # Clear existing to prevent source overlap
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
            
        # Initialize HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        print(f"Embedding {len(chunks)} chunks into {path}...")
        
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=path,
            collection_name=collection
        )
        
        print(f"Vectorstore saved to {path}.")
        return vectorstore
    except Exception as e:
        print(f"Error creating vectorstore: {e}")
        raise

def load_vectorstore(path: str = CHROMA_PATH, collection: str = COLLECTION_NAME) -> Optional[object]:
    """Loads the existing Chroma DB from specified path."""
    if not os.path.exists(path):
        return None
        
    try:
        # Initialize the same HuggingFace embeddings model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        vectorstore = Chroma(
            persist_directory=path,
            embedding_function=embeddings,
            collection_name=collection
        )
        
        return vectorstore
    except Exception as e:
        print(f"Error loading vectorstore from {path}: {e}")
        return None

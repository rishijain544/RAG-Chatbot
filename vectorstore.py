from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import os
import shutil
from typing import List, Optional

# Constants
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "neural_docs"  # New name to force fresh start
MAX_CHUNKS = 500  # Scale-out: Limit memory footprint per user

def create_vectorstore(chunks: List[Document], path: str = CHROMA_PATH, collection: str = COLLECTION_NAME):
    """Creates a vectorstore from document chunks using HuggingFace embeddings."""
    try:
        # Resolve absolute path for consistency
        abs_path = os.path.abspath(path)
        
        # Clear existing to prevent source overlap
        if os.path.exists(abs_path):
            try:
                shutil.rmtree(abs_path, ignore_errors=True)
                # Double check deletion
                if os.path.exists(abs_path):
                    import time
                    time.sleep(1) # Wait for file handles to release
                    shutil.rmtree(abs_path, ignore_errors=True)
            except Exception as rmtree_err:
                print(f"Warning: Could not fully delete {abs_path}: {rmtree_err}")
            
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            
        # Scale-out: Limit chunks to keep memory low
        if len(chunks) > MAX_CHUNKS:
            print(f"Limiting chunks from {len(chunks)} to {MAX_CHUNKS}")
            chunks = chunks[:MAX_CHUNKS]
            
        # Initialize HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        print(f"Embedding {len(chunks)} chunks into {abs_path}...")
        
        # We use a context-independent initialization to avoid singleton issues
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=abs_path,
            collection_name=collection
        )
        
        print(f"Vectorstore saved to {abs_path}.")
        return vectorstore
    except Exception as e:
        print(f"Error creating vectorstore at {path}: {e}")
        # Log more details if it's a chromadb error
        if "chromadb" in str(type(e)).lower():
            print("Hint: This might be a database lock or corruption issue.")
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

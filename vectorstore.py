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

import chromadb
from chromadb.config import Settings

def create_vectorstore(chunks: List[Document], path: str = CHROMA_PATH, collection: str = COLLECTION_NAME):
    """Creates a vectorstore using an explicit PersistentClient to avoid lock issues."""
    try:
        abs_path = os.path.abspath(path)
        
        # Scale-out: Limit chunks
        if len(chunks) > MAX_CHUNKS:
            chunks = chunks[:MAX_CHUNKS]

        # Initialize HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        # Explicit Chroma Client Management
        print(f"Initializing PersistentClient at {abs_path}...")
        client = chromadb.PersistentClient(path=abs_path, settings=Settings(allow_reset=True))
        
        # Try to delete if existing to ensure metadata consistency
        try:
            client.delete_collection(collection)
        except: pass
        
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=abs_path,
            collection_name=collection,
            client=client # Pass explicit client
        )
        
        print(f"Vectorstore ready at {abs_path}.")
        return vectorstore
    except Exception as e:
        print(f"Deep Failure creating vectorstore at {path}: {e}")
        # Final fallback: Try a slightly randomized path if it's a lock issue
        if "chromadb" in str(type(e)).lower() or "InternalError" in str(e):
            try:
                import time
                new_path = f"{abs_path}_{int(time.time())}"
                print(f"🔄 Retrying with fresh path: {new_path}")
                return create_vectorstore(chunks, path=new_path, collection=collection)
            except: pass
        raise

def load_vectorstore(path: str = CHROMA_PATH, collection: str = COLLECTION_NAME) -> Optional[object]:
    """Loads the existing Chroma DB from specified path."""
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        return None
        
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        client = chromadb.PersistentClient(path=abs_path)
        
        vectorstore = Chroma(
            persist_directory=abs_path,
            embedding_function=embeddings,
            collection_name=collection,
            client=client
        )
        
        return vectorstore
    except Exception as e:
        print(f"Error loading vectorstore from {path}: {e}")
        return None

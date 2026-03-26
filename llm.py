import os
import streamlit as st
from typing import Union
from langchain_groq import ChatGroq
from langchain_ollama import OllamaLLM
from langchain_core.language_models import BaseLanguageModel

def get_groq_api_key():
    # Try Streamlit secrets first (for cloud deployment)
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
        
    # Fall back to .env for local development
    from dotenv import load_dotenv
    load_dotenv()
    return os.getenv("GROQ_API_KEY")

def get_llm(provider: str) -> BaseLanguageModel:
    """Returns a LangChain LLM object (Groq or Ollama) based on provider string."""
    try:
        if provider == "groq":
            api_key = get_groq_api_key()
            if not api_key or api_key == "your_groq_api_key_here":
                raise ValueError("GROQ_API_KEY is missing or contains the default placeholder.")
                
            print(f"Loading Groq LLM: llama-3.3-70b-versatile")
            return ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=api_key,
                temperature=0.1,
                max_tokens=2048
            )
            
        elif provider == "ollama":
            # Quick check if Ollama is running
            import requests
            try:
                requests.get("http://localhost:11434/api/tags", timeout=2)
            except Exception:
                raise ConnectionError("Ollama server is not detected at http://localhost:11434. Please ensure Ollama is running.")

            print(f"Loading Ollama LLM: mistral")
            return OllamaLLM(
                model="mistral",
                temperature=0.1,
                num_predict=2048
            )
            
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
            
    except Exception as e:
        print(f"Error loading LLM ({provider}): {e}")
        raise

if __name__ == "__main__":
    # Test block
    try:
        # Note: Groq might fail if API key is not set
        # llm = get_llm("groq")
        pass
    except Exception as e:
        print(e)

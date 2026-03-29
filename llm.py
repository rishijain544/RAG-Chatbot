import os
import streamlit as st
from langchain_groq import ChatGroq

def get_groq_api_key() -> str:
    """Retrieves the Groq API key from streamlit secrets (cloud) or .env (local)."""
    try:
        # Check Streamlit Cloud secrets first
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        # Fallback to .env for local development
        from dotenv import load_dotenv
        load_dotenv()
        return os.getenv("GROQ_API_KEY")

def get_llm(provider: str):
    """
    Returns a LangChain LLM object (Groq). 
    Note: Only Groq is supported in this version for cloud reliability.
    """
    try:
        if provider == "groq":
            api_key = get_groq_api_key()
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in Streamlit Secrets or .env file!")
                
            print(f"Loading Groq LLM: llama-3.1-8b-instant")
            return ChatGroq(
                api_key=api_key,
                model="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=2048,
                max_retries=6 # Exponential backoff handles rate limits
            )
        else:
            raise ValueError(f"Unknown provider: {provider}. Currently only 'groq' is supported.")
            
    except Exception as e:
        print(f"Error loading LLM: {e}")
        raise

if __name__ == "__main__":
    # Test block
    try:
        # This will work locally if .env has GROQ_API_KEY
        # llm = get_llm("groq")
        # print("LLM initialized successfully.")
        pass
    except Exception as e:
        print(f"Test failure: {e}")

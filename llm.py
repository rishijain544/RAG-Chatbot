import os
from typing import Union
from dotenv import load_dotenv
from langchain_groq import ChatGroq
def get_llm(provider: str = "groq") -> BaseLanguageModel:
    """Returns a LangChain LLM object (Groq) based on provider string."""
    try:
        provider = provider.lower()
        # Currently only supporting Groq for cloud reliability
        if provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key or api_key == "your_groq_api_key_here":
                raise ValueError("GROQ_API_KEY is missing or contains the default placeholder in .env file.")
                
            print(f"Loading Groq LLM: llama-3.3-70b-versatile")
            return ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=api_key,
                temperature=0.1,
                max_tokens=2048
            )
        else:
            raise ValueError(f"Unknown or unsupported LLM provider: {provider}")
            
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

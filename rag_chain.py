from typing import Dict, List, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_core.language_models import BaseLanguageModel

def get_rag_chain(vectorstore: Chroma, llm: BaseLanguageModel):
    """
    Creates a RAG chain that accepts vectorstore and llm.
    Returns: A runnable chain producing results and source documents.
    """
    try:
        # Retriever: MMR search for diversity
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8, "fetch_k": 20}
        )
        
        # Expert prompt template
        prompt_template = """
You are an expert news analyst and research assistant.
Your job is to provide detailed, accurate answers based on the provided news articles context.

INSTRUCTIONS:
- Answer in detail with all relevant facts, names, dates, and numbers
- Use bullet points for lists of facts
- If multiple articles cover the topic, synthesize all information
- Always mention specific details like player names, scores, dates
- If asked for summary, give a comprehensive paragraph summary
- If the context lacks information, say exactly what IS available
- Never make up information not present in the context

CONTEXT FROM NEWS ARTICLES:
{context}

USER QUESTION: {question}

DETAILED ANSWER:
"""
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Construct the chain
        # Using LCEL (LangChain Expression Language)
        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # To include source documents, we need a slightly more complex return structure 
        # or handle retrieved docs manually.
        
        def run_chain(question: str) -> Dict[str, Any]:
            """Explicit wrapper to run chain and return results + sources."""
            docs = retriever.invoke(question)
            context_text = format_docs(docs)
            
            # Use chain, but manually pass the already-retrieved context for consistency
            chain = (prompt | llm | StrOutputParser())
            result = chain.invoke({
                "context": context_text,
                "question": question
            })
            
            return {
                "result": result,
                "source_documents": docs
            }
        
        return run_chain
        
    except Exception as e:
        print(f"Error creating RAG chain: {e}")
        raise

if __name__ == "__main__":
    # Test block
    pass

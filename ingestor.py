import sys
import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
import re

# Ensure UTF-8 output even on Windows
if sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def ingest_urls_as_corpus(urls: List[str]) -> str:
    """
    Accepts a list of URLs, fetches content, and returns a single concatenated 
    string of all extracted news content.
    """
    corpus = []
    
    # Modern Browser Headers (Chrome 124)
    base_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }

    session = requests.Session()
    session.headers.update(base_headers)

    for url in urls:
        url = url.strip()
        if not url: continue
            
        print(f"🌐 Fetching URL: {url}...")
        try:
            # Warm-up sequence for restricted sites (NDTV, Cricbuzz)
            if any(domain in url for domain in ["ndtv.com", "ndtvimg.com", "cricbuzz.com"]):
                try:
                    base_url = "https://" + url.split("/")[2]
                    session.get(base_url, timeout=10)
                    import time
                    time.sleep(0.5)
                except: pass

            response = session.get(url, timeout=15)
            
            # Simple fallback for 403
            if response.status_code == 403:
                session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; bingbot/2.0; +http://www.bing.com/bingbot.htm)"})
                response = session.get(url, timeout=15)

            if response.status_code != 200: continue
            
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "ads", "iframe"]):
                tag.decompose()
            
            extracted_text = ""
            # Priority 1: Main content containers
            content_containers = soup.find_all(["article", "main", "div", "section"], 
                                             class_=re.compile(r'content|article|body|story|post|entry|sp-cn|ins_storybody|text-base|cb-nws-dtl-itms', re.I))
            if content_containers:
                body_text = "\n".join([c.get_text(separator=" ", strip=True) for c in content_containers if len(c.get_text()) > 100])
                extracted_text += body_text
            
            # Priority 2: Full body cleanup
            if not extracted_text.strip():
                extracted_text = soup.body.get_text(separator=" ", strip=True) if soup.body else ""
            
            # Clean and append to corpus
            full_text = re.sub(r'\n+', '\n', extracted_text).strip()
            full_text = re.sub(r' +', ' ', full_text)
            
            if len(full_text) > 100:
                print(f"✅ SUCCESS: Extracted {len(full_text)} chars from {url}")
                corpus.append(f"ARTICLE SOURCE: {url}\nCONTENT:\n{full_text}\n--- END ARTICLE ---\n")
            
        except Exception as e:
            print(f"❌ ERROR: Processing {url}: {e}")
            
    return "\n".join(corpus)

def ingest_urls(urls: List[str]) -> List[Document]:
    """Legacy wrapper for RAG logic compatibility."""
    corpus_text = ingest_urls_as_corpus(urls)
    if not corpus_text: return []
    return [Document(page_content=corpus_text, metadata={"source": "Corpus"})]

if __name__ == "__main__":
    # Test block
    test_urls = ["https://www.google.com"]
    chunks = ingest_urls(test_urls)

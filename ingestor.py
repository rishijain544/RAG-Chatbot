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

def ingest_urls(urls: List[str]) -> List[Document]:
    """
    Accepts a list of URLs, fetches content using requests (simulating browser),
    extracts clean text using BeautifulSoup, and splits into chunks.
    """
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", "! ", "? ", " "]
    )

    # Modern Browser Headers (Chrome 124)
    base_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
    }

    session = requests.Session()
    session.headers.update(base_headers)

    for url in urls:
        url = url.strip()
        if not url:
            continue
            
        print(f"🌐 Fetching URL: {url}...")
        try:
            # Step 1: Warm-up sequence for restricted sites (NDTV, Cricbuzz)
            if any(domain in url for domain in ["ndtv.com", "ndtvimg.com", "cricbuzz.com"]):
                print(f"🛡️ Detected restricted domain ({url}). Warming up session...")
                try:
                    # Get base domain for warm-up
                    base_url = "https://" + url.split("/")[2]
                    session.get(base_url, timeout=10)
                    import time
                    time.sleep(0.5)
                    session.headers.update({
                        "Referer": base_url,
                        "Sec-Fetch-Site": "same-origin"
                    })
                except:
                    pass

            # Step 2: Fetch content
            response = session.get(url, timeout=15)
            
            # Fallback if blocked (403)
            if response.status_code == 403:
                print(f"⚠️ Blocked (403). Retrying with Bingbot identity...")
                time.sleep(1)
                session.headers.update({
                    "User-Agent": "Mozilla/5.0 (compatible; bingbot/2.0; +http://www.bing.com/bingbot.htm)",
                    "Sec-Ch-Ua-Mobile": "?0",
                    "Sec-Ch-Ua-Platform": '"Windows"'
                })
                response = session.get(url, timeout=15)
                
            # Second fallback (Mobile)
            if response.status_code == 403:
                print(f"⚠️ Still blocked. Retrying with mobile identity...")
                time.sleep(1)
                session.headers.update({
                    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1",
                    "Sec-Ch-Ua-Mobile": "?1",
                    "Sec-Ch-Ua-Platform": '"iOS"'
                })
                response = session.get(url, timeout=15)

            if response.status_code != 200:
                print(f"❌ FAILED to extract content from {url} (Status: {response.status_code})")
                continue
            
            # Step 2: Extract text using BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Remove unwanted elements
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "ads", "iframe"]):
                tag.decompose()
            
            extracted_text = ""
            
            # NDTV Specific: Try Headline
            headline = soup.find(["h1", "h2"], class_=re.compile(r'sp-ttl|headline', re.I))
            if headline:
                extracted_text = headline.get_text(strip=True) + "\n\n"
            
            # Priority 1: Main content containers (including NDTV sp-cn and Cricbuzz text-base)
            content_containers = soup.find_all(["article", "main", "div", "section"], 
                                             class_=re.compile(r'content|article|body|story|post|entry|sp-cn|ins_storybody|text-base|cb-nws-dtl-itms', re.I))
            if content_containers:
                body_text = "\n".join([c.get_text(separator=" ", strip=True) for c in content_containers if len(c.get_text()) > 100])
                extracted_text += body_text
            
            # Priority 2: Semantic blocks or sections
            if not extracted_text.strip() or len(extracted_text) < 200:
                sections = soup.find_all(["section", "div"], id=re.compile(r'content|article|story|main', re.I))
                if sections:
                    extracted_text = "\n".join([s.get_text(separator=" ", strip=True) for s in sections])
            
            # Priority 3: all <p> tags (more aggressive)
            if not extracted_text.strip() or len(extracted_text) < 200:
                p_tags = soup.find_all("p")
                if p_tags:
                    # Filter out short teaser paragraphs
                    long_p = [p.get_text(separator=" ", strip=True) for p in p_tags if len(p.get_text()) > 20]
                    extracted_text = "\n".join(long_p)
            
            # Priority 4: all <h1><h2><h3> + <p> tags
            if not extracted_text.strip():
                tags = soup.find_all(["h1", "h2", "h3", "p"])
                if tags:
                    extracted_text = "\n".join([t.get_text(separator=" ", strip=True) for t in tags])
            
            # Priority 5: full body text as last resort
            if not extracted_text.strip() and soup.body:
                extracted_text = soup.body.get_text(separator=" ", strip=True)
            
            # Clean text
            full_text = re.sub(r'\n+', '\n', extracted_text).strip()
            full_text = re.sub(r' +', ' ', full_text)
            
            # Validation
            if len(full_text) < 100:
                print(f"⚠️ WARNING: Extracted text too short from {url} (Length: {len(full_text)})")
                continue
                
            # Step 3: Create documents and split
            doc = Document(page_content=full_text, metadata={"source": url})
            chunks = text_splitter.split_documents([doc])
            
            # Add detailed metadata to every chunk
            total_chunks = len(chunks)
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_index": i,
                    "total_chunks": total_chunks,
                    "content_preview": chunk.page_content[:100].replace("\n", " ")
                })
                all_chunks.append(chunk)
            
            print(f"✅ SUCCESS: Extracted {len(full_text)} chars, created {len(chunks)} chunks from {url}")
            
        except Exception as e:
            print(f"❌ ERROR: Could not extract content from {url}. Error: {e}")
            
    print(f"\n🚀 Total chunks created across all URLs: {len(all_chunks)}")
    return all_chunks

if __name__ == "__main__":
    # Test block
    test_urls = ["https://www.google.com"]
    chunks = ingest_urls(test_urls)

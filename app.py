# Place robot image as "background.jpg" in project folder
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.getLogger('tensorflow').setLevel(logging.ERROR)
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import os
import time
import json
import uuid
import base64
import io
import re
from datetime import datetime
# Added for Tier 1 features
import random

# --- RAG Logic Imports ---
from ingestor import ingest_urls
from vectorstore import create_vectorstore, load_vectorstore
from llm import get_llm
from rag_chain import get_rag_chain

# --- Page Configuration ---
st.set_page_config(
    page_title="🤖 RAG Assistant AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Background Image Setup ---
def get_base64_image(image_path):
    try:
        if os.path.exists(image_path):
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
    except Exception: return None
    return None

bg_image = get_base64_image("background.jpg")

# --- Constants & Storage ---
HISTORY_FILE = "chat_history.json"
SAVED_URLS_FILE = "saved_urls.json"

# --- Tier 1 Helper Functions ---

def load_saved_urls() -> dict:
    if os.path.exists(SAVED_URLS_FILE):
        try:
            with open(SAVED_URLS_FILE, "r") as f:
                return json.load(f)
        except: pass
    return {"favourite_urls": [], "auto_fetch_enabled": False, "last_fetched": None}

def save_saved_urls(data):
    with open(SAVED_URLS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def add_saved_url(url, name, category="General"):
    data = load_saved_urls()
    if not any(u["url"] == url for u in data["favourite_urls"]):
        data["favourite_urls"].append({
            "url": url, "name": name,
            "category": category,
            "added_date": datetime.now().strftime("%Y-%m-%d")
        })
        save_saved_urls(data)

def remove_saved_url(url):
    data = load_saved_urls()
    data["favourite_urls"] = [u for u in data["favourite_urls"] if u["url"] != url]
    save_saved_urls(data)

def detect_topics(chunks, llm) -> list:
    try:
        sample_text = " ".join([c.page_content for c in chunks[:5]])
        prompt = f"""
        Analyze this news content and identify the main topics.
        Return ONLY a JSON list of 4-6 short topic tags.
        Example: ["Sports", "Cricket", "IPL", "Politics"]
        Content: {sample_text[:2000]}
        Return only the JSON list, nothing else.
        """
        response = llm.invoke(prompt)
        match = re.search(r'\[.*?\]', str(response.content) if hasattr(response, 'content') else str(response), re.DOTALL)
        if match:
            return json.loads(match.group())
    except: pass
    return ["General", "News", "Latest"]

def get_session_path(db_type="main"):
    """Returns a session-specific path for the vectorstore."""
    session_id = st.session_state.get("conversation_id", "default")
    return os.path.join(".", "data", "sessions", session_id, f"chroma_db_{db_type}")

def cleanup_old_sessions(max_age_hours=12):
    """Deletes session directories older than max_age_hours to save disk space."""
    base_path = os.path.join(".", "data", "sessions")
    if not os.path.exists(base_path):
        return
        
    try:
        now = time.time()
        for session_id in os.listdir(base_path):
            session_dir = os.path.join(base_path, session_id)
            if os.path.isdir(session_dir):
                mtime = os.path.getmtime(session_dir)
                if (now - mtime) > (max_age_hours * 3600):
                    import shutil
                    shutil.rmtree(session_dir, ignore_errors=True)
                    print(f"Cleaned up old session: {session_id}")
    except Exception as e:
        print(f"Cleanup error: {e}")

def calculate_confidence(answer, source_docs, question):
    if not source_docs: return 15, "low"
    
    # Factor 1: Number of source chunks found (max 40%)
    chunk_score = min(len(source_docs) / 8 * 40, 40)
    
    # Factor 2: Check if answer says "don't have info" (0% if so)
    negative_phrases = ["don't have enough", "not enough information", "cannot find", "no information", "i don't know"]
    if any(phrase in answer.lower() for phrase in negative_phrases):
        return 15, "low"
    
    # Factor 3: Answer length (longer = more confident, max 30%)
    length_score = min(len(answer) / 500 * 30, 30)
    
    # Factor 4: Source diversity (different URLs, max 30%)
    unique_sources = len(set([doc.metadata.get("source", "Unknown") for doc in source_docs]))
    diversity_score = min(unique_sources / 3 * 30, 30)
    
    total = min(int(chunk_score + length_score + diversity_score), 98)
    level = "high" if total >= 75 else "medium" if total >= 45 else "low"
    return total, level

def generate_followups(answer, question, llm) -> list:
    try:
        prompt = f"""
        Based on this Q&A, generate exactly 3 short follow-up questions a user might ask next.
        Original Question: {question}
        Answer given: {answer[:500]}
        Return ONLY a JSON list of 3 strings.
        Return only the JSON list, nothing else.
        """
        response = llm.invoke(prompt)
        text = str(response.content) if hasattr(response, 'content') else str(response)
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            return json.loads(match.group())[:3]
    except: pass
    return ["Tell me more about this", "What are the key facts?", "Who are the main people involved?"]

# --- Existing Storage Helpers ---
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"conversations": []}

def save_conversation(conv_data):
    history = load_history()
    found = False
    for i, conv in enumerate(history["conversations"]):
        if conv["id"] == conv_data["id"]:
            history["conversations"][i] = conv_data
            found = True
            break
    if not found:
        history["conversations"].append(conv_data)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def delete_conversation(conv_id):
    history = load_history()
    history["conversations"] = [c for c in history["conversations"] if c["id"] != conv_id]
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    if st.session_state.get("active_history_id") == conv_id:
        st.session_state["messages"] = []
        st.session_state["active_history_id"] = None
    st.rerun()

def toggle_pin(conv_id):
    history = load_history()
    for conv in history["conversations"]:
        if conv["id"] == conv_id:
            conv["pinned"] = not conv.get("pinned", False)
            break
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    st.rerun()

# --- Session State Management ---
if "messages" not in st.session_state: st.session_state["messages"] = []
if "conversation_id" not in st.session_state: st.session_state["conversation_id"] = str(uuid.uuid4())
if "db_ready" not in st.session_state: st.session_state["db_ready"] = os.path.exists("./chroma_db")
if "show_welcome" not in st.session_state: st.session_state["show_welcome"] = not os.path.exists(HISTORY_FILE)
if "viewing_history" not in st.session_state: st.session_state["viewing_history"] = False
if "active_history_id" not in st.session_state: st.session_state["active_history_id"] = None
if "model_used" not in st.session_state: st.session_state["model_used"] = "Groq"
if "current_urls" not in st.session_state: st.session_state["current_urls"] = []
if "search_query" not in st.session_state: st.session_state["search_query"] = ""
if "hist_search" not in st.session_state: st.session_state["hist_search"] = ""
if "faq_query" not in st.session_state: st.session_state["faq_query"] = None

# New Tier 1 State Variables
if "compare_mode" not in st.session_state: st.session_state["compare_mode"] = False
if "db_A_ready" not in st.session_state: st.session_state["db_A_ready"] = False
if "db_B_ready" not in st.session_state: st.session_state["db_B_ready"] = False
if "active_topic" not in st.session_state: st.session_state["active_topic"] = "All"
if "topics" not in st.session_state: st.session_state["topics"] = []
if "language" not in st.session_state: st.session_state["language"] = "English"
if "suggested_queries" not in st.session_state: st.session_state["suggested_queries"] = []
if "auto_query" not in st.session_state: st.session_state["auto_query"] = None
if "collection_main" not in st.session_state: st.session_state["collection_main"] = "neural_docs"
if "collection_A" not in st.session_state: st.session_state["collection_A"] = "source_A"
if "collection_B" not in st.session_state: st.session_state["collection_B"] = "source_B"

# --- CSS Theme Injection (LASER AI THEME) ---
bg_img_css = f'background-image: url("data:image/jpeg;base64,{bg_image}");' if bg_image else "background-color: #000000;"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');

    :root {{
        --laser-cyan: #00ffff;
        --laser-purple: #bf00ff;
        --laser-blue: #0066ff;
        --laser-pink: #ff00aa;
        --glow-cyan: 0 0 20px #00ffff, 0 0 40px #00ffff44;
        --glow-purple: 0 0 20px #bf00ff, 0 0 40px #bf00ff44;
    }}

    .stApp {{
        {bg_img_css}
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
        color: #e6edf3;
        font-family: 'Rajdhani', sans-serif;
    }}

    /* Dark Overlay with Cyan Tint */
    .stApp::before {{
        content: '';
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background: linear-gradient(
            135deg,
            rgba(0,0,0,0.85) 0%,
            rgba(5,5,20,0.80) 50%,
            rgba(0,255,150,0.02) 75%,
            rgba(0,0,0,0.85) 100%
        );
        z-index: 0;
        pointer-events: none;
    }}

    /* Animated Laser Grid */
    .stApp::after {{
        content: '';
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 200%;
        background-image: 
            linear-gradient(rgba(0,255,255,0.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,255,255,0.04) 1px, transparent 1px);
        background-size: 50px 50px;
        z-index: 1;
        pointer-events: none;
        animation: grid-move 50s linear infinite;
    }}

    @keyframes grid-move {{
        from {{ transform: translateY(0); }}
        to {{ transform: translateY(-50px); }}
    }}

    /* Laser Beams */
    .laser-beam {{
        position: fixed;
        width: 100%;
        height: 1px;
        opacity: 0.12;
        z-index: 1;
        pointer-events: none;
    }}
    .beam-cyan {{ background: #00ffff; top: 30%; animation: shoot 8s infinite; }}
    .beam-purple {{ background: #bf00ff; top: 60%; animation: shoot 12s infinite reverse; }}
    .beam-pink {{ background: #ff00aa; top: 45%; animation: shoot 15s infinite; }}
    
    @keyframes shoot {{
        0% {{ transform: translateX(-100%) scaleX(0); }}
        50% {{ transform: translateX(0) scaleX(1); }}
        100% {{ transform: translateX(100%) scaleX(0); }}
    }}

    /* Laser Scan Line */
    .scan-line {{
        position: fixed;
        top: 0; left: 0; width: 100%; height: 2px;
        background: rgba(0,255,255,0.15);
        box-shadow: 0 0 15px rgba(0,255,255,0.5);
        z-index: 9999;
        pointer-events: none;
        animation: scan 8s linear infinite;
        opacity: 0.08;
    }}
    @keyframes scan {{
        from {{ top: 0; }}
        to {{ top: 100%; }}
    }}

    /* Particles */
    .particle-container {{
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        z-index: 1;
        pointer-events: none;
    }}
    .particle {{
        position: absolute;
        width: 2px; height: 2px;
        background: #00ffff;
        border-radius: 50%;
        opacity: 0.3;
        animation: float-up 20s linear infinite;
    }}
    @keyframes float-up {{
        from {{ transform: translateY(110vh) translateX(0); }}
        to {{ transform: translateY(-10vh) translateX(20px); }}
    }}

    /* Layout Transparency */
    section[data-testid="stSidebar"] {{
        background: rgba(5,5,16,0.92) !important;
        backdrop-filter: blur(15px) !important;
        border-right: 1px solid rgba(0,255,255,0.1);
        z-index: 100;
    }}

    [data-testid="stMain"] {{
        background: transparent !important;
        z-index: 10;
        position: relative;
    }}

    .main-chat-container {{
        background: rgba(0,0,0,0.6);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(0,255,255,0.1);
        border-radius: 20px;
        padding: 20px;
        margin-top: 20px;
        position: relative;
    }}

    /* Corner Brackets */
    .corner-bracket {{
        position: absolute; width: 20px; height: 20px;
        border: 2px solid #00ffff;
        z-index: 100;
    }}
    .top-left {{ top: 10px; left: 10px; border-right: 0; border-bottom: 0; }}
    .top-right {{ top: 10px; right: 10px; border-left: 0; border-bottom: 0; }}
    .bottom-left {{ bottom: 10px; left: 10px; border-right: 0; border-top: 0; }}
    .bottom-right {{ bottom: 10px; right: 10px; border-left: 0; border-top: 0; }}

    /* Title Styling */
    .laser-title {{
        font-family: 'Orbitron', sans-serif;
        font-weight: 900;
        font-size: 2.8rem;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #00ffff, #bf00ff, #ff00aa, #00ffff);
        background-size: 300% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: text-flow 4s linear infinite;
        filter: drop-shadow(0 0 10px rgba(0,255,255,0.5));
    }}
    @keyframes text-flow {{
        to {{ background-position: 300% center; }}
    }}

    .laser-subtitle {{
        font-family: 'Rajdhani', sans-serif;
        font-weight: 700;
        letter-spacing: 10px;
        text-align: center;
        color: #00ffff;
        font-size: 0.8rem;
        margin-top: -15px;
        margin-bottom: 30px;
        animation: pulse-glow 2s infinite;
    }}
    @keyframes pulse-glow {{
        0%, 100% {{ opacity: 0.6; text-shadow: 0 0 5px #00ffff; }}
        50% {{ opacity: 1; text-shadow: 0 0 20px #00ffff; }}
    }}

    /* System Status Bar */
    .status-bar {{
        font-family: 'Orbitron', sans-serif;
        font-size: 9px;
        color: rgba(0,255,255,0.7);
        display: flex; gap: 20px;
        justify-content: center;
        margin-bottom: 20px;
        border-bottom: 1px solid rgba(0,255,255,0.1);
        padding-bottom: 5px;
    }}
    .status-dot {{
        height: 6px; width: 6px; background-color: #3fb950;
        border-radius: 50%; display: inline-block;
        box-shadow: 0 0 8px #3fb950; animation: dot-blink 1.5s infinite;
    }}
    @keyframes dot-blink {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.3; }} }}

    /* Chat Bubbles */
    .user-bubble {{
        background: rgba(0,102,255,0.25) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(191,0,255,0.6) !important;
        color: white;
        padding: 1rem 1.2rem;
        border-radius: 18px 18px 4px 18px;
        margin-bottom: 1.5rem;
        float: right; clear: both;
        max-width: 80%;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }}
    
    .assistant-bubble {{
        background: rgba(0,5,16,0.75) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0,255,255,0.2) !important;
        border-left: 3px solid #00ffff !important;
        color: #e6edf3;
        padding: 1.2rem;
        border-radius: 18px 18px 18px 4px;
        margin-bottom: 1.5rem;
        float: left; clear: both;
        max-width: 85%;
        box-shadow: 0 4px 30px rgba(0,0,0,0.5);
    }}

    /* Sidebar Footer */
    .laser-footer {{
        border: 1px solid rgba(255,215,0,0.2);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 0 20px rgba(255,215,0,0.08);
        background: rgba(0,0,0,0.4);
        margin-top: 30px;
    }}
    .footer-name {{
        font-family: 'Orbitron', sans-serif;
        font-size: 14px;
        font-weight: 900;
        background: linear-gradient(90deg, #ffd700, #ff8c00, #ffd700);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gold-shimmer 2s infinite;
    }}
    @keyframes gold-shimmer {{
        0% {{ background-position: 0% 50% }}
        100% {{ background-position: 200% 50% }}
    }}
    .footer-stack {{ font-size: 0.65rem; color: #8b949e; margin-top: 10px; }}

    /* Custom Buttons */
    .stButton>button {{
        border-radius: 12px !important;
        font-family: 'Orbitron', sans-serif !important;
        text-transform: uppercase;
        font-weight: 700 !important;
    }}
    
    [data-testid="baseButton-primary"] {{
        background: linear-gradient(135deg, #bf00ff, #0066ff) !important;
        border: none !important;
        box-shadow: 0 0 15px rgba(191,0,255,0.4) !important;
    }}
    
    .new-chat-btn button {{ border: 1px solid #00ffff !important; color: #00ffff !important; background: transparent !important; }}
    .new-chat-btn button:hover {{ box-shadow: 0 0 15px #00ffff !important; background: rgba(0,255,255,0.1) !important; }}
    
    .clear-btn button {{ border: 1px solid #ff00aa !important; color: #ff00aa !important; background: transparent !important; }}
    .clear-btn button:hover {{ box-shadow: 0 0 15px #ff00aa !important; background: rgba(255,0,170,0.1) !important; }}

    /* History Cards */
    .history-card {{
        background: rgba(8,8,21,0.8) !important;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(0,255,255,0.15) !important;
        border-left: 0 !important;
        transition: 0.3s;
    }}
    .history-card:hover {{
        border-left: 3px solid #00ffff !important;
        box-shadow: 0 0 15px rgba(0,255,255,0.2) !important;
    }}

    /* Scrollbar */
    ::-webkit-scrollbar {{ width: 5px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: linear-gradient(#00ffff, #bf00ff); border-radius: 10px; }}

    /* Typing Indicator */
    .typing-text {{ font-family: 'Orbitron', sans-serif; font-size: 0.7rem; color: #00ffff; }}
    
    /* FAQ Badges */
    .faq-container {{
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
        margin-bottom: 25px;
        justify-content: center;
        z-index: 100;
    }}
    .faq-badge button {{
        background: rgba(0,255,255,0.05) !important;
        border: 1px solid rgba(0,255,255,0.3) !important;
        border-radius: 20px !important;
        color: #00ffff !important;
        font-family: 'Orbitron', sans-serif !important;
        font-size: 0.65rem !important;
        padding: 5px 15px !important;
        transition: 0.3s !important;
        letter-spacing: 1px !important;
    }}
    .faq-badge button:hover {{
        background: rgba(0,255,255,0.15) !important;
        border-color: #00ffff !important;
        box-shadow: 0 0 15px rgba(0,255,255,0.4) !important;
        transform: translateY(-2px);
    }}
    /* Input Styling */
    .stChatInputContainer {{
        background: rgba(5,5,20,0.8) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(0,255,255,0.3) !important;
        border-radius: 15px !important;
    }}
    .stChatInputContainer:focus-within {{
        border-color: #00ffff !important;
        box-shadow: 0 0 15px rgba(0,255,255,0.3) !important;
    }}
</style>

<div class="particle-container">
    <div class="particle" style="left:10%; animation-delay:0s;"></div>
    <div class="particle" style="left:30%; animation-delay:5s;"></div>
    <div class="particle" style="left:50%; animation-delay:2s;"></div>
    <div class="particle" style="left:70%; animation-delay:8s;"></div>
    <div class="particle" style="left:90%; animation-delay:4s;"></div>
</div>
<div class="laser-beam beam-cyan"></div>
<div class="laser-beam beam-purple"></div>
<div class="laser-beam beam-pink"></div>
<div class="scan-line"></div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<div style="text-align: center; margin-bottom: 20px;"><span style="font-size: 40px;">🤖</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="laser-title" style="font-size: 1.5rem;">NEURAL HUB</div>', unsafe_allow_html=True)
    
    # Session Controls
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
        if st.button("➕ NEW", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["conversation_id"] = str(uuid.uuid4())
            st.session_state["active_history_id"] = None
            st.session_state["current_urls"] = []
            st.session_state["db_ready"] = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with s_col2:
        st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
        with st.popover("🗑️ RESET", use_container_width=True):
            if st.button("CONFIRM WIPE", type="primary", use_container_width=True):
                st.session_state["messages"] = []
                st.toast("Neural cache cleared!")
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # Data Ingestion
    st.markdown('<div style="font-family: Orbitron; font-size: 0.8rem; color: #00ffff; margin-bottom: 10px;">1. DATA INGESTION</div>', unsafe_allow_html=True)
    st.info("⚡ System optimized for 1000+ concurrent sessions. (Cleanup active)")
    
    st.session_state["compare_mode"] = st.toggle("⚡ COMPARE MODE (SOURCE A vs B)", value=st.session_state["compare_mode"])
    
    if not st.session_state["compare_mode"]:
        url_text = st.text_area("TARGET ARRAYS", height=80, placeholder="Input neural sources (URLs)...", label_visibility="collapsed")
        if st.button("🚀 EXECUTE INGESTION", type="primary", use_container_width=True):
            if url_text:
                progress = st.progress(0)
                status = st.empty()
                try:
                    urls = [u.strip() for u in url_text.split("\n") if u.strip()]
                    st.session_state["current_urls"] = urls
                    status.info("⚡ SYNCING NEUROLINKS...")
                    progress.progress(30)
                    chunks = ingest_urls(urls)
                    if chunks:
                        status.info("🧠 MAPPING KNOWLEDGE GRAPH...")
                        progress.progress(70)
                        # Unique collection ID to avoid metadata mismatch
                        st.session_state["collection_main"] = f"main_{uuid.uuid4().hex[:8]}"
                        create_vectorstore(chunks, path=get_session_path("main"), collection=st.session_state["collection_main"])
                        st.session_state["topics"] = detect_topics(chunks, get_llm("groq"))
                        st.session_state["active_topic"] = "All" # Reset active filter
                        st.session_state["db_ready"] = True
                        progress.progress(100)
                        st.success("GRAPH SYNC COMPLETE!")
                        time.sleep(1)
                        st.rerun()
                except Exception as e: st.error(f"SYSTEM FAILURE: {e}")
    else:
        # Compare Mode Inputs
        c_a, c_b = st.columns(2)
        with c_a:
            st.markdown('<div style="font-size: 0.7rem; color: #bf00ff;">📰 SOURCE A URLs</div>', unsafe_allow_html=True)
            url_a = st.text_area("SOURCE A", height=60, placeholder="URLs for A...", label_visibility="collapsed", key="url_a")
            if st.button("PROCESS A", use_container_width=True):
                if url_a:
                    urls_a = [u.strip() for u in url_a.split("\n") if u.strip()]
                    chunks_a = ingest_urls(urls_a)
                    if chunks_a:
                        st.session_state["collection_A"] = f"source_A_{uuid.uuid4().hex[:8]}"
                        create_vectorstore(chunks_a, path=get_session_path("A"), collection=st.session_state["collection_A"])
                        st.session_state["db_A_ready"] = True
                        st.session_state["topics"] = detect_topics(chunks_a, get_llm("groq"))
                        st.session_state["active_topic"] = "All"
                        st.success("✅ Source A Ready")
        with c_b:
            st.markdown('<div style="font-size: 0.7rem; color: #00ffff;">📰 SOURCE B URLs</div>', unsafe_allow_html=True)
            url_b = st.text_area("SOURCE B", height=60, placeholder="URLs for B...", label_visibility="collapsed", key="url_b")
            if st.button("PROCESS B", use_container_width=True):
                if url_b:
                    urls_b = [u.strip() for u in url_b.split("\n") if u.strip()]
                    chunks_b = ingest_urls(urls_b)
                    if chunks_b:
                        st.session_state["collection_B"] = f"source_B_{uuid.uuid4().hex[:8]}"
                        create_vectorstore(chunks_b, path=get_session_path("B"), collection=st.session_state["collection_B"])
                        st.session_state["db_B_ready"] = True
                        st.session_state["topics"] = detect_topics(chunks_b, get_llm("groq"))
                        st.session_state["active_topic"] = "All"
                        st.success("✅ Source B Ready")

    if st.session_state["current_urls"]:
        with st.expander("📊 SOURCE PREVIEWS"):
            for url in st.session_state["current_urls"]:
                st.markdown(f'<div style="font-size: 0.7rem; color: #00ffff;">🔗 {url[:40]}...</div>', unsafe_allow_html=True)

    st.divider()

    st.divider()

    # Topic Filter Preview (if any)
    if st.session_state["topics"]:
        st.markdown(f'<div style="font-size: 0.7rem; color: #00ff88;">🏷️ DETECTED TOPICS: {", ".join(st.session_state["topics"])}</div>', unsafe_allow_html=True)

    st.divider()

    # Saved Sources
    saved_data = load_saved_urls()
    favs = saved_data["favourite_urls"]
    st.markdown(f'<div style="font-family: Orbitron; font-size: 0.8rem; color: #ffd700; margin-bottom: 10px;">2. ⭐ SAVED SOURCES ({len(favs)})</div>', unsafe_allow_html=True)
    
    with st.expander("MANAGE FAVOURITES"):
        for i, fav in enumerate(favs):
            f_col1, f_col2, f_col3 = st.columns([3, 1, 1])
            f_col1.markdown(f'<div style="font-size: 0.7rem;">🌐 {fav["name"]}</div>', unsafe_allow_html=True)
            if f_col2.button("LOAD", key=f"fav_load_{i}"):
                with st.status(f"🔄 LOADING {fav['name']}...", expanded=False) as status:
                    st.session_state["current_urls"] = [fav["url"]]
                    st.session_state["url_input"] = fav["url"]
                    chunks = ingest_urls([fav["url"]])
                    if chunks:
                        # Unique collection ID to avoid metadata mismatch
                        st.session_state["collection_main"] = f"main_{uuid.uuid4().hex[:8]}"
                        create_vectorstore(chunks, path=get_session_path("main"), collection=st.session_state["collection_main"])
                        llm_load = get_llm("groq")
                        st.session_state["topics"] = detect_topics(chunks, llm_load)
                        st.session_state["active_topic"] = "All"
                        st.session_state["db_ready"] = True
                        status.update(label="✅ SOURCE LOADED!", state="complete")
                        time.sleep(1)
                        st.rerun()
            if f_col3.button("🗑️", key=f"fav_del_{i}"):
                remove_saved_url(fav["url"])
                st.rerun()
        
        st.divider()
        new_url = st.text_input("New URL", key="new_fav_url")
        new_name = st.text_input("Name", key="new_fav_name")
        if st.button("⭐ SAVE TO LIST", use_container_width=True):
            if new_url and new_name:
                add_saved_url(new_url, new_name)
                st.success("Saved!")
                time.sleep(0.5)
                st.rerun()

    # Auto-fetch toggle
    auto_fetch = st.toggle("🔄 Auto-fetch on startup", value=saved_data.get("auto_fetch_enabled", False))
    if auto_fetch != saved_data.get("auto_fetch_enabled", False):
        saved_data["auto_fetch_enabled"] = auto_fetch
        save_saved_urls(saved_data)

    st.divider()

    # Model & Language
    st.markdown('<div style="font-family: Orbitron; font-size: 0.8rem; color: #0066ff; margin-bottom: 10px;">3. ENGINE SETTINGS</div>', unsafe_allow_html=True)
    st.session_state["model_used"] = "groq"
    
    language_flags = {
        "English": "🇺🇸", "Hindi": "🇮🇳", "Spanish": "🇪🇸", "French": "🇫🇷",
        "German": "🇩🇪", "Arabic": "🇸🇦", "Portuguese": "🇧🇷", "Japanese": "🇯🇵",
        "Chinese": "🇨🇳", "Russian": "🇷🇺"
    }
    st.session_state["language"] = st.selectbox(
        "RESPONSE LANGUAGE",
        options=list(language_flags.keys()),
        index=list(language_flags.keys()).index(st.session_state["language"]),
        format_func=lambda x: f"{language_flags[x]} {x}"
    )

    st.divider()

    # Chat History
    st.markdown('<div style="font-family: Orbitron; font-size: 0.8rem; color: #ff00aa; margin-bottom: 10px;">3. NEURAL ARCHIVE</div>', unsafe_allow_html=True)
    hist_search = st.text_input("SEARCH", placeholder="Locate pattern...", label_visibility="collapsed")
    
    h_data = load_history()
    if h_data["conversations"]:
        all_convs = h_data["conversations"]
        if hist_search:
            all_convs = [c for c in all_convs if hist_search.lower() in c["title"].lower()]
        all_convs = sorted(all_convs, key=lambda x: (x.get("pinned", False), x.get("date_ts", 0)), reverse=True)
        
        for conv in all_convs:
            active = " border-left: 3px solid #00ffff;" if st.session_state["active_history_id"] == conv["id"] else ""
            conf_html = f'<div style="font-size: 0.6rem; color: #00ff88; margin-top: 5px;">🎯 {conv.get("confidence", 0)}% CONFIDENCE</div>' if "confidence" in conv else ""
            with st.container():
                st.markdown(f"""
                <div class="history-card" style="{active} padding: 10px; margin-bottom: 10px; border-radius: 8px;">
                    <div style="font-family: Orbitron; font-size: 0.75rem; color: white;">{conv['title']}</div>
                    <div style="font-size: 0.6rem; color: #8b949e; margin-top: 5px;">{conv['date']} | {conv['model_used']}</div>
                    {conf_html}
                </div>
                """, unsafe_allow_html=True)
                
                h_col1, h_col2, h_col3 = st.columns([3, 1, 1])
                if h_col1.button("LOAD", key=f"load_{conv['id']}", use_container_width=True):
                    st.session_state["messages"] = conv["messages"]
                    st.session_state["conversation_id"] = conv["id"]
                    st.session_state["active_history_id"] = conv["id"]
                    st.rerun()
                if h_col2.button("📌", key=f"pin_{conv['id']}"): toggle_pin(conv["id"])
                if h_col3.button("🗑️", key=f"del_{conv['id']}"): delete_conversation(conv["id"])

    # Footer
    st.markdown(f"""
    <div class="laser-footer">
        <div style="font-family: Orbitron; color: #00ffff; font-size: 0.9rem; margin-bottom: 5px;">⚡ RAG ASSISTANT AI</div>
        <div style="font-family: Orbitron; color: #bf00ff; font-size: 0.6rem; letter-spacing: 5px; margin-bottom: 15px;">VERSION 2.0</div>
        <div style="height: 1px; background: linear-gradient(90deg, transparent, rgba(0,255,255,0.3), transparent); margin-bottom: 15px;"></div>
        <div style="font-size: 0.7rem; color: #8b949e; margin-bottom: 5px;">ARCHITECTED BY</div>
        <div class="footer-name">✨ Rishi Jain ✨</div>
        <div style="height: 1px; background: linear-gradient(90deg, transparent, rgba(0,255,255,0.3), transparent); margin: 15px 0;"></div>
        <div class="footer-stack">⚡ GROQ | 🤗 HF | 🦜 LC</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Export Button
    st.divider()
    if st.session_state["messages"]:
        chat_text = ""
        for msg in st.session_state["messages"]:
            chat_text += f"{msg['role'].upper()}: {msg['content']}\n\n"
        st.download_button(
            label="💾 EXPORT CONVERSATION",
            data=chat_text,
            file_name=f"chat_export_{st.session_state['conversation_id'][:8]}.txt",
            mime="text/plain",
            use_container_width=True
        )

# --- AUTO-FETCH LOGIC ---
if "first_run" not in st.session_state:
    st.session_state["first_run"] = False
    # Scale-out: Cleanup old data on new connection
    cleanup_old_sessions()
    
    s_data = load_saved_urls()
    if s_data.get("auto_fetch_enabled") and s_data.get("favourite_urls"):
        with st.sidebar:
            with st.status("🔄 AUTO-FETCHING SAVED SOURCES...", expanded=True) as status:
                fav_urls = [f["url"] for f in s_data["favourite_urls"]]
                st.session_state["current_urls"] = fav_urls
                a_chunks = ingest_urls(fav_urls)
                if a_chunks:
                    create_vectorstore(a_chunks, path=get_session_path("main"))
                    st.session_state["topics"] = detect_topics(a_chunks, get_llm("groq"))
                    st.session_state["db_ready"] = True
                    status.update(label=f"✅ AUTO-FETCH COMPLETE! {len(a_chunks)} chunks ready", state="complete")
                    s_data["last_fetched"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                    save_saved_urls(s_data)
                    time.sleep(1)
                    st.rerun()

# --- MAIN CHAT AREA ---
st.markdown('<div class="laser-title">RAG ASSISTANT AI</div>', unsafe_allow_html=True)
st.markdown('<div class="laser-subtitle">NEURAL NEWS INTELLIGENCE SYSTEM</div>', unsafe_allow_html=True)

# System Status Bar
lang_badge = f'<div>🌍 {st.session_state["language"].upper()}</div>'
topic_badge = f'<div>🏷️ #{st.session_state["active_topic"].upper()}</div>' if st.session_state["active_topic"] != "All" else ""

st.markdown(f"""
<div class="status-bar">
    <div><span class="status-dot"></span> SYSTEM ONLINE</div>
    <div>⚡ {st.session_state["model_used"].upper()} ENGINE CONNECTED</div>
    <div>🧠 CHROMA VECTOR READY</div>
    {lang_badge}
    {topic_badge}
</div>
""", unsafe_allow_html=True)

# Topic Filter Tags
if st.session_state.get("topics"):
    st.markdown('<div class="faq-container" style="margin-bottom: 20px;">', unsafe_allow_html=True)
    t_cols = st.columns(len(st.session_state["topics"]) + 1)
    
    with t_cols[0]:
        active_style = "border: 1px solid #00ffff; box-shadow: 0 0 10px rgba(0,255,255,0.4); color: #00ffff;" if st.session_state["active_topic"] == "All" else "color: #00ffff88; border: 1px solid #00ffff44;"
        st.markdown(f'<div class="faq-badge" style="{active_style}">', unsafe_allow_html=True)
        if st.button("🌐 ALL", key="topic_all", use_container_width=True):
            st.session_state["active_topic"] = "All"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
    for i, topic in enumerate(st.session_state["topics"]):
        with t_cols[i+1]:
            active_style = "border: 1px solid #00ffff; box-shadow: 0 0 10px rgba(0,255,255,0.4); color: #00ffff;" if st.session_state["active_topic"] == topic else "color: #00ffff88; border: 1px solid #00ffff44;"
            st.markdown(f'<div class="faq-badge" style="{active_style}">', unsafe_allow_html=True)
            if st.button(f"#{topic.upper()}", key=f"topic_{i}", use_container_width=True):
                st.session_state["active_topic"] = topic
                st.session_state["auto_query"] = f"Give me a detailed summary of the latest news related to {topic}."
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Quick Action FAQs
st.markdown('<div class="faq-container">', unsafe_allow_html=True)
faq_queries = ["📋 NEWS SUMMARY", "🔥 TOP 10 HEADLINES", "🌍 WORLD UPDATE", "💻 TECH TRENDS"]
faq_map = {
    "📋 NEWS SUMMARY": "Give me a concise summary of the latest news from the ingested sources.",
    "🔥 TOP 10 HEADLINES": "What are the top 10 most important news stories today based on the provided links?",
    "🌍 WORLD UPDATE": "Provide an update on the most significant global news events from the sources.",
    "💻 TECH TRENDS": "Summarize the key technology news and trends mentioned in the ingested articles."
}
f_cols = st.columns(len(faq_queries))
for i, label in enumerate(faq_queries):
    with f_cols[i]:
        st.markdown('<div class="faq-badge">', unsafe_allow_html=True)
        if st.button(label, key=f"faq_{i}", use_container_width=True):
            st.session_state["faq_query"] = faq_map[label]
st.markdown('</div>', unsafe_allow_html=True)

# Suggested Follow-ups
if st.session_state.get("suggested_queries") and not st.session_state.get("thinking"):
    st.markdown('<div class="faq-container" style="margin-top: 10px;">', unsafe_allow_html=True)
    s_cols = st.columns(len(st.session_state["suggested_queries"]))
    for j, s_label in enumerate(st.session_state["suggested_queries"]):
        with s_cols[j]:
            st.markdown('<div class="faq-badge">', unsafe_allow_html=True)
            if st.button(s_label, key=f"suggest_{j}", use_container_width=True):
                st.session_state["faq_query"] = s_label
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Chat Input Section
user_input = st.chat_input("⚡ ENTER NEURAL QUERY...")

# Main Chat Display
with st.container():
    st.markdown('<div class="main-chat-container">', unsafe_allow_html=True)
    st.markdown('<div class="corner-bracket top-left"></div><div class="corner-bracket top-right"></div>', unsafe_allow_html=True)
    st.markdown('<div class="corner-bracket bottom-left"></div><div class="corner-bracket bottom-right"></div>', unsafe_allow_html=True)
    
    for i, msg in enumerate(st.session_state["messages"]):
        if msg["role"] == "user":
            st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            if msg.get("is_comparison"):
                # --- NATIVE STREAMLIT COMPARISON (Fixes overlap & code display) ---
                st.markdown(f'<div style="font-family: Orbitron; color: #00ffff; margin-bottom: 10px; font-size: 1.2rem;">📊 NEURAL COMPARISON: {msg.get("timestamp")}</div>', unsafe_allow_html=True)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"""
                    <div style="background: rgba(191,0,255,0.08); border: 1px solid rgba(191,0,255,0.3); padding: 15px; border-radius: 12px; height: 100%;">
                        <h4 style="color: #bf00ff; margin-top: 0; font-family: Orbitron;">📰 SOURCE A</h4>
                        <div style="font-size: 0.9rem; line-height: 1.6;">{msg.get('content_a', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div style="background: rgba(0,255,255,0.08); border: 1px solid rgba(0,255,255,0.3); padding: 15px; border-radius: 12px; height: 100%;">
                        <h4 style="color: #00ffff; margin-top: 0; font-family: Orbitron;">📰 SOURCE B</h4>
                        <div style="font-size: 0.9rem; line-height: 1.6;">{msg.get('content_b', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Synthesis below
                st.markdown(f"""
                <div style="margin-top: 20px; background: rgba(0,0,0,0.4); border: 1px solid rgba(255,215,0,0.2); padding: 20px; border-radius: 12px; border-left: 5px solid #ffd700;">
                    <h4 style="color: #ffd700; margin-top: 0; font-family: Orbitron;">🔄 AI SYNTHESIS</h4>
                    <div style="font-size: 0.95rem; line-height: 1.7;">{msg.get('content_synth', '')}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence and Time
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:12px; margin-top:15px; background: rgba(0,0,0,0.3); padding: 10px; border-radius: 10px;">
                  <div style="border: 1px solid #00ff88; border-radius: 20px; padding: 2px 12px; font-size: 10px; color: #00ff88; font-family: 'Orbitron';">🎯 95% CONFIDENCE</div>
                  <div style="flex: 1; height: 4px; background: rgba(255,255,255,0.1); border-radius: 2px;"><div style="width: 95%; height: 100%; background: #00ff88; border-radius:2px;"></div></div>
                  <div style="font-family: Orbitron; font-size: 8px; color: #00ffff88;">⚡ {msg.get('time', '0')}s</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Standard Assistant Bubble (Fixed rendering)
                score = msg.get("confidence", 100)
                level = msg.get("confidence_level", "high")
                colors = {"high": "#00ff88", "medium": "#ffaa00", "low": "#ff4444"}
                color = colors.get(level, "#00ffff")
                
                # Render content first
                st.markdown(f'<div class="assistant-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
                
                # Render confidence score as a separate HTML element below
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:12px; margin-top:-10px; margin-bottom: 20px; background: rgba(0,0,0,0.3); padding: 8px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.05); max-width: 85%; margin-left: auto;">
                  <div style="background: rgba(0,0,0,0.4); border: 1px solid {color}; border-radius: 20px; padding: 2px 10px; font-size: 10px; color: {color}; box-shadow: 0 0 8px {color}44; font-family: 'Orbitron';">
                    🎯 {score}% — {level.upper()}
                  </div>
                  <div style="flex: 1; height: 3px; background: rgba(255,255,255,0.1); border-radius: 2px; overflow: hidden;">
                    <div style="width: {score}%; height: 100%; background: linear-gradient(90deg, {color}88, {color}); border-radius: 2px;"></div>
                  </div>
                  <div style="font-family: Orbitron; font-size: 8px; color: #00ffff88;">⚡ {msg.get('time', '0')}s</div>
                </div>
                """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Typing Indicator
if st.session_state.get("thinking", False):
    st.markdown("""
    <div class="assistant-bubble">
        <div class="typing-text">PROCESSING NEURAL QUERY...</div>
        <div style="display: flex; gap: 6px; margin-top: 10px;">
            <div style="width: 5px; height: 5px; background: #00ffff; border-radius: 50%; opacity: 0.3; animation: dot-blink 1.4s infinite alternate;"></div>
            <div style="width: 5px; height: 5px; background: #00ffff; border-radius: 50%; opacity: 0.3; animation: dot-blink 1.4s infinite alternate 0.2s;"></div>
            <div style="width: 5px; height: 5px; background: #00ffff; border-radius: 50%; opacity: 0.3; animation: dot-blink 1.4s infinite alternate 0.4s;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- RAG EXECUTION ---
# Handle auto_query from follow-up chips
if st.session_state.get("auto_query"):
    user_q = st.session_state.pop("auto_query")
else:
    user_q = user_input if user_input else st.session_state.get("faq_query")

if user_q:
    st.session_state["faq_query"] = None # Reset FAQ state
    timestamp = datetime.now().strftime("%I:%M %p")
    st.session_state["messages"].append({"role": "user", "content": user_q, "timestamp": timestamp})
    st.session_state["thinking"] = True
    st.rerun()

if st.session_state.get("thinking"):
    if not st.session_state["db_ready"] and not st.session_state["compare_mode"]:
        st.warning("⚠️ KNOWLEDGE GRAPH EMPTY. SYNC TARGET ARRAYS IN SIDEBAR.")
        st.session_state["thinking"] = False
    elif st.session_state["compare_mode"] and (not st.session_state["db_A_ready"] or not st.session_state["db_B_ready"]):
        st.warning("⚠️ BOTH SOURCES A & B MUST BE PROCESSED FOR COMPARISON.")
        st.session_state["thinking"] = False
    else:
        try:
            start_t = time.time()
            llm = get_llm("groq")
            
            # Prepare Instructions
            language_map = {
                "Hindi": "हिंदी में जवाब दें।", "Spanish": "Responde en español.",
                "French": "Réponds en français.", "German": "Antworte auf Deutsch.",
                "Arabic": "أجب باللغة العربية.", "Portuguese": "Responda em português.",
                "Japanese": "日本語で答えてください。", "Chinese": "请用中文回答。",
                "Russian": "Ответьте на русском языке.", "English": "Answer in English."
            }
            lang_inst = language_map.get(st.session_state["language"], "Answer in English.")
            topic_inst = f"Focus specifically on information related to {st.session_state['active_topic']}." if st.session_state["active_topic"] != "All" else ""
            
            final_query = f"{st.session_state['messages'][-1]['content']}\n\nIMPORTANT: {lang_inst} {topic_inst}"
            
            if not st.session_state["compare_mode"]:
                # Normal Mode
                v_coll = st.session_state.get("collection_main", "neural_docs")
                vstore = load_vectorstore(path=get_session_path("main"), collection=v_coll)
                if vstore:
                    chain = get_rag_chain(vstore, llm)
                    resp = chain(final_query)
                    
                    duration = round(time.time() - start_t, 2)
                    timestamp = datetime.now().strftime("%I:%M %p")
                    score, level = calculate_confidence(resp["result"], resp["source_documents"], user_q)
                    
                    asst_msg = {
                        "role": "assistant",
                        "content": resp["result"],
                        "sources": list(set([d.metadata.get("source") for d in resp["source_documents"]])),
                        "time": duration,
                        "timestamp": timestamp,
                        "confidence": score,
                        "confidence_level": level
                    }
                    st.session_state["messages"].append(asst_msg)
                    st.session_state["suggested_queries"] = generate_followups(resp["result"], user_q, llm)
            else:
                # Compare Mode (Dual RAG)
                coll_a = st.session_state.get("collection_A", "source_A")
                coll_b = st.session_state.get("collection_B", "source_B")
                v_a = load_vectorstore(path=get_session_path("A"), collection=coll_a)
                v_b = load_vectorstore(path=get_session_path("B"), collection=coll_b)
                
                if v_a and v_b:
                    resp_a = get_rag_chain(v_a, llm)(final_query)
                    resp_b = get_rag_chain(v_b, llm)(final_query)
                    
                    # AI Synthesis
                    synth_prompt = f"""
                    Given these two perspectives on the same question:
                    Source A says: {resp_a['result']}
                    Source B says: {resp_b['result']}
                    Question was: {user_q}
                    Provide a brief synthesis: What do both agree on? What are the key differences? Which is more detailed?
                    """
                    synthesis = llm.invoke(synth_prompt)
                    synth_text = str(synthesis.content) if hasattr(synthesis, 'content') else str(synthesis)
                    
                    duration = round(time.time() - start_t, 2)
                    timestamp = datetime.now().strftime("%I:%M %p")
                    
                    asst_msg = {
                        "role": "assistant",
                        "content": "NEURAL COMPARISON COMPLETE", # Placeholder for history
                        "content_a": resp_a['result'],
                        "content_b": resp_b['result'],
                        "content_synth": synth_text,
                        "time": duration,
                        "timestamp": timestamp,
                        "confidence": 95,
                        "confidence_level": "high",
                        "is_comparison": True
                    }
                    st.session_state["messages"].append(asst_msg)
                    st.session_state["suggested_queries"] = generate_followups(resp_a["result"], user_q, llm)

            # Auto-save
            if st.session_state["messages"]:
                conv_data = {
                    "id": st.session_state["conversation_id"],
                    "title": st.session_state["messages"][0]["content"][:35],
                    "date": datetime.now().strftime("%d %b %Y"),
                    "date_ts": time.time(),
                    "messages": st.session_state["messages"],
                    "model_used": st.session_state["model_used"],
                    "confidence": st.session_state["messages"][-1].get("confidence", 0) if st.session_state["messages"][-1]["role"] == "assistant" else 0,
                    "total_messages": len(st.session_state["messages"])
                }
                save_conversation(conv_data)
                
            st.session_state["thinking"] = False
            st.rerun()
        except Exception as e:
            st.error(f"NEURAL HANDSHAKE FAILURE: {e}")
            st.session_state["thinking"] = False

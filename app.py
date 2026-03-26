# Place robot image as "background.jpg" in project folder
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
from datetime import datetime
# Removed speech_recognition and mic_recorder imports

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
    except Exception:
        return None
    return None

bg_image = get_base64_image("background.jpg")

# --- Constants & Storage ---
HISTORY_FILE = "chat_history.json"

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
                    create_vectorstore(chunks)
                    st.session_state["db_ready"] = True
                    progress.progress(100)
                    st.success("GRAPH SYNC COMPLETE!")
                    time.sleep(1)
                    st.rerun()
            except Exception as e:
                st.error(f"SYSTEM FAILURE: {e}")

    if st.session_state["current_urls"]:
        with st.expander("📊 SOURCE PREVIEWS"):
            for url in st.session_state["current_urls"]:
                st.markdown(f'<div style="font-size: 0.7rem; color: #00ffff;">🔗 {url[:40]}...</div>', unsafe_allow_html=True)

    st.divider()

    # Model Selection
    st.markdown('<div style="font-family: Orbitron; font-size: 0.8rem; color: #bf00ff; margin-bottom: 10px;">2. CORE SELECTOR</div>', unsafe_allow_html=True)
    llm_choice = st.radio("CORE", ["⚡ GROQ (LLAMA-3)", "🦙 OLLAMA (MISTRAL)"], label_visibility="collapsed")
    st.session_state["model_used"] = "Groq" if "GROQ" in llm_choice else "Ollama"

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
            with st.container():
                st.markdown(f"""
                <div class="history-card" style="{active} padding: 10px; margin-bottom: 10px; border-radius: 8px;">
                    <div style="font-family: Orbitron; font-size: 0.75rem; color: white;">{conv['title']}</div>
                    <div style="font-size: 0.6rem; color: #8b949e; margin-top: 5px;">{conv['date']} | {conv['model_used']}</div>
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
        <div class="footer-stack">⚡ GROQ | 🦙 OLLAMA | 🦜 LC</div>
    </div>
    """, unsafe_allow_html=True)

# --- MAIN CHAT AREA ---
st.markdown('<div class="laser-title">RAG ASSISTANT AI</div>', unsafe_allow_html=True)
st.markdown('<div class="laser-subtitle">NEURAL NEWS INTELLIGENCE SYSTEM</div>', unsafe_allow_html=True)

# System Status Bar
st.markdown(f"""
<div class="status-bar">
    <div><span class="status-dot"></span> SYSTEM ONLINE</div>
    <div>⚡ {st.session_state["model_used"].upper()} ENGINE CONNECTED</div>
    <div>🧠 CHROMA VECTOR READY</div>
</div>
""", unsafe_allow_html=True)

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
            st.markdown(f"""
            <div class="assistant-bubble">
                {msg['content']}
                <div style="display: flex; gap: 10px; margin-top: 15px; font-family: Orbitron; font-size: 8px;">
                    <span style="border: 1px solid #00ffff; color: #00ffff; padding: 2px 8px; border-radius: 4px;">⚡ LATENCY: {msg.get('time', '0')}s</span>
                    <span style="border: 1px solid #bf00ff; color: #bf00ff; padding: 2px 8px; border-radius: 4px;">🎯 CONFIDENCE: HIGH</span>
                </div>
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
user_q = user_input if user_input else st.session_state.get("faq_query")

if user_q:
    st.session_state["faq_query"] = None # Reset FAQ state
    timestamp = datetime.now().strftime("%I:%M %p")
    st.session_state["messages"].append({"role": "user", "content": user_q, "timestamp": timestamp})
    st.session_state["thinking"] = True
    st.rerun()

if st.session_state.get("thinking"):
    if not st.session_state["db_ready"]:
        st.warning("⚠️ KNOWLEDGE GRAPH EMPTY. SYNC TARGET ARRAYS IN SIDEBAR.")
        st.session_state["thinking"] = False
    else:
        try:
            start_t = time.time()
            vstore = load_vectorstore()
            if vstore:
                llm = get_llm("groq" if st.session_state["model_used"] == "Groq" else "ollama")
                chain = get_rag_chain(vstore, llm)
                resp = chain(st.session_state["messages"][-1]["content"])
                
                duration = round(time.time() - start_t, 2)
                timestamp = datetime.now().strftime("%I:%M %p")
                
                asst_msg = {
                    "role": "assistant",
                    "content": resp["result"],
                    "sources": list(set([d.metadata.get("source") for d in resp["source_documents"]])),
                    "time": duration,
                    "timestamp": timestamp
                }
                st.session_state["messages"].append(asst_msg)
                
                # Auto-save
                conv_data = {
                    "id": st.session_state["conversation_id"],
                    "title": st.session_state["messages"][0]["content"][:35],
                    "date": datetime.now().strftime("%d %b %Y"),
                    "date_ts": time.time(),
                    "messages": st.session_state["messages"],
                    "model_used": st.session_state["model_used"],
                    "total_messages": len(st.session_state["messages"])
                }
                save_conversation(conv_data)
            st.session_state["thinking"] = False
            st.rerun()
        except Exception as e:
            st.error(f"NEURAL HANDSHAKE FAILURE: {e}")
            st.session_state["thinking"] = False

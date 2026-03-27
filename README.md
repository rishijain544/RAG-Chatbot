# 🤖 Neural News AI: RAG Intelligence System

<div align="center">
  <img src="https://img.shields.io/badge/Neural-AI-00ffff?style=for-the-badge&logo=ai-network&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-Enabled-bf00ff?style=for-the-badge&logo=chainlink&logoColor=white" />
  <img src="https://img.shields.io/badge/Laser-UI-ff00aa?style=for-the-badge&logo=visual-studio-code&logoColor=white" />
  <br>
  <a href="https://rag-chatbot-udfuro6tu4xzzsr3puquwj.streamlit.app/">
    <img src="https://img.shields.io/badge/Live-Deployment-00ff88?style=for-the-badge&logo=streamlit&logoColor=white" />
  </a>
</div>

---

## 🌌 Overview
**Neural News AI** is a premium Retrieval-Augmented Generation (RAG) assistant designed to ingest, process, and analyze real-time news data with tactical precision. Featuring a high-tech **Laser AI** interface, it provides a seamless bridge between raw information and actionable intelligence.

---

## ✨ Premium Neural Features

### 🚀 Advanced Intelligence (Tier 1)
- **⚡ Neural Comparison Mode**: Compare news coverage between Source A and Source B side-by-side.
- **🏷️ Automated Topic Mapping**: AI-driven topic detection and categorisation for ingested content.
- **🎯 Dynamic Confidence Scoring**: Real-time evaluation of response accuracy based on source diversity and context match.
- **🌍 Polyglot Responses**: Support for 10+ global languages including Hindi, Spanish, French, and more.
- **🧠 Predictive Follow-ups**: Intelligent suggestion of subsequent queries based on chat context.

### 🎨 Visual & Experience
- **🎯 Laser AI Interface**: Glassmorphism UI with animated laser grids, shooting beams, and glowing scan-lines.
- **⚡ Quick Action FAQs**: Instant news summaries and "Top 10 Headlines" at the click of a button.
- **🕵️ Robust News Extraction**: Advanced scraper optimized for NDTV, Cricbuzz, and restricted hubs.
- **🧬 Neural Archive**: Persistent chat history with search and pinning capabilities.

---

## 🛠️ Neural Tech Stack
- **Engine**: [LangChain](https://www.langchain.com/)
- **Intelligence**: [Groq API](https://groq.com/) (LLaMA-3.1-8B)
- **Vector Core**: [ChromaDB](https://www.trychroma.com/)
- **Embeddings**: [HuggingFace](https://huggingface.co/) (Sentence Transformers)
- **Interface**: [Streamlit](https://streamlit.io/) with Custom Vanilla CSS

---

## 🚀 Neural Uplink (Setup)

### 1. Requirements
Ensure you have Python 3.10+ installed.

### 2. Configuration (Local)
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Installation & Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

### ☁️ Cloud Deployment (Streamlit)
1. Add `GROQ_API_KEY` to **Advanced Settings > Secrets**.
2. Deployment handles `sqlite3` compatibility automatically via `pysqlite3-binary`.

---

## 📁 Neural Project Registry
- `app.py`: The central Neural Hub (UI/UX).
- `ingestor.py`: Information extraction logic.
- `vectorstore.py`: Neural memory management (ChromaDB).
- `rag_chain.py`: RAG intelligence logic.
- `llm.py`: LLM integration bridge.
- `requirements.txt`: System dependencies.

---

## 🏆 Credits
**Architect**: [Rishi Jain](https://github.com/rishijain544)  
**System**: Neural News Intelligence Framework  

---

<div align="center">
  <p><i>Powering the future of news analysis with tactical AI.</i></p>
  <img src="https://img.shields.io/badge/Status-Neural_Sync_Complete-00ff00?style=flat-square" />
</div>
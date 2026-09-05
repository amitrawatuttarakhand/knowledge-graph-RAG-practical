# 🎬 Knowledge Graph + RAG Movie Assistant

This project demonstrates a Hybrid Retrieval-Augmented Generation (RAG) system:
- **Graph Retriever (Neo4j)** → factual queries (actors, directors, collaborations)
- **Vector Retriever (FAISS + HuggingFace)** → semantic queries (recommendations, moods, themes)
- **Groq LLM** → generates polished answers
- **Streamlit UI** → interactive interface with help tips

## 🚀 Run
```bash
pip install -r requirements.txt
streamlit run app.py

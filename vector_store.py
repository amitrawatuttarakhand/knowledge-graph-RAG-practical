"""
===========================================================================================
vector_store.py — FAISS Vector Store Builder (Groq/Free Version)
===========================================================================================

PURPOSE:
  Converts movie text into embeddings using HuggingFace/SentenceTransformers
  instead of OpenAI, then stores them in FAISS for semantic search.
===========================================================================================
"""

from __future__ import annotations
from typing import Optional, List

# HuggingFace embeddings (free, local models)
from langchain_community.embeddings import HuggingFaceEmbeddings

# FAISS for similarity search
from langchain_community.vectorstores import FAISS

# LangChain Document class
from langchain_core.documents import Document

# Import movie dataset
from graph_loader import MOVIES_DATA

# ─────────────────────────────────────────────────────────────────────────────────────────
# SECTION 1: Document Builder
# ─────────────────────────────────────────────────────────────────────────────────────────

def build_documents() -> list[Document]:
    docs = []
    for movie in MOVIES_DATA:
        content = (
            f"Title: {movie['title']}\n"
            f"Year: {movie['year']}\n"
            f"Genre: {movie['genre']}\n"
            f"Director: {movie['director']}\n"
            f"Cast: {', '.join(movie['actors'])}\n"
            f"Description: {movie['description']}"
        )
        metadata = {
            "title": movie["title"],
            "year": movie["year"],
            "genre": movie["genre"],
            "director": movie["director"],
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs

# ─────────────────────────────────────────────────────────────────────────────────────────
# SECTION 2: Vector Store Factory
# ─────────────────────────────────────────────────────────────────────────────────────────

def create_vector_store() -> FAISS:
    """
    Build FAISS vector index using HuggingFace embeddings.
    """
    # Use a free HuggingFace model (MiniLM is fast & accurate)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    docs = build_documents()
    print(f"  Embedding {len(docs)} movie documents ...")

    vs = FAISS.from_documents(docs, embeddings)
    print("  [OK] FAISS vector store built.")
    return vs

# ─────────────────────────────────────────────────────────────────────────────────────────
# SECTION 3: Singleton Accessor
# ─────────────────────────────────────────────────────────────────────────────────────────

_vector_store: Optional[FAISS] = None

def get_vector_store() -> "FAISS":
    global _vector_store
    if _vector_store is None:
        _vector_store = create_vector_store()
    return _vector_store

# ─────────────────────────────────────────────────────────────────────────────────────────
# SECTION 4: Public API — Search Function
# ─────────────────────────────────────────────────────────────────────────────────────────

def similarity_search(query: str, k: int = 4) -> List[Document]:
    vs = get_vector_store()
    return vs.similarity_search(query, k=k)

# ─────────────────────────────────────────────────────────────────────────────────────────
# SECTION 5: Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building vector store ...")
    vs = create_vector_store()

    test_queries = [
        "space exploration and simulated reality",
        "romantic love story and disaster",
        "crime corruption and moral dilemmas",
    ]

    print()
    for q in test_queries:
        print(f"Query: '{q}'")
        results = vs.similarity_search(q, k=3)
        for r in results:
            print(f"  -> {r.metadata['title']} ({r.metadata['year']}) [{r.metadata['genre']}]")
        print()

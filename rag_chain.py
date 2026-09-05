"""
===========================================================================================
rag_chain.py — Hybrid RAG Pipeline Orchestrator (Groq Version)
===========================================================================================

PURPOSE:
  Orchestrates the hybrid RAG pipeline using Groq LLM instead of OpenAI.
  Combines router, retrievers, and Groq LLM to answer user questions with sourced context.
===========================================================================================
"""

# Import Groq client
from groq import Groq

# Import LangChain's prompt templating system
from langchain_core.prompts import ChatPromptTemplate

# Import LangChain's output parser (converts LLM output to strings)
from langchain_core.output_parsers import StrOutputParser

# Import LangChain's Document class (for type hints)
from langchain_core.documents import Document

# Import configuration (ensures API keys are loaded)
import os
import config

# Import the query router
from router import route

# Import the graph-based retriever
from graph_retriever import graph_retrieve

# Import the vector-based retriever
from vector_store import similarity_search

# ─────────────────────────────────────────────────────────────────────────────────────────
# SECTION 1: LLM Configuration & Prompt Template
# ─────────────────────────────────────────────────────────────────────────────────────────

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Define the prompt template for answer generation
_ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are a knowledgeable movie assistant.

Use ONLY the context below to answer the user's question. If the context
does not contain enough information to answer confidently, say so explicitly
rather than guessing.

--- CONTEXT START ---
{context}
--- CONTEXT END ---

Question: {question}

Answer:"""
)

# Custom Groq wrapper for LangChain-style chain
def groq_generate(inputs: dict) -> str:
    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are a knowledgeable movie assistant."},
            {"role": "user", "content": _ANSWER_PROMPT.format(**inputs)}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content

# Chain = prompt → Groq → string parser
_chain = lambda inputs: StrOutputParser().invoke(groq_generate(inputs))

# ─────────────────────────────────────────────────────────────────────────────────────────
# SECTION 2: Helper Functions
# ─────────────────────────────────────────────────────────────────────────────────────────

def _format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

# ─────────────────────────────────────────────────────────────────────────────────────────
# SECTION 3: Main RAG Pipeline
# ─────────────────────────────────────────────────────────────────────────────────────────

def run_hybrid_rag(query: str, verbose: bool = True) -> dict:
    retriever_type = route(query)
    if verbose:
        print(f"  [Router]  -> {retriever_type.upper()} retrieval")

    if retriever_type == "graph":
        context = graph_retrieve(query)
    else:
        docs = similarity_search(query, k=4)
        context = _format_docs(docs)

    if verbose:
        print(f"  [Context] {len(context)} chars retrieved")

    answer = _chain({"context": context, "question": query})

    return {
        "query": query,
        "retriever": retriever_type,
        "context": context,
        "answer": answer,
    }

# ─────────────────────────────────────────────────────────────────────────────────────────
# SECTION 4: Demo
# ─────────────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_queries = [
        "Who acted in The Matrix?",
        "Recommend a movie about survival on a deserted island.",
        "Who has Tom Hanks worked with across his films?",
        "Find me something romantic and emotional to watch.",
        "What did Christopher Nolan direct?",
    ]

    print("Hybrid RAG Demo (Groq)\n" + "=" * 60)

    for q in sample_queries:
        print(f"\nQ: {q}")
        result = run_hybrid_rag(q, verbose=True)
        print(f"A: {result['answer']}")
        print("-" * 60)

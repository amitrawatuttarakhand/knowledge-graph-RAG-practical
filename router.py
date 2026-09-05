"""
===========================================================================================
router.py — Intelligent Query Router (Groq Version)
===========================================================================================

PURPOSE:
  Classifies user queries as either "graph" (structural) or "vector" (semantic).
  Uses keyword heuristics first, then Groq LLM fallback for ambiguous queries.
===========================================================================================
"""

from __future__ import annotations
from typing import Optional
import os
from groq import Groq
from langchain_core.prompts import ChatPromptTemplate

import config

# ─────────────────────────────────────────────────────────────────────────────────────────
# SECTION 1: Keyword Sets for Heuristic Routing
# ─────────────────────────────────────────────────────────────────────────────────────────

GRAPH_KEYWORDS = {
    "who", "which actor", "which director", "acted with", "co-star", "directed by",
    "appeared in", "movies by", "films by", "starred in", "worked with", "same movie",
    "co-act", "what actors", "who directed", "director of", "cast of", "who else",
    "list", "how many movies", "filmography", "what movies", "what films",
}

VECTOR_KEYWORDS = {
    "about", "similar to", "like", "recommend", "find me", "suggest", "describe",
    "what kind", "theme", "genre", "feel", "mood", "story about", "involving",
    "related to", "tell me about", "summarize", "based on", "sounds like",
    "i want to watch",
}

# ─────────────────────────────────────────────────────────────────────────────────────────
# SECTION 2: Keyword-Based Heuristic Router
# ─────────────────────────────────────────────────────────────────────────────────────────

def _keyword_route(query: str) -> Optional[str]:
    q = query.lower()
    graph_score = sum(1 for kw in GRAPH_KEYWORDS if kw in q)
    vector_score = sum(1 for kw in VECTOR_KEYWORDS if kw in q)

    if graph_score > vector_score:
        return "graph"
    if vector_score > graph_score:
        return "vector"
    return None

# ─────────────────────────────────────────────────────────────────────────────────────────
# SECTION 3: Groq-Based Fallback Router
# ─────────────────────────────────────────────────────────────────────────────────────────

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

_ROUTING_PROMPT = ChatPromptTemplate.from_template(
    """You are a query routing assistant for a movie knowledge system.

Classify the user's question into exactly one of two categories:

  "graph"  — The question asks about specific entities, relationships, or
             structured facts (e.g. "Who acted with Tom Hanks?",
             "What did Christopher Nolan direct?", "Who is in Inception?").

  "vector" — The question is semantic, thematic, or recommendation-based
             (e.g. "Recommend a sci-fi movie", "Find something romantic",
             "Movies about survival", "Something like The Matrix").

Question: {query}

Respond with ONLY the single word "graph" or "vector".
"""
)

def _llm_route(query: str) -> str:
    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are a query routing assistant."},
            {"role": "user", "content": _ROUTING_PROMPT.format(query=query)}
        ],
        temperature=0
    )
    answer = response.choices[0].message.content.strip().lower()
    return "graph" if "graph" in answer else "vector"

# ─────────────────────────────────────────────────────────────────────────────────────────
# SECTION 4: Public API — Main Routing Function
# ─────────────────────────────────────────────────────────────────────────────────────────

def route(query: str) -> str:
    decision = _keyword_route(query)
    if decision:
        return decision
    return _llm_route(query)

# ─────────────────────────────────────────────────────────────────────────────────────────
# SECTION 5: Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        ("Who acted in The Matrix?", "graph"),
        ("What movies did Tom Hanks appear in?", "graph"),
        ("Who directed Inception?", "graph"),
        ("Recommend a sci-fi movie involving space.", "vector"),
        ("I want to watch something romantic and emotional.", "vector"),
    ]

    print("Query Routing Test (Groq)\n" + "=" * 60)
    correct = 0
    for query, expected in test_queries:
        actual = route(query)
        ok = "[OK]" if actual == expected else "[FAIL]"
        if actual == expected:
            correct += 1
        print(f"  {ok} [{actual.upper():6}] {query}")
    print(f"\nAccuracy: {correct}/{len(test_queries)}")

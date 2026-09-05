import streamlit as st
from rag_chain import run_hybrid_rag

HELP_TEXT = """
Tip -- try queries like:
  * Who acted in The Matrix?             
  * What movies did Keanu Reeves star in? 
  * Who has Tom Hanks worked with?       
  * Recommend a sci-fi movie             
  * Find me something romantic to watch  
  * A film about survival in the wild     

Commands: quit | exit | help
"""

st.title("🎬 Knowledge Graph + RAG Movie Assistant")

# Show help text in sidebar
st.sidebar.header("ℹ️ Help")
st.sidebar.text(HELP_TEXT)

# User input
query = st.text_input("Ask a movie question:")

if st.button("Run Query"):
    if query.strip():
        result = run_hybrid_rag(query, verbose=False)
        st.write("**Retriever:**", result["retriever"])
        st.write("**Context:**")
        st.text(result["context"])
        st.write("**Answer:**", result["answer"])
    else:
        st.warning("Please enter a question.")

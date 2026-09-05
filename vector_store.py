from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from graph_retriever import graph_retrieve_all_movies

_vector_store = None

def build_documents():
    movies = graph_retrieve_all_movies()
    docs = []
    for movie in movies:
        content = (
            f"Title: {movie.get('title','')}\n"
            f"Year: {movie.get('year','')}\n"
            f"Genre: {movie.get('genre','')}\n"
            f"Description: {movie.get('description','')}\n"
            f"Director: {movie.get('director','')}\n"
            f"Actors: {', '.join(movie.get('actors', []))}\n"
        )
        docs.append(Document(page_content=content))
    return docs

def create_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docs = build_documents()
    return FAISS.from_documents(docs, embeddings)

def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = create_vector_store()
    return _vector_store

def similarity_search(query: str, k: int = 4):
    vs = get_vector_store()
    return vs.similarity_search(query, k=k)

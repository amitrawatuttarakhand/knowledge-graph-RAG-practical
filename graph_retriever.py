"""
graph_retriever.py — Handles Neo4j queries for Hybrid Graph RAG.
"""

from neo4j import GraphDatabase

# Neo4j driver setup
def get_driver(uri, user, password):
    return GraphDatabase.driver(uri, auth=(user, password))

# Example function: retrieve actors of a movie
def graph_retrieve(query, uri, user, password):
    driver = get_driver(uri, user, password)
    cypher = """
    MATCH (m:Movie {title: $title})<-[:ACTED_IN]-(a:Actor)
    RETURN a.name AS actor
    """
    results = []
    with driver.session() as session:
        records = session.run(cypher, {"title": query})
        for record in records:
            results.append(record["actor"])
    driver.close()
    return results

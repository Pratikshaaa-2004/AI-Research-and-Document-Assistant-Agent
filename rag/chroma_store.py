"""
chroma_store.py

Standalone test script for ChromaDB.
Run this file directly (not through Streamlit) to confirm:
  1. ChromaDB can create/persist a collection on disk
  2. Text can be embedded and stored
  3. Similarity search returns sensible results

This file is NOT imported by the main app — it's purely a sandbox
for testing ChromaDB before we build the real retriever.py.
"""

from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

# ---------- CONFIG ----------
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_PATH = str((BASE_DIR / "chroma_db").resolve())
COLLECTION_NAME = "test_collection"  # a throwaway collection just for this test
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # small, fast, good-quality embedding model


def get_test_collection():
    """Creates (or loads) a persistent test collection."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"description": "Sandbox collection for testing ChromaDB"}
    )
    return collection


def seed_dummy_documents(collection):
    """Adds a few sample documents, but only if the collection is empty."""
    if collection.count() > 0:
        print(f"Collection already has {collection.count()} documents — skipping seed.")
        return

    documents = [
        "Machine learning allows computers to learn patterns from data without being explicitly programmed.",
        "Deep learning uses neural networks with multiple layers to model complex patterns.",
        "RAG retrieves relevant information from a knowledge source before generating an answer.",
        "ChromaDB is a vector database designed for storing and searching embeddings efficiently.",
        "Embeddings are numerical vector representations of text that capture semantic meaning.",
    ]

    ids = [f"doc{i+1}" for i in range(len(documents))]
    metadatas = [{"source": "seed_data", "index": i} for i in range(len(documents))]

    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )
    print(f"Seeded {len(documents)} documents into '{COLLECTION_NAME}'.")


def run_test_query(collection, question: str, top_k: int = 2):
    """Runs a similarity search and pretty-prints the results."""
    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )

    print(f"\nQuestion: {question}")
    print("-" * 60)

    documents = results["documents"][0]
    distances = results["distances"][0]

    if not documents:
        print("No results found.")
        return

    for rank, (doc, dist) in enumerate(zip(documents, distances), start=1):
        similarity_pct = max(0, (1 - dist)) * 100  # rough similarity estimate
        print(f"{rank}. (similarity ~{similarity_pct:.1f}%) {doc}")


if __name__ == "__main__":
    print("Connecting to ChromaDB...")
    collection = get_test_collection()

    print("Checking/seeding sample documents...")
    seed_dummy_documents(collection)

    print(f"\nTotal documents in collection: {collection.count()}")

    # Run a few test queries
    test_questions = [
        "What is RAG?",
        "Tell me about vector databases",
        "How do neural networks work?",
    ]

    for q in test_questions:
        run_test_query(collection, q)

    print("\nDone. If the results above look relevant, ChromaDB is working correctly.")
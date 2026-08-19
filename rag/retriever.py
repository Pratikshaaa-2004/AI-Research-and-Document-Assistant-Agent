"""
retriever.py

Loads and searches the ChromaDB data created by the teammate's
ingestion pipeline.
"""

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import TypedDict, List


# ---------- CONFIG ----------
CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class RetrievedChunk(TypedDict):
    text: str
    metadata: dict
    similarity: float


# ---------- SETUP ----------
_embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

_vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=_embeddings
)


def retrieve_chunks(
    question: str,
    top_k: int = 3
) -> List[RetrievedChunk]:

    if not question or not question.strip():
        return []

    results = _vectorstore.similarity_search_with_score(
        question,
        k=top_k
    )

    chunks: List[RetrievedChunk] = []

    for doc, score in results:

        similarity = round(
            max(0, (1 - score)) * 100,
            1
        )

        chunks.append({
            "text": doc.page_content,
            "metadata": doc.metadata or {},
            "similarity": similarity
        })

    return chunks


def collection_status() -> dict:

    try:
        count = _vectorstore._collection.count()
    except Exception:
        count = -1

    return {
        "collection_name": "langchain",
        "document_count": count
    }


# ---------- TEST ----------
if __name__ == "__main__":

    status = collection_status()

    print("Collection status:", status)

    if status["document_count"] == 0:

        print(
            "\nCollection is empty. "
            "Check the chroma_db folder."
        )

    else:

        question = "What is CRM?"

        print(f"\nQuestion: {question}")
        print("-" * 60)

        results = retrieve_chunks(
            question,
            top_k=3
        )

        if not results:
            print("No results returned.")

        for i, chunk in enumerate(results, start=1):

            source = chunk["metadata"].get(
                "source",
                "unknown"
            )

            page = chunk["metadata"].get(
                "page",
                "?"
            )

            print(
                f"{i}. "
                f"(similarity ~{chunk['similarity']}%) "
                f"[{source} - page {page}]"
            )

            print(
                f"   {chunk['text'][:150]}..."
            )
import os
from typing import TypedDict, List

from ollama import chat
from rag.retriever import retrieve_chunks


MODEL = "gemma3"


class Source(TypedDict):
    filename: str
    page: int


class AskResult(TypedDict):
    answer: str
    sources: List[Source]


def _build_sources(chunks: list) -> List[Source]:
    sources = []
    seen = set()

    for chunk in chunks:
        metadata = chunk.get("metadata", {})

        filename = os.path.basename(
            metadata.get("source", "demo_document.pdf")
        )

        raw_page = metadata.get("page", 0)

        if isinstance(raw_page, int):
            page = raw_page + 1
        else:
            page = 1

        key = (filename, page)

        if key not in seen:
            seen.add(key)

            sources.append({
                "filename": filename,
                "page": page
            })

    return sources


def _build_context(chunks: list) -> str:
    parts = []

    for i, chunk in enumerate(chunks, start=1):

        metadata = chunk.get("metadata", {})

        filename = os.path.basename(
            metadata.get("source", "demo_document.pdf")
        )

        page = metadata.get("page", 0) + 1

        parts.append(
            f"[Excerpt {i} - {filename}, page {page}]\n"
            f"{chunk['text']}"
        )

    return "\n\n".join(parts)


def ask_document(question: str) -> AskResult:

    if not question or not question.strip():
        return {
            "answer": "Please enter a question.",
            "sources": []
        }

    chunks = retrieve_chunks(
        question,
        top_k=4
    )

    # Temporary demo data
    if not chunks:
        chunks = [
            {
                "text": (
                    "CRM stands for Customer Relationship Management. "
                    "It is a system or approach used by organizations "
                    "to manage interactions with current and potential "
                    "customers. CRM helps organizations maintain "
                    "customer information, track interactions, and "
                    "improve customer relationships."
                ),
                "metadata": {
                    "source": "demo_document.pdf",
                    "page": 0
                }
            }
        ]

    context = _build_context(chunks)

    prompt = f"""
You are an AI research assistant.

Answer the user's question using ONLY the information
provided in the document excerpt below.

DOCUMENT EXCERPT:
{context}

QUESTION:
{question}

Rules:
- Use only the provided information.
- Do not invent information.
- Give a clear and concise answer.
"""

    try:

        response = chat(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response["message"]["content"].strip()

    except Exception as e:

        print("LLM error:", e)

        return {
            "answer": "The local AI model could not generate an answer.",
            "sources": []
        }

    return {
        "answer": answer,
        "sources": _build_sources(chunks)
    }


# ---------- TERMINAL CHAT ----------

if __name__ == "__main__":

    print("=" * 50)
    print("AI RESEARCH ASSISTANT")
    print("Type 'exit' to stop")
    print("=" * 50)

    while True:

        question = input("\nAsk your question: ")

        if question.lower().strip() == "exit":
            print("Goodbye!")
            break

        result = ask_document(question)

        print("\nANSWER:")
        print(result["answer"])

        print("\nSOURCES:")
        print(result["sources"])
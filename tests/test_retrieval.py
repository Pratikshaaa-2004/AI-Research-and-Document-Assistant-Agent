"""
test_retrieval.py

Quick sanity-check script for your RAG pipeline.
Run this instead of opening Streamlit every time you want to test.

Usage:
    python -m tests.test_retrieval
"""

from rag.retriever import retrieve_chunks, collection_status
from rag.ask_document import ask_document


def test_collection_has_data():
    status = collection_status()

    print(f"\n[TEST] Collection status: {status}")

    assert status["document_count"] > 0, \
        "❌ FAILED: No documents in collection"

    print("✅ PASSED: Collection has documents")


def test_retrieval_returns_results():
    question = "What is CRM?"

    chunks = retrieve_chunks(
        question,
        top_k=3
    )

    print(
        f"\n[TEST] retrieve_chunks('{question}') "
        f"returned {len(chunks)} chunks"
    )

    assert len(chunks) > 0, \
        "❌ FAILED: No chunks retrieved"

    for i, c in enumerate(chunks, 1):
        print(
            f"  {i}. similarity={c['similarity']}% | "
            f"{c['text'][:80]}..."
        )

    print("✅ PASSED: Retrieval returned results")


def test_ask_document_end_to_end():
    question = "What is CRM?"

    result = ask_document(question)

    print(
        f"\n[TEST] ask_document('{question}')"
    )

    print(
        f"  Answer: {result['answer'][:200]}..."
    )

    print(
        f"  Sources: {result['sources']}"
    )

    assert result["answer"], \
        "❌ FAILED: Empty answer"

    assert isinstance(result["sources"], list), \
        "❌ FAILED: Sources not a list"

    print(
        "✅ PASSED: ask_document returned a full result"
    )


def test_empty_question_handled():
    result = ask_document("")

    print(
        f"\n[TEST] ask_document('') -> {result}"
    )

    assert result["sources"] == [], \
        "❌ FAILED: Empty question should return no sources"

    print(
        "✅ PASSED: Empty question handled gracefully"
    )


def test_irrelevant_question_handled():
    question = (
        "asdkjhaskjdh random gibberish query xyz123"
    )

    result = ask_document(question)

    print(
        f"\n[TEST] ask_document('{question}')"
    )

    print(
        f"  Answer: {result['answer'][:150]}"
    )

    print(
        "✅ PASSED: Irrelevant question didn't crash"
    )


if __name__ == "__main__":

    print("=" * 60)
    print("RUNNING RETRIEVAL & RAG PIPELINE TESTS")
    print("=" * 60)

    tests = [
        test_collection_has_data,
        test_retrieval_returns_results,
        test_ask_document_end_to_end,
        test_empty_question_handled,
        test_irrelevant_question_handled,
    ]

    passed = 0
    failed = 0

    for test in tests:

        try:
            test()
            passed += 1

        except AssertionError as e:

            print(str(e))
            failed += 1

        except Exception as e:

            print(
                f"❌ ERROR in {test.__name__}: {e}"
            )

            failed += 1

    print("\n" + "=" * 60)

    print(
        f"RESULTS: {passed} passed, {failed} failed"
    )

    print("=" * 60)
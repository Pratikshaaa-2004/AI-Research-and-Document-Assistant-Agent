from rag.retriever import retrieve_chunks


def test_retriever_exports_retrieve_chunks():
    assert callable(retrieve_chunks)

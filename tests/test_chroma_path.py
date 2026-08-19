from pathlib import Path

from rag import chroma_store


def test_chroma_path_is_project_relative():
    expected = (Path(chroma_store.__file__).resolve().parents[1] / "chroma_db").resolve()
    assert Path(chroma_store.CHROMA_PATH).resolve() == expected

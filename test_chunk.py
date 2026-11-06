from langchain_core.documents import Document

from src.chunk import chunk_docs

LONG = "Sentence about attention. " * 200


def _docs():
    return [Document(page_content=LONG, metadata={"doc": "x.pdf", "page": 4}),
            Document(page_content="tiny", metadata={"doc": "x.pdf", "page": 5})]


def test_fixed_chunking_splits_and_keeps_metadata():
    chunks = chunk_docs(_docs(), "fixed", 500, 100, min_chars=80)
    assert len(chunks) > 1
    assert all(c.metadata["page"] == 4 for c in chunks)   # tiny page 5 dropped


def test_min_chars_filter():
    assert all(len(c.page_content) >= 80 for c in chunk_docs(_docs(), "fixed", 500, 100, 80))


def test_per_page_keeps_one_chunk_per_page():
    chunks = chunk_docs(_docs(), "per_page", min_chars=1)
    assert len(chunks) == 2

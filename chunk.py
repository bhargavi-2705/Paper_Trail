"""Baseline chunkers. Splitting happens per page Document, so chunks never
cross page boundaries and every chunk keeps an exact (doc, page) citation.
Known limitation for prose books (ideas split across pages) -- to be tested
in an ablation."""
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_docs(docs, strategy="fixed", chunk_chars=1200, overlap_chars=200, min_chars=30):
    if strategy == "per_page":
        chunks = list(docs)
    elif strategy == "fixed":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_chars,
            chunk_overlap=overlap_chars,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(docs)
    else:
        raise ValueError(f"unknown chunking strategy: {strategy}")
    return [c for c in chunks if len(c.page_content) >= min_chars]

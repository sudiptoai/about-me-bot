"""
Unit tests for the RAG pipeline (src/rag.py).

These tests use only numpy and standard-library primitives – no API calls,
no sentence-transformers model download.  The Embedder is monkey-patched with
a deterministic stub.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.rag import Document, DocumentLoader, Embedder, RAGPipeline, VectorStore


# ---------------------------------------------------------------------------
# DocumentLoader tests
# ---------------------------------------------------------------------------


class TestDocumentLoader:
    def test_load_returns_documents(self, tmp_path: Path) -> None:
        md = tmp_path / "test.md"
        md.write_text("# Hello\n\nThis is paragraph one.\n\nThis is paragraph two.", encoding="utf-8")
        loader = DocumentLoader(chunk_size=200, chunk_overlap=20)
        docs = loader.load(md)
        assert len(docs) >= 1
        assert all(isinstance(d, Document) for d in docs)

    def test_chunk_indices_are_sequential(self, tmp_path: Path) -> None:
        text = "\n\n".join(f"Paragraph {i}." for i in range(20))
        md = tmp_path / "big.md"
        md.write_text(text, encoding="utf-8")
        loader = DocumentLoader(chunk_size=50, chunk_overlap=10)
        docs = loader.load(md)
        assert [d.chunk_index for d in docs] == list(range(len(docs)))

    def test_source_is_set(self, tmp_path: Path) -> None:
        md = tmp_path / "bio.md"
        md.write_text("Some content here.", encoding="utf-8")
        loader = DocumentLoader()
        docs = loader.load(md)
        assert docs[0].source == str(md)

    def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        md = tmp_path / "empty.md"
        md.write_text("", encoding="utf-8")
        loader = DocumentLoader()
        docs = loader.load(md)
        assert docs == []

    def test_split_normalises_blank_lines(self) -> None:
        loader = DocumentLoader(chunk_size=500, chunk_overlap=50)
        chunks = loader._split("Para one.\n\n\n\n\nPara two.")
        assert len(chunks) == 1
        assert "Para one." in chunks[0]
        assert "Para two." in chunks[0]


# ---------------------------------------------------------------------------
# VectorStore tests
# ---------------------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    """Return *v* normalised to unit length."""
    norm = np.linalg.norm(v)
    return (v / norm).astype(np.float32)


class TestVectorStore:
    def _make_docs(self, n: int) -> list[Document]:
        return [Document(content=f"doc {i}", source="test", chunk_index=i) for i in range(n)]

    def _make_embeddings(self, n: int, dim: int = 4) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.random((n, dim)).astype(np.float32)

    def test_add_and_size(self) -> None:
        store = VectorStore()
        docs = self._make_docs(3)
        embs = self._make_embeddings(3)
        store.add(docs, embs)
        assert store.size == 3

    def test_search_returns_top_k(self) -> None:
        store = VectorStore()
        docs = self._make_docs(10)
        embs = self._make_embeddings(10, dim=8)
        store.add(docs, embs)
        query = self._make_embeddings(1, dim=8)[0]
        results = store.search(query, top_k=3)
        assert len(results) == 3

    def test_search_scores_descending(self) -> None:
        store = VectorStore()
        docs = self._make_docs(5)
        embs = self._make_embeddings(5, dim=8)
        store.add(docs, embs)
        query = self._make_embeddings(1, dim=8)[0]
        results = store.search(query, top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_exact_match(self) -> None:
        """A query identical to a stored vector should get score ~1.0."""
        store = VectorStore()
        docs = self._make_docs(3)
        embs = np.eye(3, dtype=np.float32)  # orthonormal basis
        store.add(docs, embs)
        # Query equal to the second vector
        query = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        results = store.search(query, top_k=1)
        assert results[0].document.content == "doc 1"
        assert math.isclose(results[0].score, 1.0, abs_tol=1e-5)

    def test_add_mismatch_raises(self) -> None:
        store = VectorStore()
        with pytest.raises(ValueError):
            store.add(self._make_docs(3), self._make_embeddings(5))

    def test_search_empty_store(self) -> None:
        store = VectorStore()
        results = store.search(np.ones(4, dtype=np.float32), top_k=3)
        assert results == []

    def test_search_zero_query(self) -> None:
        store = VectorStore()
        docs = self._make_docs(2)
        store.add(docs, self._make_embeddings(2))
        results = store.search(np.zeros(4, dtype=np.float32), top_k=1)
        assert results == []


# ---------------------------------------------------------------------------
# RAGPipeline tests (Embedder stubbed)
# ---------------------------------------------------------------------------


def _stub_embedder(texts: list[str]) -> np.ndarray:
    """Deterministic stub: each text gets a unique one-hot embedding."""
    dim = max(len(texts), 1)
    embs = np.zeros((len(texts), dim), dtype=np.float32)
    for i in range(len(texts)):
        embs[i, i % dim] = 1.0
    return embs


class TestRAGPipeline:
    def _pipeline(self, tmp_path: Path, content: str) -> RAGPipeline:
        md = tmp_path / "bio.md"
        md.write_text(content, encoding="utf-8")

        pipe = RAGPipeline.__new__(RAGPipeline)
        pipe.top_k = 3
        from src.rag import DocumentLoader, VectorStore

        pipe._loader = DocumentLoader(chunk_size=200, chunk_overlap=20)
        pipe._store = VectorStore()
        # Stub the embedder
        stub = MagicMock()
        stub.embed.side_effect = _stub_embedder
        pipe._embedder = stub
        pipe.load(md)
        return pipe

    def test_document_count_positive(self, tmp_path: Path) -> None:
        text = "\n\n".join(f"Section {i}: some content." for i in range(5))
        pipe = self._pipeline(tmp_path, text)
        assert pipe.document_count > 0

    def test_retrieve_returns_results(self, tmp_path: Path) -> None:
        text = "\n\n".join(f"Fact {i}: interesting detail." for i in range(6))
        pipe = self._pipeline(tmp_path, text)
        results = pipe.retrieve("interesting detail")
        assert len(results) >= 1

    def test_get_context_is_string(self, tmp_path: Path) -> None:
        text = "Personal info.\n\nProfessional info.\n\nHobbies."
        pipe = self._pipeline(tmp_path, text)
        ctx = pipe.get_context("hobbies")
        assert isinstance(ctx, str)
        assert len(ctx) > 0

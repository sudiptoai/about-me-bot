"""
RAG (Retrieval-Augmented Generation) pipeline for the About-Me Bot.

Responsibilities:
- Load and chunk a Markdown knowledge-base file
- Embed each chunk with sentence-transformers
- Retrieve the most relevant chunks for a query using cosine similarity
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Document:
    """A single chunk of text with its source metadata."""

    content: str
    source: str = ""
    chunk_index: int = 0


@dataclass
class RetrievalResult:
    """A retrieved document together with its similarity score."""

    document: Document
    score: float


# ---------------------------------------------------------------------------
# Document loading & chunking
# ---------------------------------------------------------------------------


class DocumentLoader:
    """Load a Markdown file and split it into overlapping text chunks."""

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 80) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, path: str | Path) -> List[Document]:
        """Load *path* and return a list of :class:`Document` chunks."""
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        chunks = self._split(text)
        return [
            Document(content=chunk, source=str(path), chunk_index=i)
            for i, chunk in enumerate(chunks)
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split(self, text: str) -> List[str]:
        """Split *text* into overlapping chunks, preferring paragraph breaks."""
        # Normalise line endings and collapse excessive blank lines
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        paragraphs = text.split("\n\n")
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            para_len = len(para)

            if current_len + para_len > self.chunk_size and current:
                chunks.append("\n\n".join(current))
                # Keep overlap: drop paragraphs from the front until we are
                # within the overlap budget.
                while current and current_len > self.chunk_overlap:
                    removed = current.pop(0)
                    current_len -= len(removed) + 2  # +2 for "\n\n"

            current.append(para)
            current_len += para_len + 2

        if current:
            chunks.append("\n\n".join(current))

        return chunks


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------


class Embedder:
    """Thin wrapper around a sentence-transformers model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        # Lazily import so unit tests can mock this without the full model.
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        """Return an (N, D) float32 embedding matrix for *texts*."""
        embeddings = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# Vector store (in-memory, NumPy cosine similarity)
# ---------------------------------------------------------------------------


class VectorStore:
    """Simple in-memory vector store backed by NumPy arrays.

    Documents are pre-normalised, so retrieval is a single matrix-vector
    dot product (equivalent to cosine similarity for unit vectors).
    """

    def __init__(self) -> None:
        self._documents: List[Document] = []
        self._matrix: np.ndarray | None = None  # shape (N, D)

    # ------------------------------------------------------------------
    # Building the index
    # ------------------------------------------------------------------

    def add(self, documents: List[Document], embeddings: np.ndarray) -> None:
        """Append *documents* and their *embeddings* to the store."""
        if len(documents) != embeddings.shape[0]:
            raise ValueError(
                f"Number of documents ({len(documents)}) must match "
                f"number of embeddings ({embeddings.shape[0]})"
            )
        # Normalise rows to unit length for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalised = (embeddings / norms).astype(np.float32)

        self._documents.extend(documents)
        if self._matrix is None:
            self._matrix = normalised
        else:
            self._matrix = np.vstack([self._matrix, normalised])

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[RetrievalResult]:
        """Return the *top_k* most similar documents to *query_embedding*."""
        if self._matrix is None or len(self._documents) == 0:
            return []

        # Normalise query vector
        norm = float(np.linalg.norm(query_embedding))
        if norm == 0:
            return []
        q = (query_embedding / norm).astype(np.float32).flatten()

        scores: np.ndarray = self._matrix @ q  # (N,)

        top_k = min(top_k, len(self._documents))
        # Descending order; argpartition is O(N) then we sort only top_k
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [
            RetrievalResult(document=self._documents[i], score=float(scores[i]))
            for i in top_indices
        ]

    @property
    def size(self) -> int:
        return len(self._documents)


# ---------------------------------------------------------------------------
# RAG Pipeline (facade)
# ---------------------------------------------------------------------------


class RAGPipeline:
    """End-to-end RAG pipeline: load → embed → index → retrieve."""

    def __init__(
        self,
        biodata_path: str | Path | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 400,
        chunk_overlap: int = 80,
        top_k: int = 5,
    ) -> None:
        self.top_k = top_k
        self._loader = DocumentLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._embedder = Embedder(model_name=embedding_model)
        self._store = VectorStore()

        if biodata_path is not None:
            self.load(biodata_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, path: str | Path) -> None:
        """Load *path* into the vector store (can be called multiple times)."""
        documents = self._loader.load(path)
        if not documents:
            return
        texts = [doc.content for doc in documents]
        embeddings = self._embedder.embed(texts)
        self._store.add(documents, embeddings)

    def retrieve(self, query: str, top_k: int | None = None) -> List[RetrievalResult]:
        """Return the *top_k* most relevant chunks for *query*."""
        k = top_k if top_k is not None else self.top_k
        query_embedding = self._embedder.embed([query])[0]
        return self._store.search(query_embedding, top_k=k)

    def get_context(self, query: str, top_k: int | None = None) -> str:
        """Return retrieved chunks joined as a single context string."""
        results = self.retrieve(query, top_k=top_k)
        return "\n\n---\n\n".join(r.document.content for r in results)

    @property
    def document_count(self) -> int:
        return self._store.size

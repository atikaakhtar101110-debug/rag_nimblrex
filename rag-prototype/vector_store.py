"""A tiny in-memory FAISS vector store (cosine similarity)."""

from __future__ import annotations

import faiss
import numpy as np

from ingestion import Chunk


class VectorStore:
    def __init__(self) -> None:
        self._index: faiss.Index | None = None
        self._chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def sources(self) -> list[str]:
        return sorted({c.source for c in self._chunks})

    @staticmethod
    def _prepare(vectors: list[list[float]]) -> np.ndarray:
        arr = np.asarray(vectors, dtype="float32")
        if arr.ndim != 2:
            raise ValueError("Embeddings must be a 2-D array.")
        faiss.normalize_L2(arr)  # inner product on unit vectors == cosine similarity
        return arr

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length.")
        if not chunks:
            return
        arr = self._prepare(vectors)
        if self._index is None:
            self._index = faiss.IndexFlatIP(arr.shape[1])
        elif arr.shape[1] != self._index.d:
            raise ValueError("Embedding dimension does not match the existing index.")
        self._index.add(arr)
        self._chunks.extend(chunks)

    def search(self, query_vector: list[float], k: int = 4) -> list[tuple[Chunk, float]]:
        if self._index is None or not self._chunks:
            return []
        q = self._prepare([query_vector])
        scores, ids = self._index.search(q, min(k, len(self._chunks)))
        return [
            (self._chunks[i], float(s))
            for s, i in zip(scores[0], ids[0])
            if i != -1
        ]

    def remove_source(self, source: str) -> None:
        """Remove all chunks from one file by rebuilding the index."""
        if self._index is None:
            return
        keep = [i for i, c in enumerate(self._chunks) if c.source != source]
        if len(keep) == len(self._chunks):
            return
        vectors = np.vstack([self._index.reconstruct(i) for i in keep]) if keep else None
        self._chunks = [self._chunks[i] for i in keep]
        if vectors is None:
            self._index = None
        else:
            self._index = faiss.IndexFlatIP(vectors.shape[1])
            self._index.add(vectors.astype("float32"))

    def clear(self) -> None:
        self._index = None
        self._chunks = []

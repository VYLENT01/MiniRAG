"""
Embedding Cache Proxy.

Implements the Proxy pattern to optionally cache embeddings to disk.
This allows rebuilding vector stores or switching vector DBs without 
re-computing embeddings, saving significant time and API costs.
"""

import uuid
from pathlib import Path
from typing import List, Optional

import numpy as np

from minirag.embeddings.base import BaseEmbedding


class CachedEmbedding(BaseEmbedding):
    """A wrapper that adds disk-caching functionality to any BaseEmbedding."""

    def __init__(self, inner_embedder: BaseEmbedding, cache_dir: Path, model_name: str):
        self.inner_embedder = inner_embedder
        self.model_name = model_name.replace("/", "_") # Sanitize folder name
        self.cache_dir = cache_dir / self.model_name

    def embed(self, texts: List[str], doc_id: Optional[uuid.UUID] = None) -> np.ndarray:
        # If no doc_id is provided (e.g., during a query), we cannot cache.
        # Just pass through to the real embedder.
        if doc_id is None or not texts:
            return self.inner_embedder.embed(texts, doc_id)

        cache_file = self.cache_dir / f"{doc_id}.npy"

        # 1. Check if cache exists
        if cache_file.exists():
            try:
                return np.load(cache_file)
            except Exception:
                pass # If file is corrupted, fall through to re-compute

        # 2. Cache miss: Compute embeddings using the real model
        embeddings = self.inner_embedder.embed(texts, doc_id)

        # 3. Save to cache
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, embeddings)
        except Exception as e:
            # Cache saving should never crash the main pipeline
            print(f"Warning: Failed to save embedding cache to {cache_file}: {e}")

        return embeddings

    @property
    def dimension(self) -> int:
        return self.inner_embedder.dimension
"""
FAISS Vector Store Implementation.

Uses FAISS IndexFlatL2 for exact nearest neighbor search.
Educational note: IndexFlatL2 is brute-force and perfectly accurate, 
making it ideal for V1. Future versions can swap this for IndexIVFFlat 
for faster, approximate search on millions of vectors.
"""

import json
import uuid
from pathlib import Path
from typing import List, Tuple

import numpy as np

from minirag.vector_stores.base import BaseVectorStore
from minirag.exceptions import VectorStoreError

try:
    import faiss
except ImportError:
    raise ImportError(
        "faiss-cpu is required for the vector store. "
        "Please install it via: pip install faiss-cpu"
    )


class FAISSStore(BaseVectorStore):
    """Vector store implementation using FAISS."""

    def __init__(self, index_dir: Path, dimension: int):
        self.index_dir = index_dir
        self.dimension = dimension
        
        self.index_path = self.index_dir / "faiss.index"
        self.id_map_path = self.index_dir / "faiss_id_map.json"
        
        self._index: faiss.IndexFlatL2 = None  # type: ignore
        self._id_map: List[str] = [] # Stores UUIDs as strings for JSON compatibility

    def _init_index(self) -> None:
        """Initialize an empty FAISS index."""
        # L2 distance requires float32 vectors
        self._index = faiss.IndexFlatL2(self.dimension)
        self._id_map = []

    def add(self, ids: List[uuid.UUID], vectors: np.ndarray) -> None:
        if self._index is None:
            self._init_index()

        if not ids or len(vectors) == 0:
            return

        # FAISS strictly requires contiguous float32 arrays
        vectors = np.ascontiguousarray(vectors, dtype='float32')
        
        # Add to FAISS internal index
        self._index.add(vectors)
        
        # Update our Python-side UUID map
        for chunk_id in ids:
            self._id_map.append(str(chunk_id))

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[uuid.UUID, float]]:
        if self._index is None or self._index.ntotal == 0:
            return []

        # Reshape query vector to 2D (1, dimension) and ensure float32
        q_vec = np.ascontiguousarray(query_vector.reshape(1, -1), dtype='float32')
        
        # FAISS returns distances (lower is better for L2) and internal integer IDs
        distances, internal_ids = self._index.search(q_vec, top_k)
        
        results = []
        for dist, int_id in zip(distances[0], internal_ids[0]):
            if int_id == -1: # FAISS returns -1 if fewer vectors exist than top_k
                continue
                
            chunk_uuid = uuid.UUID(self._id_map[int_id])
            
            # Convert L2 distance to a similarity score (0 to 1 range)
            # Formula: 1 / (1 + distance). Simple and effective for UI display.
            similarity = 1.0 / (1.0 + float(dist))
            
            results.append((chunk_uuid, similarity))
            
        return results

    def save(self) -> None:
        if self._index is None:
            return

        try:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(self.index_path))
            with open(self.id_map_path, 'w') as f:
                json.dump(self._id_map, f)
        except Exception as e:
            raise VectorStoreError(f"Failed to save FAISS index: {e}") from e

    def load(self) -> bool:
        if not self.index_path.exists() or not self.id_map_path.exists():
            return False

        try:
            self._index = faiss.read_index(str(self.index_path))
            with open(self.id_map_path, 'r') as f:
                self._id_map = json.load(f)
            return True
        except Exception as e:
            # If loading fails, re-initialize to a safe empty state
            self._init_index()
            raise VectorStoreError(f"Failed to load FAISS index: {e}") from e
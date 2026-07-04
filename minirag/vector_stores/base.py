"""
Base Vector Store Interface.

Defines the contract for storing and searching vectors.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import uuid
import numpy as np


class BaseVectorStore(ABC):
    """Abstract base class for vector databases."""

    @abstractmethod
    def add(self, ids: List[uuid.UUID], vectors: np.ndarray) -> None:
        """
        Add vectors to the index.

        Args:
            ids: List of UUIDs corresponding to the vectors.
            vectors: 2D numpy array of vectors to add.
        """
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[uuid.UUID, float]]:
        """
        Search for the most similar vectors.

        Args:
            query_vector: 1D numpy array representing the query.
            top_k: Number of results to return.

        Returns:
            A list of tuples containing (chunk_uuid, similarity_score).
            Similarity score should be between 0.0 and 1.0 (higher is better).
        """
        pass

    @abstractmethod
    def save(self) -> None:
        """Persist the index to disk."""
        pass

    @abstractmethod
    def load(self) -> bool:
        """
        Load the index from disk.
        
        Returns:
            True if an existing index was loaded, False otherwise.
        """
        pass
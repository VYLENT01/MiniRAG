"""
Base Embedding Interface.

Defines the contract for converting text to vector representations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import uuid
import numpy as np


class BaseEmbedding(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def embed(self, texts: List[str], doc_id: Optional[uuid.UUID] = None) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed.
            doc_id: Optional document UUID. Used internally by caching proxies 
                    to know where to save the file. Ignored by base models.

        Returns:
            A 2D numpy array of shape (len(texts), embedding_dimension).
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the size of the embedding vectors."""
        pass
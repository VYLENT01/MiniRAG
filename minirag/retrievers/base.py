"""
Base Retriever Interface.

Defines the contract for searching the knowledge base.
"""

from abc import ABC, abstractmethod
from typing import List

from minirag.models.search import SearchResult


class BaseRetriever(ABC):
    """Abstract base class for retrievers."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Search for chunks relevant to the query.

        Args:
            query: The user's question.
            top_k: Number of top results to return.

        Returns:
            A list of SearchResult objects, sorted by similarity (highest first).
        """
        pass
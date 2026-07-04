"""
Base Chunker Interface.

Defines the contract for splitting text into smaller pieces.
"""

from abc import ABC, abstractmethod
from typing import List

from minirag.models.chunk import Chunk
from minirag.models.document import DocumentMetadata


class BaseChunker(ABC):
    """Abstract base class for text chunkers."""

    @abstractmethod
    def chunk(self, text: str, doc_metadata: DocumentMetadata) -> List[Chunk]:
        """
        Split text into chunks and attach metadata.

        Args:
            text: The cleaned text to split.
            doc_metadata: The metadata of the source document.

        Returns:
            A list of Chunk objects.
        """
        pass
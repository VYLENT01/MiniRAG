"""
Base Cleaner Interface.

Defines the contract for text normalization.
"""

from abc import ABC, abstractmethod


class BaseCleaner(ABC):
    """Abstract base class for text cleaners."""

    @abstractmethod
    def clean(self, text: str) -> str:
        """
        Normalize and clean extracted text.

        Args:
            text: Raw text from a loader.

        Returns:
            Cleaned text.
        """
        pass
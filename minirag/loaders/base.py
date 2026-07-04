"""
Base Loader Interface.

Defines the contract for all document loaders.
Loaders are responsible ONLY for converting file bytes into raw text.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from minirag.exceptions import LoaderError


class BaseLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self, file_path: Path) -> str:
        """
        Extract raw text from a document.

        Args:
            file_path: Absolute path to the document.

        Returns:
            The extracted raw text as a single string.

        Raises:
            LoaderError: If the file cannot be read or parsed.
        """
        pass
"""
Text Loader Implementation.

Reads plain text files with UTF-8 encoding fallback.
"""

from pathlib import Path

from minirag.exceptions import LoaderError
from minirag.loaders.base import BaseLoader


class TextLoader(BaseLoader):
    """Loads text from .txt files."""

    def load(self, file_path: Path) -> str:
        if not file_path.exists():
            raise LoaderError(f"File not found: {file_path}")

        try:
            # Attempt UTF-8 first, fallback to latin-1 which rarely fails
            try:
                return file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return file_path.read_text(encoding="latin-1")
        except Exception as e:
            raise LoaderError(f"Failed to read text file {file_path}: {e}") from e
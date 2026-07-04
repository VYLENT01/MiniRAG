"""
Loaders Factory.

Maps file extensions to their appropriate loader implementations.
"""

from pathlib import Path
from minirag.loaders.base import BaseLoader
from minirag.loaders.pdf_loader import PDFLoader
from minirag.loaders.text_loader import TextLoader
from minirag.loaders.markdown_loader import MarkdownLoader
from minirag.loaders.json_loader import JSONLoader
from minirag.exceptions import LoaderError

# Registry mapping extensions to loader classes
_LOADER_REGISTRY = {
    ".pdf": PDFLoader,
    ".txt": TextLoader,
    ".md": MarkdownLoader,
    ".markdown": MarkdownLoader,
    ".json": JSONLoader,
}


def get_loader(file_path: Path) -> BaseLoader:
    """
    Factory function to retrieve the correct loader based on file extension.

    Args:
        file_path: Path to the document.

    Returns:
        An instance of a BaseLoader.

    Raises:
        LoaderError: If the file type is not supported.
    """
    ext = file_path.suffix.lower()
    loader_class = _LOADER_REGISTRY.get(ext)
    
    if not loader_class:
        supported = ", ".join(_LOADER_REGISTRY.keys())
        raise LoaderError(
            f"Unsupported file type: '{ext}'. Supported types: {supported}"
        )
        
    return loader_class()
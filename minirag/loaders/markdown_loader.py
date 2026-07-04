"""
Markdown Loader Implementation.

In V1, Markdown is treated as plain text. 
Future versions might parse the AST to preserve heading structures for chunking.
"""

from pathlib import Path

from minirag.loaders.base import BaseLoader
from minirag.loaders.text_loader import TextLoader


class MarkdownLoader(TextLoader):
    """Loads text from .md files. Inherits UTF-8 handling from TextLoader."""
    pass
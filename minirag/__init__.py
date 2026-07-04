"""
MiniRAG - A lightweight, educational RAG engine.

This is the only module Vylent or the end-user should import.
"""

from minirag.facade import MiniRAG
from minirag.config import Config
from minirag.exceptions import MiniRAGError

__version__ = "1.0.0"

__all__ = [
    "MiniRAG",
    "Config",
    "MiniRAGError"
]
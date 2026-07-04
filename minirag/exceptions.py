"""
MiniRAG Exception Hierarchy.

All custom exceptions in MiniRAG inherit from MiniRAGError.
This allows the public API (facade) to catch a single exception type
while allowing internal modules to raise highly specific errors.
"""


class MiniRAGError(Exception):
    """Base exception for all MiniRAG operations."""
    pass


class ConfigurationError(MiniRAGError):
    """Raised when there is an issue with the provided configuration."""
    pass


class DocumentNotFoundError(MiniRAGError):
    """Raised when a requested document ID does not exist in the registry."""
    pass


class DuplicateDocumentError(MiniRAGError):
    """Raised when attempting to index a document that is already indexed (based on SHA256)."""
    pass


class LoaderError(MiniRAGError):
    """Raised when a document fails to load or parse."""
    pass


class CleanerError(MiniRAGError):
    """Raised when text cleaning fails unexpectedly."""
    pass


class ChunkerError(MiniRAGError):
    """Raised when text chunking fails."""
    pass


class EmbeddingError(MiniRAGError):
    """Raised when generating embeddings fails (e.g., model not found, API error)."""
    pass


class VectorStoreError(MiniRAGError):
    """Raised when interacting with the vector database fails."""
    pass


class RetrieverError(MiniRAGError):
    """Raised when the retrieval step fails."""
    pass


class LLMError(MiniRAGError):
    """Raised when the LLM fails to generate a response."""
    pass


class PipelineError(MiniRAGError):
    """Raised when a high-level pipeline (indexing or querying) fails."""
    pass
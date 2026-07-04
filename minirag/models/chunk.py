"""
Chunk Data Model.

Represents a single segment of a document after the chunking phase.
Contains both the actual text content and its associated metadata.
"""

from __future__ import annotations  # <- این خط مشکل را حل می‌کند
import uuid
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ChunkMetadata:
    """Immutable metadata for a specific chunk."""
    
    uuid: uuid.UUID
    document_uuid: uuid.UUID
    chunk_index: int  # 0-based index of the chunk in the original document
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section: Optional[str] = None
    heading: Optional[str] = None
    char_count: int = 0
    token_count: Optional[int] = None  # Optional in V1, requires a tokenizer to populate accurately
    content_type: str = "text"  # e.g., 'text', 'code', 'table'
    sha256: str = ""  # Hash of the chunk text for integrity checks


@dataclass
class Chunk:
    """
    Mutable container for chunk data. 
    The text is mutable to allow cleaning processes to modify it,
    but the metadata assignment is typically done once.
    """
    text: str
    metadata: ChunkMetadata

    @property
    def uuid(self) -> uuid.UUID:
        return self.metadata.uuid

    @property
    def document_uuid(self) -> uuid.UUID:
        return self.metadata.document_uuid
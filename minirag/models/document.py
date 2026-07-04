"""
Document Metadata Model.

Represents the metadata stored in the registry for a single document.
It does NOT contain the document content, only the information needed 
to manage its lifecycle.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DocumentMetadata:
    """Immutable data structure holding document metadata."""
    
    uuid: uuid.UUID
    file_name: str
    absolute_path: Path
    sha256: str
    file_type: str  # e.g., 'pdf', 'markdown', 'txt', 'json'
    language: str = "en"
    file_size: int = 0  # in bytes
    created_date: Optional[datetime] = None  # OS file creation time
    indexed_date: datetime = field(default_factory=datetime.now)
    
    # Pipeline parameters used during indexing (important for rebuilds)
    num_pages: Optional[int] = None  # None for non-paginated formats like txt/md
    chunk_count: int = 0
    embedding_model: str = "unknown"
    chunk_size: int = 0
    chunk_overlap: int = 0
    
    # Schema versioning for future migrations
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        # Ensure absolute_path is strictly a Path object, even if passed as string
        object.__setattr__(self, 'absolute_path', Path(self.absolute_path))
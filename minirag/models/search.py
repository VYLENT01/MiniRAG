"""
Search Result Model.

Defines the structure of a single result returned by the Retriever.
It bridges the gap between internal vector stores and the external API.
"""

import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class SearchResult:
    """Represents a single retrieved chunk relevant to a query."""
    
    text: str
    similarity_score: float
    chunk_id: uuid.UUID
    document_name: str
    page: Optional[int] = None
    section: Optional[str] = None
    metadata: Dict[str, Any] = None  # Catch-all for any extra metadata needed by prompts

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
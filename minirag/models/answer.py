"""
Answer and Citation Models.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import uuid


@dataclass
class Citation:
    """Represents a single source citation attached to the answer."""
    
    document_name: str
    page: Optional[int] = None
    similarity_score: float = 0.0
    chunk_id: uuid.UUID = None  # type: ignore
    exact_snippet: str = "" 


@dataclass
class Answer:
    """The final formatted output returned to the user."""
    
    text: str
    is_faithful: bool = True
    citations: List[Citation] = field(default_factory=list)
    
    # ویژگی‌های جدید برای V1.1
    confidence_score: float = 0.0  # میانگین امتیازات شباهت (بین 0 تا 1)
    confidence_level: str = "LOW" # LOW, MEDIUM, HIGH
    
    def add_citation(self, citation: Citation) -> None:
        """Helper method to safely append a citation."""
        self.citations.append(citation)
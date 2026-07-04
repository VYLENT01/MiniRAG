"""
Answer, Citation, and Trace Models.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import uuid


@dataclass
class Citation:
    document_name: str
    page: Optional[int] = None
    similarity_score: float = 0.0
    chunk_id: uuid.UUID = None  # type: ignore
    exact_snippet: str = "" 

@dataclass
class RejectionReason:
    """Stores exactly why a specific LLM quote was rejected."""
    quote: str
    reason: str  # e.g., "Too short", "Missing in corpus"

@dataclass
class DebugTrace:
    """Captures the internal decision-making process of the pipeline."""
    llm_raw_json: str = ""
    extracted_quotes: List[str] = field(default_factory=list)
    rejected_quotes: List[RejectionReason] = field(default_factory=list)
    retrieval_scores: List[float] = field(default_factory=list)

@dataclass
class Answer:
    text: str
    is_faithful: bool = True
    citations: List[Citation] = field(default_factory=list)
    
    # V1.1 Features
    confidence_score: float = 0.0
    confidence_level: str = "LOW"
    
    # V1.1 Debugging
    trace: DebugTrace = field(default_factory=DebugTrace)
    
    def add_citation(self, citation: Citation) -> None:
        self.citations.append(citation)
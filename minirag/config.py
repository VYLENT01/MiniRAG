"""
MiniRAG Configuration.

A centralized, immutable configuration object. 
Defaults are provided for out-of-the-box local execution.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Config:
    """Immutable configuration for the MiniRAG engine."""

    # --- Data Directories ---
    base_data_dir: Path = field(default_factory=lambda: Path.cwd() / "minirag_data")
    
    @property
    def chunks_dir(self) -> Path:
        return self.base_data_dir / "chunks"

    @property
    def embeddings_dir(self) -> Path:
        return self.base_data_dir / "embeddings"

    @property
    def vector_db_dir(self) -> Path:
        return self.base_data_dir / "vector_db"

    @property
    def metadata_dir(self) -> Path:
        return self.base_data_dir / "metadata"

    @property
    def logs_dir(self) -> Path:
        return self.base_data_dir / "logs"

    # --- Chunking Parameters ---
    chunk_size: int = 512
    chunk_overlap: int = 50

    # --- Embedding Parameters ---
    embedding_provider: str = "bge-m3"
    use_embedding_cache: bool = True

    # --- Vector Store Parameters ---
    vector_store_provider: str = "faiss"
    
    # --- Retrieval Parameters ---
    top_k: int = 5
    similarity_threshold: float = 0.40
    retriever_provider: str = "hybrid"  # گزینه‌ها: "semantic", "bm25", "hybrid"
    
    # --- LLM Parameters ---
    primary_llm_provider: str = "ollama"
    secondary_llm_provider: Optional[str] = "gemini"
    ollama_model: str = "llama3"
    gemini_model: str = "gemini-2.0-flash"
    gemini_api_key: Optional[str] = "YOUR_API_KEY"
    llm_timeout: int = 30

    # --- Generation Parameters ---
    temperature: float = 0.7

    # --- System Parameters ---
    debug_mode: bool = False
    schema_version: str = "1.2" # آپدیت ورژن

    def __post_init__(self) -> None:
        object.__setattr__(self, 'base_data_dir', Path(self.base_data_dir))
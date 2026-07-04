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
    # Defaults to a 'data' folder in the current working directory
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
    embedding_provider: str = "bge-m3"  # Maps to implementations in embeddings/
    use_embedding_cache: bool = True    # Implements the optional .npy caching logic

    # --- Vector Store Parameters ---
    vector_store_provider: str = "faiss"

    # --- LLM Parameters ---
    primary_llm_provider: str = "ollama"  # تغییر از gemini به ollama
    secondary_llm_provider: Optional[str] = "gemini"  # چون فقط gemini دارید، این را None بگذارید
    ollama_model: str = "llama3"
    gemini_model: str = "gemini-2.0-flash"
    
    # ---------------------------------------------------------
    # 👇 اینجا جای دقیق قرارگیری کلید API است (به جای None بگذارید)
    # ---------------------------------------------------------
    gemini_api_key: Optional[str] = "YOUR_API_KEY"  # جایگزین با کلید واقعی خود کنید
    
    llm_timeout: int = 30  # seconds before fallback triggers

    # --- Retrieval Parameters ---
    top_k: int = 5  # Number of chunks to retrieve
    similarity_threshold: float = 0.40  # <- جدید: چانک‌هایی با امتیاز کمتر از این، زباله در نظر گرفته می‌شوند
    # --- Generation Parameters ---
    temperature: float = 0.7

    # --- System Parameters ---
    debug_mode: bool = False
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        # Ensure base_data_dir is a Path object even if passed as string
        object.__setattr__(self, 'base_data_dir', Path(self.base_data_dir))
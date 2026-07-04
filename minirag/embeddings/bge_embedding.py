"""
BGE-M3 Embedding Implementation.

Uses SentenceTransformers to load the BAAI/bge-m3 model locally.
Educational note: BGE-m3 supports dense, sparse, and ColBERT retrieval,
but for V1 we only use the standard dense embeddings.
"""

from typing import List, Optional

import numpy as np

from minirag.embeddings.base import BaseEmbedding
from minirag.exceptions import EmbeddingError

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "sentence-transformers is required for local embeddings. "
        "Please install it via: pip install sentence-transformers"
    )


class BGEEmbedding(BaseEmbedding):
    """Embedding provider using BAAI/bge-m3."""

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        try:
            # Setting device='cpu' explicitly for maximum compatibility
            self.model = SentenceTransformer(model_name, device='cpu')
            self._dimension = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            raise EmbeddingError(f"Failed to load embedding model {model_name}: {e}") from e

    def embed(self, texts: List[str], doc_id: Optional[str] = None) -> np.ndarray:
        if not texts:
            return np.array([])
        
        try:
            # normalize_embeddings=True is crucial for cosine similarity search in FAISS
            embeddings = self.model.encode(texts, normalize_embeddings=True)
            return np.array(embeddings)
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embeddings: {e}") from e

    @property
    def dimension(self) -> int:
        return self._dimension
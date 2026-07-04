"""
Embeddings Factory.

Wires up the embedding model and optionally wraps it with the caching proxy.
"""

from pathlib import Path
from minirag.embeddings.base import BaseEmbedding
from minirag.embeddings.bge_embedding import BGEEmbedding
from minirag.embeddings.cache import CachedEmbedding
from minirag.config import Config


def get_embedder(config: Config) -> BaseEmbedding:
    """
    Factory function to create an embedding instance based on Config.
    """
    # In future versions, a dictionary mapping can be used here for multiple models
    if config.embedding_provider == "bge-m3":
        real_embedder = BGEEmbedding(model_name="BAAI/bge-m3")
    else:
        raise ValueError(f"Unknown embedding provider: {config.embedding_provider}")

    # Apply Cache Proxy if enabled in config
    if config.use_embedding_cache:
        return CachedEmbedding(
            inner_embedder=real_embedder,
            cache_dir=config.embeddings_dir,
            model_name=config.embedding_provider
        )

    return real_embedder
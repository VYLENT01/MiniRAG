"""
Semantic Retriever Implementation.

Converts queries to vectors, searches the FAISS index, 
and maps the resulting IDs back to rich SearchResult objects.
Includes Similarity Thresholding to prevent garbage context.
"""

import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from minirag.embeddings.base import BaseEmbedding
from minirag.retrievers.base import BaseRetriever
from minirag.vector_stores.base import BaseVectorStore
from minirag.models.chunk import Chunk, ChunkMetadata
from minirag.models.search import SearchResult
from minirag.exceptions import RetrieverError


class SemanticRetriever(BaseRetriever):
    """Retrieves documents using dense vector similarity with strict filtering."""

    def __init__(self, embedder: BaseEmbedding, vector_store: BaseVectorStore, chunks_dir: Path, registry=None, similarity_threshold: float = 0.40):
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunks_dir = chunks_dir
        self.registry = registry
        self.similarity_threshold = similarity_threshold  # <- جدید
        self._chunk_cache: Optional[Dict[str, Chunk]] = None

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        # 1. Embed the query
        query_vector = self.embedder.embed([query])
        if query_vector.size == 0:
            raise RetrieverError("Failed to generate embedding for the query.")

        # 2. Search the vector store (fetch more than top_k initially to allow for threshold filtering)
        fetch_limit = max(top_k * 2, 10) 
        raw_results = self.vector_store.search(query_vector[0], fetch_limit)
        
        # 3. APPLY THRESHOLD FILTER (Crucial V1.1 Feature)
        filtered_results = [
            (chunk_id, score) for chunk_id, score in raw_results 
            if score >= self.similarity_threshold
        ]
        
        # If after filtering nothing is left, return empty list (Prompt will handle the "I don't know" logic)
        if not filtered_results:
            return []

        # Slice to the actual requested top_k after filtering
        final_results = filtered_results[:top_k]
        
        # 4. Map results to SearchResult objects
        results = []
        chunk_map = self._load_chunk_map()

        for chunk_id, score in final_results:
            chunk_str_id = str(chunk_id)
            chunk = chunk_map.get(chunk_str_id)
            
            if not chunk:
                continue

            # Find document metadata
            doc_name = ""
            if self.registry:
                doc_meta = self.registry.get(chunk.document_uuid)
                if doc_meta:
                    doc_name = doc_meta.file_name

            result = SearchResult(
                text=chunk.text,
                similarity_score=score,
                chunk_id=chunk.uuid,
                document_name=doc_name,
                page=chunk.metadata.page_start,
                section=chunk.metadata.section,
                metadata={"heading": chunk.metadata.heading}
            )
            results.append(result)

        return results

    def _load_chunk_map(self) -> Dict[str, Chunk]:
        """Loads all chunks from disk into a dictionary for O(1) lookup by UUID."""
        if self._chunk_cache is not None:
            return self._chunk_cache

        self._chunk_cache = {}
        if not self.chunks_dir.exists():
            return self._chunk_cache

        try:
            for file_path in self.chunks_dir.glob("*.json"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    chunks_data = json.load(f)
                    for c_dict in chunks_data:
                        chunk = Chunk(
                            text=c_dict["text"],
                            metadata=ChunkMetadata(**c_dict["metadata"])
                        )
                        self._chunk_cache[str(chunk.uuid)] = chunk
        except Exception as e:
            raise RetrieverError(f"Failed to load chunk data from disk: {e}") from e

        return self._chunk_cache
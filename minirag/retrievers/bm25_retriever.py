"""
BM25 Retriever Implementation.

Performs keyword-based (Lexical) search using the BM25 algorithm.
Excellent for exact keyword matches (e.g., specific names, IDs, terms) 
where dense vector search might fail due to semantic over-generalization.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from minirag.retrievers.base import BaseRetriever
from minirag.models.chunk import Chunk, ChunkMetadata
from minirag.models.search import SearchResult
from minirag.exceptions import RetrieverError

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("rank_bm25 is required for lexical search. Please install it via: pip install rank_bm25")


class BM25Retriever(BaseRetriever):
    """Retrieves documents using traditional keyword matching (BM25Okapi)."""

    def __init__(self, chunks_dir: Path, registry=None):
        self.chunks_dir = chunks_dir
        self.registry = registry
        self._corpus: List[str] = []
        self._chunk_map: Dict[str, Chunk] = {}
        self._bm25: BM25Okapi = None
        self._tokenized_corpus: List[List[str]] = []

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple whitespace and punctuation tokenizer.
        Good enough for BM25 without adding heavy NLP dependencies.
        """
        # Remove punctuation, split by whitespace, lowercase
        return re.findall(r'\w+', text.lower())

    def _ensure_index_loaded(self) -> None:
        """Lazily builds the BM25 index from disk on the first query."""
        if self._bm25 is not None:
            return

        if not self.chunks_dir.exists():
            return

        try:
            for file_path in self.chunks_dir.glob("*.json"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    chunks_data = json.load(f)
                    for c_dict in chunks_data:
                        chunk = Chunk(text=c_dict["text"], metadata=ChunkMetadata(**c_dict["metadata"]))
                        self._chunk_map[str(chunk.uuid)] = chunk
                        self._corpus.append(chunk.text)

            # Tokenize corpus for BM25
            self._tokenized_corpus = [self._tokenize(doc) for doc in self._corpus]
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        except Exception as e:
            raise RetrieverError(f"Failed to build BM25 index from disk: {e}") from e

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        self._ensure_index_loaded()
        
        if not self._bm25 or not self._corpus:
            return []

        tokenized_query = self._tokenize(query)
        
        # Get raw BM25 scores for all documents
        raw_scores = self._bm25.get_scores(tokenized_query)
        
        # Pair scores with their document index
        scored_docs: List[Tuple[int, float]] = []
        for idx, score in enumerate(raw_scores):
            if score > 0: # Only consider documents that have at least one matching keyword
                scored_docs.append((idx, score))

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Take Top-K
        top_docs = scored_docs[:top_k]
        
        if not top_docs:
            return []

        # Normalize BM25 scores to a 0-1 range for compatibility with FAISS scores
        max_score = top_docs[0][1]
        min_score = top_docs[-1][1]
        score_range = max_score - min_score if max_score != min_score else 1.0

        results = []
        for idx, score in top_docs:
            doc_text = self._corpus[idx]
            # Find the corresponding chunk object
            chunk = next((c for c in self._chunk_map.values() if c.text == doc_text), None)
            
            if not chunk:
                continue

            # Normalize score between 0.0 and 1.0
            normalized_score = (score - min_score) / score_range

            # Get Document Name from Registry
            doc_name = ""
            if self.registry:
                doc_meta = self.registry.get(chunk.document_uuid)
                if doc_meta:
                    doc_name = doc_meta.file_name

            results.append(SearchResult(
                text=chunk.text,
                similarity_score=normalized_score,
                chunk_id=chunk.uuid,
                document_name=doc_name,
                page=chunk.metadata.page_start,
                section=chunk.metadata.section,
                metadata={"heading": chunk.metadata.heading}
            ))

        return results
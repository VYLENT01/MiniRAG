"""
Hybrid Retriever Implementation.

Combines Dense (FAISS) and Sparse (BM25) retrieval using 
Reciprocal Rank Fusion (RRF). This provides the best of both worlds:
- Dense: Understands semantic meaning and synonyms.
- Sparse: Excels at exact keyword matches.
"""

from typing import List
from minirag.retrievers.base import BaseRetriever
from minirag.models.search import SearchResult


class HybridRetriever(BaseRetriever):
    """Retrieves documents by fusing results from Semantic and BM25 retrievers."""

    def __init__(self, semantic_retriever: BaseRetriever, bm25_retriever: BaseRetriever, k: int = 60):
        """
        Args:
            semantic_retriever: An instance of SemanticRetriever.
            bm25_retriever: An instance of BM25Retriever.
            k: The RRF constant (default 60 is standard in literature).
        """
        self.semantic = semantic_retriever
        self.bm25 = bm25_retriever
        self.k = k

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        # We fetch more results from each retriever to have a rich candidate pool for fusion
        fetch_limit = max(top_k * 3, 15)

        # 1. Get candidates from both retrievers
        sem_results = self.semantic.retrieve(query, top_k=fetch_limit)
        bm25_results = self.bm25.retrieve(query, top_k=fetch_limit)

        # 2. Calculate RRF scores
        rrf_scores = {}
        result_map = {} # Maps chunk_id -> SearchResult object to preserve metadata

        # Process Semantic results
        for rank, res in enumerate(sem_results, start=1):
            cid = str(res.chunk_id)
            result_map[cid] = res
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.k + rank))

        # Process BM25 results
        for rank, res in enumerate(bm25_results, start=1):
            cid = str(res.chunk_id)
            result_map[cid] = res
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.k + rank))

        # 3. Sort by the fused RRF score (Descending)
        sorted_cids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # 4. Slice to the final requested top_k
        final_results = []
        for cid, rrf_score in sorted_cids[:top_k]:
            original_res = result_map[cid]
            
            # Return a new SearchResult object, replacing the old similarity score with the new RRF score
            # This ensures the downstream components (like Confidence calculation) see the fused score
            final_results.append(SearchResult(
                text=original_res.text,
                similarity_score=rrf_score, 
                chunk_id=original_res.chunk_id,
                document_name=original_res.document_name,
                page=original_res.page,
                section=original_res.section,
                metadata=original_res.metadata
            ))

        return final_results
"""
Query Engine.
"""

from typing import List
from minirag.embeddings.base import BaseEmbedding
from minirag.retrievers.base import BaseRetriever
from minirag.prompts.base import BasePromptBuilder
from minirag.llms.base import BaseLLM
from minirag.models.search import SearchResult
from minirag.models.answer import Answer, Citation
from minirag.exceptions import PipelineError


class QueryEngine:
    """Handles the end-to-end question answering process."""

    def __init__(self, embedder: BaseEmbedding, retriever: BaseRetriever, prompt_builder: BasePromptBuilder, llm: BaseLLM, default_top_k: int = 5):
        self.embedder = embedder
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.llm = llm
        self.default_top_k = default_top_k

    def ask(self, question: str, top_k: int = None) -> Answer:
        if not question.strip():
            raise PipelineError("Question cannot be empty.")

        k = top_k if top_k is not None else self.default_top_k

        try:
            search_results: List[SearchResult] = self.retriever.retrieve(question, top_k=k)

            # محاسبه Confidence Score بر اساس امتیازات بازگردانده شده از FAISS
            confidence_score = 0.0
            confidence_level = "LOW"
            is_faithful = len(search_results) > 0
            
            if is_faithful:
                confidence_score = sum(r.similarity_score for r in search_results) / len(search_results)
                if confidence_score >= 0.65:
                    confidence_level = "HIGH"
                elif confidence_score >= 0.45:
                    confidence_level = "MEDIUM"
                else:
                    confidence_level = "LOW"

            prompt = self.prompt_builder.build(search_results, question)
            llm_response_text = self.llm.generate(prompt)

            citations = [
                Citation(
                    exact_snippet=r.text[:150] + "...",
                    document_name=r.document_name,
                    page=r.page,
                    similarity_score=r.similarity_score,
                    chunk_id=r.chunk_id
                )
                for r in search_results
            ]

            return Answer(
                text=llm_response_text, 
                is_faithful=is_faithful, 
                citations=citations,
                confidence_score=confidence_score,
                confidence_level=confidence_level
            )

        except Exception as e:
            if isinstance(e, PipelineError):
                raise
            raise PipelineError(f"Failed to process query: {e}") from e
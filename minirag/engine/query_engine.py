"""
Query Engine.
Orchestrates Retrieval -> Raw LLM Extraction -> Deterministic Formatting
"""

from typing import List
from minirag.embeddings.base import BaseEmbedding
from minirag.retrievers.base import BaseRetriever
from minirag.prompts.base import BasePromptBuilder
from minirag.llms.base import BaseLLM
from minirag.models.search import SearchResult
from minirag.models.answer import Answer
from minirag.engine.formatter import AnswerFormatter
from minirag.exceptions import PipelineError


class QueryEngine:
    """Handles the end-to-end question answering process."""

    def __init__(self, embedder: BaseEmbedding, retriever: BaseRetriever, prompt_builder: BasePromptBuilder, llm: BaseLLM, default_top_k: int = 5):
        self.embedder = embedder
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.llm = llm
        self.formatter = AnswerFormatter() # تزریق فرمت‌کننده قطعی
        self.default_top_k = default_top_k

    def ask(self, question: str, top_k: int = None, similarity_threshold: float = 0.4) -> Answer:
        if not question.strip():
            raise PipelineError("Question cannot be empty.")

        k = top_k if top_k is not None else self.default_top_k

        # LAYER 1: RETRIEVAL HARDENING
        search_results: List[SearchResult] = self.retriever.retrieve(question, top_k=k)

        # Quality Gate: If nothing passes threshold, skip LLM entirely
        if not search_results:
            return self.formatter.format("", [], similarity_threshold)

        try:
            # LAYER 2: STRUCTURED LLM OUTPUT
            prompt = self.prompt_builder.build(search_results, question)
            raw_llm_json = self.llm.generate(prompt)
            
            # LAYER 3: DETERMINISTIC FORMATTING
            # The LLM is no longer responsible for citations or metadata!
            final_answer = self.formatter.format(raw_llm_json, search_results, similarity_threshold)

            return final_answer

        except Exception as e:
            if isinstance(e, PipelineError):
                raise
            raise PipelineError(f"Failed to process query: {e}") from e
"""
Base Prompt Builder Interface.

Defines the contract for formatting context and queries into LLM prompts.
"""

from abc import ABC, abstractmethod
from typing import List

from minirag.models.search import SearchResult


class BasePromptBuilder(ABC):
    """Abstract base class for prompt builders."""

    @abstractmethod
    def build(self, context: List[SearchResult], question: str) -> str:
        """
        Construct the final prompt string.

        Args:
            context: The retrieved chunks relevant to the question.
            question: The user's raw question.

        Returns:
            A fully formatted prompt string ready for the LLM.
        """
        pass
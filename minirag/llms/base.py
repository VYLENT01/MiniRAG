"""
Base LLM Interface.

Defines the contract for language model providers.
"""

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a text response based on the provided prompt.

        Args:
            prompt: The fully formatted prompt string.

        Returns:
            The generated text response.

        Raises:
            LLMError: If generation fails, times out, or API returns an error.
        """
        pass
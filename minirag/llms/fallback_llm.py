"""
Fallback LLM Implementation.

Implements a resilience pattern by wrapping multiple LLM providers.
If the primary provider fails, it seamlessly falls back to the next one.
"""

from typing import List

from minirag.llms.base import BaseLLM
from minirag.exceptions import LLMError


class FallbackLLM(BaseLLM):
    """Wraps multiple LLMs and attempts them in order until one succeeds."""

    def __init__(self, providers: List[BaseLLM]):
        if not providers:
            raise ValueError("FallbackLLM requires at least one LLM provider.")
        self.providers = providers

    def generate(self, prompt: str) -> str:
        errors = []
        
        for i, provider in enumerate(self.providers):
            provider_name = provider.__class__.__name__
            try:
                # Attempt to generate with the current provider
                return provider.generate(prompt)
            except LLMError as e:
                errors.append(f"{provider_name}: {str(e)}")
                # If this is the last provider, don't continue
                if i == len(self.providers) - 1:
                    break
                # Otherwise, silently swallow the error and try the next
                continue
                
        # If we exit the loop, all providers failed
        error_details = "\n- ".join(errors)
        raise LLMError(
            f"All LLM providers failed sequentially:\n- {error_details}"
        )
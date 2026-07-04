"""
LLMs Factory.

Wires up LLM providers and wraps them in the fallback mechanism if configured.
"""

from typing import List, Optional

from minirag.llms.base import BaseLLM
from minirag.llms.ollama_llm import OllamaLLM
from minirag.llms.gemini_llm import GeminiLLM
from minirag.llms.fallback_llm import FallbackLLM
from minirag.config import Config
from minirag.exceptions import ConfigurationError


def get_llm(config: Config) -> BaseLLM:
    """
    Factory function to create an LLM instance based on Config.
    Configures the fallback chain automatically.
    """
    providers: List[BaseLLM] = []

    # 1. Initialize Primary LLM
    if config.primary_llm_provider == "ollama":
        providers.append(OllamaLLM(model=config.ollama_model, timeout=config.llm_timeout))
        
    elif config.primary_llm_provider == "gemini":
        if not config.gemini_api_key or config.gemini_api_key == "YOUR_API_KEY_HERE":
            raise ConfigurationError("Gemini API key is required when it is set as the primary provider.")
        providers.append(
            GeminiLLM(
                model=config.gemini_model, 
                api_key=config.gemini_api_key, 
                timeout=config.llm_timeout
            )
        )
    else:
        raise ConfigurationError(f"Unsupported primary LLM provider: {config.primary_llm_provider}")

    # 2. Initialize Secondary LLM (if configured and different from primary)
    if config.secondary_llm_provider and config.secondary_llm_provider != config.primary_llm_provider:
        if config.secondary_llm_provider == "gemini":
            if not config.gemini_api_key or config.gemini_api_key == "YOUR_API_KEY_HERE":
                raise ConfigurationError("Gemini API key is required for fallback.")
            providers.append(
                GeminiLLM(
                    model=config.gemini_model, 
                    api_key=config.gemini_api_key, 
                    timeout=config.llm_timeout
                )
            )
        elif config.secondary_llm_provider == "ollama":
             providers.append(OllamaLLM(model=config.ollama_model, timeout=config.llm_timeout))
        else:
            raise ConfigurationError(f"Unsupported secondary LLM provider: {config.secondary_llm_provider}")

    # 3. Return Fallback wrapper if we have multiple providers, else return the single one
    if len(providers) > 1:
        return FallbackLLM(providers)
    
    return providers[0]
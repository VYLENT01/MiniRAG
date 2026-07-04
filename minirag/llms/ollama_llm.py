"""
Ollama LLM Implementation.

Connects to a locally running Ollama instance via its REST API.
Educational note: Ollama exposes an OpenAI-compatible API, but its native 
API is even simpler. We use the native /api/generate endpoint.
"""

import requests

from minirag.llms.base import BaseLLM
from minirag.exceptions import LLMError


class OllamaLLM(BaseLLM):
    """LLM provider for locally hosted Ollama models."""

    def __init__(self, model: str = "llama3", timeout: int = 30):
        self.model = model
        self.timeout = timeout
        # Ollama's default local REST API endpoint
        self.base_url = "http://localhost:11434/api/generate"

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False  # We want the full response at once in V1
        }

        try:
            response = requests.post(
                self.base_url, 
                json=payload, 
                timeout=self.timeout
            )
            response.raise_for_status()  # Raises HTTPError for bad status codes
            
            data = response.json()
            if "response" not in data:
                raise LLMError(f"Unexpected response format from Ollama: {data}")
                
            return data["response"].strip()

        except requests.exceptions.Timeout:
            raise LLMError(f"Ollama request timed out after {self.timeout} seconds.")
        except requests.exceptions.ConnectionError:
            raise LLMError(
                "Failed to connect to Ollama. Is it running? "
                "Start it with the 'ollama serve' command."
            )
        except requests.exceptions.HTTPError as e:
            raise LLMError(f"Ollama returned an HTTP error: {e.response.text}") from e
        except Exception as e:
            raise LLMError(f"An unexpected error occurred with Ollama: {e}") from e
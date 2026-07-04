"""
Gemini LLM Implementation.

Connects to Google's Gemini API using the standard REST endpoint.
Educational note: This avoids the need for the official `google-generativeai` 
SDK, demonstrating how LLM APIs work at the HTTP level.
"""

import requests

from minirag.llms.base import BaseLLM
from minirag.exceptions import LLMError


class GeminiLLM(BaseLLM):
    """LLM provider for Google Gemini API."""

    def __init__(self, model: str = "gemini-1.5-flash", api_key: str = None, timeout: int = 30):
        if not api_key:
            raise ValueError("Gemini API key is required.")
            
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    def generate(self, prompt: str) -> str:
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            # Optional: Enforce deterministic output for V1
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 2048
            }
        }

        try:
            response = requests.post(
                self.base_url, 
                json=payload, 
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Navigate the Gemini response structure
            candidates = data.get("candidates", [])
            if not candidates:
                raise LLMError(f"No candidates returned from Gemini. Response: {data}")
                
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts or "text" not in parts[0]:
                raise LLMError(f"Malformed response structure from Gemini: {data}")
                
            return parts[0]["text"].strip()

        except requests.exceptions.Timeout:
            raise LLMError(f"Gemini request timed out after {self.timeout} seconds.")
        except requests.exceptions.HTTPError as e:
            error_msg = e.response.text if e.response is not None else str(e)
            raise LLMError(f"Gemini returned an HTTP error: {error_msg}") from e
        except Exception as e:
            raise LLMError(f"An unexpected error occurred with Gemini: {e}") from e
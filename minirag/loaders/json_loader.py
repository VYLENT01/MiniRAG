"""
JSON Loader Implementation.

Recursively extracts all string values from a JSON file.
This prevents JSON structural characters (like braces) from polluting the text,
while still capturing all meaningful data.
"""

import json
from pathlib import Path
from typing import Any

from minirag.exceptions import LoaderError
from minirag.loaders.base import BaseLoader


class JSONLoader(BaseLoader):
    """Loads text by extracting string values from JSON structures."""

    def load(self, file_path: Path) -> str:
        if not file_path.exists():
            raise LoaderError(f"File not found: {file_path}")

        try:
            raw_json = file_path.read_text(encoding="utf-8")
            data = json.loads(raw_json)
            
            extracted_strings = []
            self._extract_strings(data, extracted_strings)
            
            if not extracted_strings:
                raise LoaderError(f"No string values found in JSON: {file_path}")
                
            return "\n".join(extracted_strings)
        except json.JSONDecodeError as e:
            raise LoaderError(f"Invalid JSON format in {file_path}: {e}") from e
        except Exception as e:
            raise LoaderError(f"Failed to process JSON {file_path}: {e}") from e

    def _extract_strings(self, data: Any, out_list: list) -> None:
        """Helper to recursively find strings in JSON."""
        if isinstance(data, str):
            out_list.append(data)
        elif isinstance(data, dict):
            for value in data.values():
                self._extract_strings(value, out_list)
        elif isinstance(data, list):
            for item in data:
                self._extract_strings(item, out_list)
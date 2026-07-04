"""
Document Registry.

Manages document metadata and ensures data integrity (e.g., SHA256 deduplication).
"""

import hashlib
import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime

import uuid

from minirag.models.document import DocumentMetadata
from minirag.config import Config
from minirag.exceptions import PipelineError


class DocumentRegistry:
    """Handles persistence and retrieval of document metadata."""

    def __init__(self, config: Config):
        self.metadata_dir = config.metadata_dir
        self.registry_path = self.metadata_dir / "registry.json"
        self._ensure_dirs()
        self._data: List[dict] = self._load()

    def _ensure_dirs(self) -> None:
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def _load(self) -> List[dict]:
        if not self.registry_path.exists():
            return []
        try:
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            raise PipelineError("Registry file is corrupted. Cannot read metadata.")

    def _save(self) -> None:
        try:
            with open(self.registry_path, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, indent=4, default=str)
        except Exception as e:
            raise PipelineError(f"Failed to save registry: {e}")

    @staticmethod
    def calculate_sha256(file_path: Path) -> str:
        """Calculates the SHA256 hash of a file efficiently."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def get_by_sha256(self, sha256: str) -> Optional[DocumentMetadata]:
        """Checks if a document with this hash is already indexed."""
        for item in self._data:
            if item["sha256"] == sha256:
                return DocumentMetadata(**item)
        return None

    def add(self, doc_meta: DocumentMetadata) -> None:
        """Adds a new document metadata to the registry and saves to disk."""
        # Convert dataclass to dict for JSON serialization
        self._data.append(doc_meta.__dict__)
        self._save()

    def get(self, doc_id: uuid.UUID) -> Optional[DocumentMetadata]:
        """Retrieves a specific document by UUID."""
        for item in self._data:
            if item["uuid"] == str(doc_id):
                return DocumentMetadata(**item)
        return None

    def list_all(self) -> List[DocumentMetadata]:
        """Returns all indexed documents."""
        return [DocumentMetadata(**item) for item in self._data]

    def remove(self, doc_id: uuid.UUID) -> bool:
        """Removes a document from the registry. Returns True if successful."""
        initial_len = len(self._data)
        self._data = [item for item in self._data if item["uuid"] != str(doc_id)]
        
        if len(self._data) < initial_len:
            self._save()
            return True
        return False
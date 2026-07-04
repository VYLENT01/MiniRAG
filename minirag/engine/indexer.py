"""
Document Indexer.

Orchestrates the ingestion pipeline:
Load -> Clean -> Chunk -> Save Chunks -> Embed -> Vector Store -> Update Registry
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from minirag.config import Config
from minirag.engine.registry import DocumentRegistry
from minirag.exceptions import DuplicateDocumentError, PipelineError
from minirag.loaders import get_loader
from minirag.cleaners import TextCleaner
from minirag.chunkers import FixedSizeChunker
from minirag.embeddings.base import BaseEmbedding
from minirag.vector_stores.base import BaseVectorStore
from minirag.models.document import DocumentMetadata
from minirag.models.chunk import Chunk


class Indexer:
    """Handles the end-to-end document ingestion process."""

    def __init__(self, config: Config, registry: DocumentRegistry, embedder: BaseEmbedding, vector_store: BaseVectorStore):
        self.config = config
        self.registry = registry
        self.embedder = embedder
        self.vector_store = vector_store
        
        self.cleaner = TextCleaner()
        self.chunker = FixedSizeChunker(
            chunk_size=config.chunk_size, 
            chunk_overlap=config.chunk_overlap
        )

    def index_document(self, file_path: Path) -> DocumentMetadata:
        """
        Executes the full indexing pipeline for a single document.
        
        Returns:
            The DocumentMetadata of the newly indexed document.
        """
        file_path = Path(file_path).resolve()
        if not file_path.exists():
            raise PipelineError(f"File not found: {file_path}")

        # 1. Deduplication Check
        sha256 = DocumentRegistry.calculate_sha256(file_path)
        existing = self.registry.get_by_sha256(sha256)
        if existing:
            raise DuplicateDocumentError(
                f"Document is already indexed with ID: {existing.uuid} "
                f"(Name: {existing.file_name})"
            )

        # 2. Prepare Metadata
        doc_id = uuid.uuid4()
        doc_meta = DocumentMetadata(
            uuid=doc_id,
            file_name=file_path.name,
            absolute_path=file_path,
            sha256=sha256,
            file_type=file_path.suffix.lower().replace(".", ""),
            file_size=file_path.stat().st_size,
            created_date=datetime.fromtimestamp(file_path.stat().st_ctime),
            indexed_date=datetime.now(),
            embedding_model=self.config.embedding_provider,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )

        try:
            # 3. Load
            loader = get_loader(file_path)
            raw_text = loader.load(file_path)

            # 4. Clean
            cleaned_text = self.cleaner.clean(raw_text)

            # 5. Chunk
            chunks = self.chunker.chunk(cleaned_text, doc_meta)
            doc_meta = DocumentMetadata(
                **{**doc_meta.__dict__, "chunk_count": len(chunks)}
            )

            # 6. Save Chunks to Disk
            self._save_chunks(doc_id, chunks)

            # 7. Embed
            chunk_texts = [c.text for c in chunks]
            vectors = self.embedder.embed(chunk_texts, doc_id=doc_id)

            # 8. Add to Vector Store
            chunk_ids = [c.uuid for c in chunks]
            self.vector_store.add(chunk_ids, vectors)
            self.vector_store.save()

            # 9. Commit to Registry (Acts as transaction finalization)
            self.registry.add(doc_meta)

            return doc_meta

        except Exception as e:
            # In a production system, we would rollback (delete chunks, remove vectors)
            # For V1, we raise the error and leave cleanup to the `rebuild` command.
            if not isinstance(e, (DuplicateDocumentError, PipelineError)):
                raise PipelineError(f"Indexing failed for {file_path.name}: {e}") from e
            raise

    def _save_chunks(self, doc_id: uuid.UUID, chunks: list) -> None:
        """Saves chunk data and metadata to a JSON file."""
        from dataclasses import asdict
        
        chunks_dir = self.config.chunks_dir
        chunks_dir.mkdir(parents=True, exist_ok=True)
        
        chunk_file = chunks_dir / f"{doc_id}.json"
        
        # Use asdict for clean, recursive dataclass serialization
        serializable_chunks = [asdict(c) for c in chunks]
        
        with open(chunk_file, 'w', encoding='utf-8') as f:
            # default=str safely handles UUIDs and any unexpected types
            json.dump(serializable_chunks, f, indent=2, ensure_ascii=False, default=str)
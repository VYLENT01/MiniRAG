"""
Fixed Size Chunker Implementation.

Splits text into chunks of approximately `chunk_size` characters 
with an `overlap` of `chunk_overlap` characters between them.
"""

import hashlib
import uuid
from typing import List

from minirag.chunkers.base import BaseChunker
from minirag.exceptions import ChunkerError
from minirag.models.chunk import Chunk, ChunkMetadata
from minirag.models.document import DocumentMetadata


class FixedSizeChunker(BaseChunker):
    """Chunks text based on character count with a sliding window overlap."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        if chunk_overlap >= chunk_size:
            raise ChunkerError("chunk_overlap must be smaller than chunk_size.")
            
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, doc_metadata: DocumentMetadata) -> List[Chunk]:
        if not text:
            return []

        chunks = []
        start = 0
        index = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Generate unique ID and hash for the chunk
            chunk_uuid = uuid.uuid4()
            chunk_hash = hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()

            chunk_meta = ChunkMetadata(
                uuid=chunk_uuid,
                document_uuid=doc_metadata.uuid,
                chunk_index=index,
                page_start=None,  # Page mapping is complex for V1, left null
                page_end=None,
                section=None,
                heading=None,
                char_count=len(chunk_text),
                token_count=None, # Token counting deferred to avoid heavy deps in chunker
                content_type="text",
                sha256=chunk_hash
            )

            chunks.append(Chunk(text=chunk_text, metadata=chunk_meta))
            
            # Move window forward, accounting for overlap
            start += self.chunk_size - self.chunk_overlap
            index += 1

        return chunks
"""
MiniRAG Facade.

The single public entry point for the library. Hides all internal complexity.
Vylent and end-users only interact with this class.
"""

import json
import uuid
from pathlib import Path
from typing import List, Optional

from minirag.config import Config
from minirag.exceptions import MiniRAGError, DocumentNotFoundError, DuplicateDocumentError
from minirag.engine import DocumentRegistry, Indexer
from minirag.engine.query_engine import QueryEngine
from minirag.embeddings import get_embedder
from minirag.vector_stores.faiss_store import FAISSStore
from minirag.retrievers.semantic_retriever import SemanticRetriever
from minirag.prompts.qa_citation_prompt import QACitationPromptBuilder
from minirag.llms import get_llm
from minirag.models.document import DocumentMetadata
from minirag.models.answer import Answer
from minirag.models.search import SearchResult


class MiniRAG:
    """
    A lightweight, educational RAG engine.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._initialize_directories()
        _initialize_components(self)

    def _initialize_directories(self) -> None:
        """Ensure all required data directories exist."""
        self.config.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.config.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.config.vector_db_dir.mkdir(parents=True, exist_ok=True)
        self.config.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.config.logs_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_components(self) -> None:
        """Wire up all internal engine components using Dependency Injection."""
        # Core data management
        self.registry = DocumentRegistry(self.config)
        
        # Embedding & Vector Store
        self.embedder = get_embedder(self.config)
        self.vector_store = FAISSStore(
            index_dir=self.config.vector_db_dir, 
            dimension=self.embedder.dimension
        )
        self.vector_store.load()

        # Pipeline engines
        self.indexer = Indexer(self.config, self.registry, self.embedder, self.vector_store)
        
        # --- Dynamic Retriever Wiring (V1.2) ---
        if self.config.retriever_provider == "bm25":
            from minirag.retrievers.bm25_retriever import BM25Retriever
            final_retriever = BM25Retriever(self.config.chunks_dir, self.registry)
            
        elif self.config.retriever_provider == "hybrid":
            from minirag.retrievers.hybrid_retriever import HybridRetriever
            from minirag.retrievers.bm25_retriever import BM25Retriever
            sem_ret = SemanticRetriever(self.embedder, self.vector_store, self.config.chunks_dir, self.registry, self.config.similarity_threshold)
            bm25_ret = BM25Retriever(self.config.chunks_dir, self.registry)
            final_retriever = HybridRetriever(semantic_retriever=sem_ret, bm25_retriever=bm25_ret)
            
        else: # Default to "semantic"
            final_retriever = SemanticRetriever(self.embedder, self.vector_store, self.config.chunks_dir, self.registry, self.config.similarity_threshold)

        self.query_engine = QueryEngine(
            embedder=self.embedder,
            retriever=final_retriever,
            prompt_builder=QACitationPromptBuilder(),
            llm=get_llm(self.config),
            default_top_k=self.config.top_k
        )

    def add_document(self, path: str) -> DocumentMetadata:
        """Index a document from an absolute file path."""
        file_path = Path(path).resolve()
        try:
            return self.indexer.index_document(file_path)
        except DuplicateDocumentError as e:
            raise MiniRAGError(str(e)) from e
    
    def retrieve(self, question: str, top_k: int = None) -> List[SearchResult]:
        """
        Retrieves the most relevant chunks for a given question without generating an answer.
        This is the primary integration point for external agents (like Vylent).
        
        Args:
            question: The user's query.
            top_k: Number of chunks to retrieve. Defaults to config.top_k.

        Returns:
            A list of SearchResult objects containing text, metadata, and similarity scores.
        """
        k = top_k if top_k is not None else self.config.top_k
        
        # Directly access the internal retriever (either Semantic, BM25, or Hybrid)
        return self.query_engine.retriever.retrieve(question, top_k=k)

    def ask(self, question: str) -> Answer:
        """Ask a question and get an answer with citations."""
        return self.query_engine.ask(question)

    def delete_document(self, document_id: str) -> bool:
        """Delete a document by its UUID."""
        try:
            doc_uuid = uuid.UUID(document_id)
        except ValueError:
            raise MiniRAGError(f"Invalid UUID format: {document_id}")

        if not self.registry.remove(doc_uuid):
            raise DocumentNotFoundError(f"Document with ID {document_id} not found.")

        # Remove the corresponding chunks file
        chunk_file = self.config.chunks_dir / f"{doc_uuid}.json"
        if chunk_file.exists():
            chunk_file.unlink()
            
        # Optionally remove embedding cache
        if self.config.use_embedding_cache:
            cache_file = self.config.embeddings_dir / self.config.embedding_provider / f"{doc_uuid}.npy"
            if cache_file.exists():
                cache_file.unlink()

        return True

    def list_documents(self) -> List[DocumentMetadata]:
        """List all indexed documents."""
        return self.registry.list_all()

    def rebuild(self) -> int:
        """
        Rebuild the entire vector store from existing chunk files.
        Useful after deletions, or if changing embedding models.
        Returns the number of documents re-indexed.
        """
        from minirag.chunkers.fixed_size_chunker import FixedSizeChunker
        from minirag.models.chunk import Chunk, ChunkMetadata

        # 1. Wipe current FAISS index and ID map
        self.vector_store._init_index() 
        self.vector_store.save()

        docs = self.list_documents()
        if not docs:
            return 0

        chunker = FixedSizeChunker(self.config.chunk_size, self.config.chunk_overlap)
        re_indexed_count = 0

        for doc in docs:
            chunk_file = self.config.chunks_dir / f"{doc.uuid}.json"
            if not chunk_file.exists():
                continue

            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunks_data = json.load(f)
                
                # Reconstruct Chunk objects
                chunks = [
                    Chunk(text=c["text"], metadata=ChunkMetadata(**c["metadata"])) 
                    for c in chunks_data
                ]

                # Re-embed and add to fresh FAISS index
                texts = [c.text for c in chunks]
                vectors = self.embedder.embed(texts, doc_id=doc.uuid)
                ids = [c.uuid for c in chunks]
                
                self.vector_store.add(ids, vectors)
                re_indexed_count += 1
            except Exception as e:
                print(f"Warning: Failed to re-index {doc.file_name}: {e}")

        # Save the final rebuilt index
        self.vector_store.save()
        return re_indexed_count
"""
Memory manager for semantic search over conversation history.

Uses ChromaDB for vector storage and Gemini for embeddings.
"""
from __future__ import annotations

import json
import hashlib
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
import google.generativeai as genai

from token_counter import get_counter

logger = logging.getLogger(__name__)

# Constants
CHUNK_SIZE = 400              # tokens per chunk
CHUNK_OVERLAP = 80            # overlap between chunks
EMBEDDING_MODEL = "models/text-embedding-004"
COLLECTION_NAME = "agent_memory"

# Paths
AGENT_DIR = Path.home() / "Code" / "Agent"
MEMORY_DIR = AGENT_DIR / "memory"
CHROMA_DIR = MEMORY_DIR / "chroma"
SECRETS_FILE = AGENT_DIR / "config" / "secrets.json"
CONVERSATION_LOG = MEMORY_DIR / "conversation_log.jsonl"
HASH_CACHE_FILE = MEMORY_DIR / ".content_hashes.json"


@dataclass
class SearchResult:
    """Result from a memory search."""
    text: str
    score: float
    source: str           # "conversation" | "memory" | "context"
    timestamp: str        # ISO format
    metadata: dict


def load_secrets() -> dict:
    """Load API keys from secrets file."""
    if not SECRETS_FILE.exists():
        logger.warning(f"Secrets file not found: {SECRETS_FILE}")
        return {}
    with open(SECRETS_FILE) as f:
        return json.load(f)


def compute_hash(text: str) -> str:
    """Compute SHA256 hash of text for change detection."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class GeminiEmbedder:
    """Generate embeddings using Gemini API."""

    def __init__(self, api_key: str):
        """Initialize with Gemini API key."""
        genai.configure(api_key=api_key)
        self._model = EMBEDDING_MODEL

    def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        result = genai.embed_content(
            model=self._model,
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a search query."""
        result = genai.embed_content(
            model=self._model,
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed(t) for t in texts]


class MemoryManager:
    """Manages vector storage and retrieval of conversation memory."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the memory manager.

        Args:
            api_key: Gemini API key. If not provided, loads from secrets.json.
        """
        # Load API key
        if api_key is None:
            secrets = load_secrets()
            api_key = secrets.get("gemini_api_key")

        if not api_key:
            raise ValueError("Gemini API key required. Set in config/secrets.json")

        # Initialize embedder
        self._embedder = GeminiEmbedder(api_key)

        # Initialize ChromaDB
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        # Load hash cache
        self._hash_cache = self._load_hash_cache()

        logger.info(f"MemoryManager initialized. Collection has {self._collection.count()} documents.")

    def _load_hash_cache(self) -> dict:
        """Load content hash cache from disk."""
        if HASH_CACHE_FILE.exists():
            with open(HASH_CACHE_FILE) as f:
                return json.load(f)
        return {}

    def _save_hash_cache(self):
        """Save content hash cache to disk."""
        with open(HASH_CACHE_FILE, "w") as f:
            json.dump(self._hash_cache, f)

    def _content_changed(self, key: str, content: str) -> bool:
        """Check if content has changed since last index."""
        new_hash = compute_hash(content)
        old_hash = self._hash_cache.get(key)
        if old_hash == new_hash:
            return False
        self._hash_cache[key] = new_hash
        return True

    def chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE,
                   overlap: int = CHUNK_OVERLAP) -> list[str]:
        """
        Split text into overlapping chunks by token count.

        Args:
            text: Text to chunk.
            chunk_size: Target tokens per chunk.
            overlap: Overlap tokens between chunks.

        Returns:
            List of text chunks.
        """
        if not text.strip():
            return []

        counter = get_counter()
        tokens = counter._encoding.encode(text)

        if len(tokens) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = counter._encoding.decode(chunk_tokens)
            chunks.append(chunk_text)

            # Move start forward, accounting for overlap
            start = end - overlap if end < len(tokens) else end

        return chunks

    def index_conversation(self, entry: dict) -> int:
        """
        Index a conversation entry into the vector store.

        Args:
            entry: Dict with timestamp, sender, user_message, agent_response.

        Returns:
            Number of chunks indexed.
        """
        timestamp = entry.get("timestamp", "")
        sender = entry.get("sender", "unknown")
        user_msg = entry.get("user_message", "")
        agent_resp = entry.get("agent_response", "")

        # Create a single document combining both sides
        text = f"[{timestamp}] User ({sender}): {user_msg}\nAgent: {agent_resp}"

        # Check if already indexed
        doc_id = f"conv_{compute_hash(text)}"
        if doc_id in self._hash_cache:
            return 0

        # Chunk and index
        chunks = self.chunk_text(text)
        if not chunks:
            return 0

        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk{i}"
            ids.append(chunk_id)
            documents.append(chunk)
            embeddings.append(self._embedder.embed(chunk))
            metadatas.append({
                "source": "conversation",
                "timestamp": timestamp,
                "sender": sender,
                "chunk_index": i,
                "total_chunks": len(chunks)
            })

        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        self._hash_cache[doc_id] = True
        self._save_hash_cache()

        logger.debug(f"Indexed conversation entry: {len(chunks)} chunks")
        return len(chunks)

    def index_file(self, path: str | Path, source_type: str = "memory") -> int:
        """
        Index a file into the vector store.

        Args:
            path: Path to file to index.
            source_type: Type label ("memory", "context", etc.)

        Returns:
            Number of chunks indexed.
        """
        path = Path(path)
        if not path.exists():
            logger.warning(f"File not found: {path}")
            return 0

        content = path.read_text()
        if not content.strip():
            return 0

        # Check if content changed
        cache_key = f"file_{path.name}"
        if not self._content_changed(cache_key, content):
            logger.debug(f"File unchanged, skipping: {path.name}")
            return 0

        # Remove old entries for this file
        try:
            existing = self._collection.get(
                where={"source": source_type, "filename": path.name}
            )
            if existing["ids"]:
                self._collection.delete(ids=existing["ids"])
        except Exception:
            pass  # Collection might not have these entries

        # Chunk and index
        chunks = self.chunk_text(content)
        if not chunks:
            return 0

        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{source_type}_{path.name}_{i}"
            ids.append(chunk_id)
            documents.append(chunk)
            embeddings.append(self._embedder.embed(chunk))
            metadatas.append({
                "source": source_type,
                "filename": path.name,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "timestamp": ""
            })

        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        self._save_hash_cache()
        logger.info(f"Indexed file {path.name}: {len(chunks)} chunks")
        return len(chunks)

    def backfill_conversations(self) -> int:
        """
        Index all existing conversations from conversation_log.jsonl.

        Returns:
            Total number of chunks indexed.
        """
        if not CONVERSATION_LOG.exists():
            logger.info("No conversation log found to backfill")
            return 0

        total_chunks = 0
        entries = []

        with open(CONVERSATION_LOG) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        logger.info(f"Backfilling {len(entries)} conversation entries...")

        for entry in entries:
            total_chunks += self.index_conversation(entry)

        logger.info(f"Backfill complete: {total_chunks} chunks indexed")
        return total_chunks

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """
        Search for relevant memory chunks.

        Args:
            query: Search query text.
            top_k: Number of results to return.

        Returns:
            List of SearchResult objects.
        """
        if self._collection.count() == 0:
            return []

        query_embedding = self._embedder.embed_query(query)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        search_results = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0

                # Convert distance to similarity score (cosine distance -> similarity)
                score = 1 - distance

                search_results.append(SearchResult(
                    text=doc,
                    score=score,
                    source=metadata.get("source", "unknown"),
                    timestamp=metadata.get("timestamp", ""),
                    metadata=metadata
                ))

        return search_results

    def get_stats(self) -> dict:
        """Get memory store statistics."""
        return {
            "total_documents": self._collection.count(),
            "hash_cache_size": len(self._hash_cache),
            "collection_name": COLLECTION_NAME,
            "chroma_path": str(CHROMA_DIR)
        }


# Module-level singleton
_manager: Optional[MemoryManager] = None


def get_manager() -> MemoryManager:
    """Get or create the singleton MemoryManager."""
    global _manager
    if _manager is None:
        _manager = MemoryManager()
    return _manager


def init_memory(api_key: Optional[str] = None, backfill: bool = True) -> MemoryManager:
    """
    Initialize the memory system.

    Args:
        api_key: Optional Gemini API key.
        backfill: Whether to backfill existing conversations.

    Returns:
        Initialized MemoryManager.
    """
    global _manager
    _manager = MemoryManager(api_key)

    if backfill:
        _manager.backfill_conversations()

        # Index any existing memory files
        memory_md = MEMORY_DIR / "MEMORY.md"
        context_md = MEMORY_DIR / "context.md"

        if memory_md.exists():
            _manager.index_file(memory_md, "memory")
        if context_md.exists():
            _manager.index_file(context_md, "context")

    return _manager


if __name__ == "__main__":
    # Test the memory manager
    logging.basicConfig(level=logging.INFO)

    print("Initializing memory manager...")
    manager = init_memory(backfill=True)

    print(f"\nStats: {manager.get_stats()}")

    # Test search
    query = "image editing"
    print(f"\nSearching for: {query}")
    results = manager.search(query, top_k=3)

    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} (score: {r.score:.3f}, source: {r.source}) ---")
        print(r.text[:200] + "..." if len(r.text) > 200 else r.text)

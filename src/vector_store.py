"""Local vector database (ChromaDB). One private collection per session."""
from __future__ import annotations

import threading
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.chunking import Chunk
from src.config import settings
from src.utils import get_logger

log = get_logger(__name__)

COLLECTION_PREFIX = "sess_"
ADD_BATCH = 100

_client: "chromadb.ClientAPI | None" = None
_client_lock = threading.Lock()


class VectorStoreError(Exception):
    """Vector database problem, worded so it can be shown to the user."""


@dataclass(frozen=True)
class Hit:
    """A chunk returned from the store (plus scores added during retrieval)."""

    id: str
    text: str
    source: str
    location: str
    kind: str
    chunk_index: int
    similarity: float = 0.0        # 0-1 meaning-based score (0 = not measured)
    keyword_coverage: float = 0.0  # share of the question's key words found in the chunk


def _get_client() -> "chromadb.ClientAPI":
    """One shared Chroma client for the whole app (created on first use)."""
    global _client
    with _client_lock:
        if _client is None:
            path = settings.data_dir / "chroma"
            path.mkdir(parents=True, exist_ok=True)
            _client = chromadb.PersistentClient(
                path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
            )
    return _client


def purge_all_sessions() -> int:
    """Delete every session collection. Call at app start: old sessions no longer exist."""
    client = _get_client()
    removed = 0
    for item in client.list_collections():
        name = str(getattr(item, "name", item))
        if name.startswith(COLLECTION_PREFIX):
            try:
                client.delete_collection(name)
                removed += 1
            except Exception:
                log.warning("Could not delete an old collection", exc_info=True)
    return removed


class SessionStore:
    """The documents of ONE user session. Other sessions can never see this collection."""

    def __init__(self, session_id: str) -> None:
        if not session_id.isalnum():
            raise ValueError("Invalid session id")
        self.session_id = session_id
        self._name = COLLECTION_PREFIX + session_id
        self._lock = threading.RLock()
        self._version = 0
        self._cache: list[Hit] | None = None
        self._closed = False
        try:
            self._collection = _get_client().get_or_create_collection(
                name=self._name,
                metadata={"hnsw:space": "cosine"},  # use cosine similarity
                embedding_function=None,            # we always supply our own vectors
            )
        except Exception as exc:
            log.exception("Could not open the vector database")
            raise VectorStoreError("The local vector database could not be opened.") from exc

    # ---- state -------------------------------------------------------
    @property
    def version(self) -> int:
        """Increases whenever the content changes (lets other code refresh caches)."""
        return self._version

    def _touch(self) -> None:
        self._version += 1
        self._cache = None

    def _require_open(self) -> None:
        if self._closed:
            raise VectorStoreError("This session's documents have been deleted.")

    # ---- writing -----------------------------------------------------
    def add_document(self, source: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Store a document's chunks. Re-adding the same file name replaces the old version."""
        self._require_open()
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return
        with self._lock:
            self.remove_document(source)
            try:
                for start in range(0, len(chunks), ADD_BATCH):
                    part = chunks[start : start + ADD_BATCH]
                    self._collection.add(
                        ids=[chunk.id for chunk in part],
                        documents=[chunk.text for chunk in part],
                        embeddings=embeddings[start : start + ADD_BATCH],
                        metadatas=[
                            {
                                "source": chunk.source,
                                "location": chunk.location,
                                "kind": chunk.kind,
                                "chunk_index": chunk.index,
                            }
                            for chunk in part
                        ],
                    )
            except Exception as exc:
                log.exception("Adding chunks failed")
                raise VectorStoreError(
                    "Could not save the document in the vector database. If you changed "
                    "EMBED_MODEL, restart the app so old data is cleared."
                ) from exc
            self._touch()

    def remove_document(self, source: str) -> None:
        self._require_open()
        with self._lock:
            try:
                self._collection.delete(where={"source": source})
            except Exception:
                log.warning("Removing a document failed", exc_info=True)
            self._touch()

    def clear(self) -> None:
        """Remove all documents but keep the (empty) collection."""
        self._require_open()
        with self._lock:
            ids = self._collection.get(include=[])["ids"]
            if ids:
                self._collection.delete(ids=ids)
            self._touch()

    def delete(self) -> None:
        """Delete the whole collection (session ended or timed out)."""
        with self._lock:
            if self._closed:
                return
            try:
                _get_client().delete_collection(self._name)
            except Exception:
                log.warning("Deleting the collection failed", exc_info=True)
            self._closed = True
            self._cache = None

    # ---- reading -----------------------------------------------------
    def all_chunks(self) -> list[Hit]:
        """Every stored chunk in reading order (cached until the content changes)."""
        self._require_open()
        with self._lock:
            if self._cache is None:
                data = self._collection.get(include=["documents", "metadatas"])
                hits = [
                    Hit(
                        id=chunk_id,
                        text=text or "",
                        source=str(meta.get("source", "")),
                        location=str(meta.get("location", "")),
                        kind=str(meta.get("kind", "text")),
                        chunk_index=int(meta.get("chunk_index", 0)),
                    )
                    for chunk_id, text, meta in zip(data["ids"], data["documents"], data["metadatas"])
                ]
                hits.sort(key=lambda hit: (hit.source, hit.chunk_index))
                self._cache = hits
            return self._cache

    def list_documents(self) -> dict[str, int]:
        """{file name: number of chunks}"""
        counts: dict[str, int] = {}
        for hit in self.all_chunks():
            counts[hit.source] = counts.get(hit.source, 0) + 1
        return counts

    def count(self) -> int:
        return len(self.all_chunks())

    def query(self, embedding: list[float], n: int, sources: list[str] | None = None) -> list[Hit]:
        """The n chunks closest in meaning to `embedding` (optionally only from some files)."""
        self._require_open()
        chunks = self.all_chunks()
        wanted = set(sources) if sources else None
        available = len(chunks) if wanted is None else sum(1 for c in chunks if c.source in wanted)
        n = min(n, available)
        if n <= 0:
            return []

        where = None
        if sources:
            where = {"source": sources[0]} if len(sources) == 1 else {"source": {"$in": list(sources)}}
        try:
            result = self._collection.query(
                query_embeddings=[embedding],
                n_results=n,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            log.exception("Vector search failed")
            raise VectorStoreError("The search in the vector database failed.") from exc

        hits: list[Hit] = []
        for chunk_id, text, meta, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            hits.append(
                Hit(
                    id=chunk_id,
                    text=text or "",
                    source=str(meta.get("source", "")),
                    location=str(meta.get("location", "")),
                    kind=str(meta.get("kind", "text")),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    similarity=min(1.0, max(0.0, 1.0 - float(distance))),  # cosine distance -> similarity
                )
            )
        return hits
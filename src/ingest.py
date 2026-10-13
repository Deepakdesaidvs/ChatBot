"""Turn an uploaded file into searchable chunks: save -> read -> chunk -> embed -> store."""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from src.chunking import chunk_document
from src.config import settings
from src.embeddings import embed_documents
from src.loaders import load_document, unsupported_reason
from src.loaders.base import ProgressFn
from src.utils import get_logger, human_size, session_dir
from src.vector_store import SessionStore

log = get_logger(__name__)

_UNSAFE_CHARS = re.compile(r"[^\w.\- ()]", re.UNICODE)


class UploadRejected(Exception):
    """The upload breaks a rule (type, size, count). The message is safe to show the user."""


@dataclass
class DocInfo:
    """What the UI shows about one processed document."""

    name: str
    size_bytes: int
    unit_count: int
    unit_name: str
    chunks: int
    chars: int
    warnings: list[str] = field(default_factory=list)


def safe_filename(name: str) -> str:
    """A harmless display name: no folders, no odd characters, lower-case extension."""
    base = Path(re.split(r"[\\/]", name)[-1])
    stem = _UNSAFE_CHARS.sub("_", base.stem).strip(" .")[:80] or "document"
    return stem + base.suffix.lower()


def check_upload(display: str, size: int, docs: Mapping[str, DocInfo]) -> None:
    """Basic rules. (Phase 7 adds deeper checks such as the file's real type.)"""
    reason = unsupported_reason(Path(display).suffix.lower())
    if reason:
        raise UploadRejected(f"{display}: {reason}")
    if size == 0:
        raise UploadRejected(f"{display}: the file is empty.")
    if size > settings.max_file_bytes:
        raise UploadRejected(
            f"{display}: {human_size(size)} is larger than the {settings.max_file_mb} MB limit."
        )
    if display not in docs and len(docs) >= settings.max_files:
        raise UploadRejected(
            f"{display}: you can have at most {settings.max_files} documents. Remove one first."
        )
    others = sum(info.size_bytes for name, info in docs.items() if name != display)
    if others + size > settings.max_total_bytes:
        raise UploadRejected(
            f"{display}: the total size would exceed {settings.max_total_mb} MB. Remove a document first."
        )


def _scaled(progress: ProgressFn | None, low: float, high: float) -> ProgressFn | None:
    """Map a 0-1 progress onto the range [low, high] of the overall job."""
    if progress is None:
        return None
    return lambda fraction, message: progress(low + (high - low) * fraction, message)


def ingest_file(
    store: SessionStore,
    session_id: str,
    filename: str,
    data: bytes,
    docs: Mapping[str, DocInfo],
    progress: ProgressFn | None = None,
) -> DocInfo:
    """Read an uploaded file and add it to the session's vector store.

    Raises UploadRejected, LoaderError, EmbeddingError or VectorStoreError (all friendly).
    """
    display = safe_filename(filename)
    check_upload(display, len(data), docs)

    # Save under a random name (the user's file name is never used on disk), read it, delete it.
    folder = session_dir(session_id) / "uploads"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{uuid.uuid4().hex}{Path(display).suffix}"
    path.write_bytes(data)
    try:
        doc = load_document(path, display_name=display, progress=_scaled(progress, 0.0, 0.6))
        chunks = chunk_document(doc)
        vectors = embed_documents([chunk.embed_text for chunk in chunks], progress=_scaled(progress, 0.6, 1.0))
        store.add_document(display, chunks, vectors)
    finally:
        path.unlink(missing_ok=True)

    log.info("Ingested %s: %d chunks", display, len(chunks))
    return DocInfo(
        name=display,
        size_bytes=len(data),
        unit_count=doc.unit_count,
        unit_name=doc.unit_name,
        chunks=len(chunks),
        chars=doc.total_chars,
        warnings=list(doc.warnings),
    )
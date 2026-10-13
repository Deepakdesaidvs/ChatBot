"""Small helpers shared by the whole app: logging, session ids, Ollama health check."""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterator

import ollama

from src.config import settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging() -> None:
    """Log to the console and to data/logs/app.log. Safe to call many times.

    Privacy note: we log events (file names, timings, errors) but never the
    contents of documents or questions.
    """
    root = logging.getLogger()
    if getattr(root, "_doc_chatbot_configured", False):
        return
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, settings.log_level, logging.INFO)
    root.setLevel(level)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        settings.logs_dir / "app.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(file_handler)

    # Quiet down chatty libraries.
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
        logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
    root._doc_chatbot_configured = True  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    """Get a named logger, e.g. log = get_logger(__name__)."""
    return logging.getLogger(name)


def new_session_id() -> str:
    """Random, unguessable id used to isolate one user's files and vectors."""
    return uuid.uuid4().hex


def session_dir(session_id: str) -> Path:
    """Folder that holds one session's uploads and vector store."""
    if not session_id.isalnum():  # blocks things like "../" path tricks
        raise ValueError("Invalid session id")
    path = settings.sessions_dir / session_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def human_size(num_bytes: float) -> str:
    """12345678 -> '11.8 MB'."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


@contextmanager
def timer(label: str) -> Iterator[None]:
    """Log how long a block takes:  with timer('embedding'): ..."""
    start = time.perf_counter()
    try:
        yield
    finally:
        get_logger("timer").info("%s took %.2fs", label, time.perf_counter() - start)


def _with_tag(name: str) -> str:
    """'nomic-embed-text' -> 'nomic-embed-text:latest' (how Ollama lists it)."""
    return name if ":" in name else f"{name}:latest"


def check_ollama(required: list[str] | None = None) -> dict:
    """Check that Ollama is running and the needed models are downloaded.

    Returns {"running": bool, "installed": [...], "missing": [...], "error": str}.
    """
    required = required if required is not None else [settings.llm_model, settings.embed_model]
    try:
        client = ollama.Client(host=settings.ollama_host, timeout=5)
        installed = [m.model for m in client.list().models if m.model]
    except Exception as exc:  # connection refused, timeout, etc.
        return {
            "running": False,
            "installed": [],
            "missing": list(required),
            "error": f"Cannot reach Ollama at {settings.ollama_host}: {exc}",
        }
    installed_set = set(installed)
    missing = [m for m in required if _with_tag(m) not in installed_set]
    return {"running": True, "installed": installed, "missing": missing, "error": ""}
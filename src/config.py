"""Central settings for the whole app.

Every value comes from environment variables (loaded from the .env file),
with safe defaults. Other modules do:  from src.config import settings
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Stop ChromaDB from sending anonymous usage statistics (privacy).
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

_log = logging.getLogger(__name__)
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class ConfigError(Exception):
    """Raised when a setting is unsafe or invalid."""


def _get_str(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        _log.warning("Setting %s=%r is not a whole number; using %s", name, raw, default)
        return default


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        _log.warning("Setting %s=%r is not a number; using %s", name, raw, default)
        return default


@dataclass(frozen=True)
class Settings:
    """All app settings. Frozen = cannot be changed accidentally at runtime."""

    # Model provider
    llm_provider: str
    ollama_host: str
    llm_model: str
    embed_model: str
    openai_api_key: str
    openai_chat_model: str
    openai_embed_model: str
    vision_model: str
    enable_vision: bool
    ocr_enabled: bool
    # Answer behaviour
    temperature: float
    num_ctx: int
    max_answer_tokens: int
    # Chunking and retrieval
    chunk_size: int
    chunk_overlap: int
    top_k: int
    min_relevance: float
    # Upload limits
    max_file_mb: int
    max_total_mb: int
    max_files: int
    max_pages: int
    # Sessions and sharing
    session_timeout_minutes: int
    app_password: str
    max_concurrent_llm: int
    rate_limit_per_minute: int
    # Misc
    data_dir: Path
    log_level: str

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_mb * 1024 * 1024

    @property
    def max_total_bytes(self) -> int:
        return self.max_total_mb * 1024 * 1024

    @classmethod
    def from_env(cls) -> "Settings":
        provider = _get_str("LLM_PROVIDER", "ollama").lower()
        if provider not in {"ollama", "openai"}:
            raise ConfigError("LLM_PROVIDER must be either 'ollama' or 'openai'.")

        host = _get_str("OLLAMA_HOST", "http://127.0.0.1:11434")
        hostname = urlparse(host).hostname or ""
        if provider == "ollama" and hostname not in _LOCAL_HOSTS and not _get_bool("ALLOW_REMOTE_OLLAMA", False):
            raise ConfigError(
                f"OLLAMA_HOST={host!r} is not localhost. For privacy the app only talks "
                "to a local Ollama. Use http://127.0.0.1:11434 (recommended)."
            )

        data_dir = Path(_get_str("DATA_DIR", "data"))
        if not data_dir.is_absolute():
            data_dir = PROJECT_ROOT / data_dir

        chunk_size = _get_int("CHUNK_SIZE", 1000, minimum=200)
        chunk_overlap = _get_int("CHUNK_OVERLAP", 150, minimum=0)
        if chunk_overlap >= chunk_size:
            raise ConfigError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

        return cls(
            llm_provider=provider,
            ollama_host=host,
            llm_model=_get_str("LLM_MODEL", "qwen2.5:3b"),
            embed_model=_get_str("EMBED_MODEL", "nomic-embed-text"),
            openai_api_key=_get_str("OPENAI_API_KEY", ""),
            openai_chat_model=_get_str("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            openai_embed_model=_get_str("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
            vision_model=_get_str("VISION_MODEL", "moondream"),
            enable_vision=_get_bool("ENABLE_VISION", False),
            ocr_enabled=_get_bool("OCR_ENABLED", True),
            temperature=min(max(_get_float("TEMPERATURE", 0.1), 0.0), 1.0),
            num_ctx=_get_int("NUM_CTX", 4096, minimum=1024),
            max_answer_tokens=_get_int("MAX_ANSWER_TOKENS", 600, minimum=50),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=_get_int("TOP_K", 6, minimum=1),
            min_relevance=min(max(_get_float("MIN_RELEVANCE", 0.35), 0.0), 1.0),
            max_file_mb=_get_int("MAX_FILE_MB", 25, minimum=1),
            max_total_mb=_get_int("MAX_TOTAL_MB", 100, minimum=1),
            max_files=_get_int("MAX_FILES", 10, minimum=1),
            max_pages=_get_int("MAX_PAGES", 200, minimum=1),
            session_timeout_minutes=_get_int("SESSION_TIMEOUT_MINUTES", 60, minimum=1),
            app_password=os.getenv("APP_PASSWORD", ""),
            max_concurrent_llm=_get_int("MAX_CONCURRENT_LLM", 1, minimum=1),
            rate_limit_per_minute=_get_int("RATE_LIMIT_PER_MINUTE", 6, minimum=1),
            data_dir=data_dir,
            log_level=_get_str("LOG_LEVEL", "INFO").upper(),
        )


settings = Settings.from_env()
"""Turn text into vectors (lists of numbers) using a local Ollama embedding model."""
from __future__ import annotations

from functools import lru_cache

import ollama
from openai import APIError, AuthenticationError

from src.config import settings
from src.loaders.base import ProgressFn, report
from src.openai_client import OpenAIConfigurationError, get_openai_client
from src.utils import get_logger, timer

log = get_logger(__name__)

BATCH_SIZE = 16          # texts sent to Ollama per request
MAX_EMBED_CHARS = 3000   # safety cut so one huge chunk cannot overflow the model


class EmbeddingError(Exception):
    """Embedding problem, worded so it can be shown to the user."""


@lru_cache(maxsize=1)
def _client() -> ollama.Client:
    return ollama.Client(host=settings.ollama_host, timeout=120)


def _prefixes() -> tuple[str, str]:
    """nomic-embed-text was trained with these task prefixes; other models use none."""
    if settings.llm_provider == "ollama" and settings.embed_model.lower().startswith("nomic-embed-text"):
        return "search_document: ", "search_query: "
    return "", ""


def _embed(texts: list[str]) -> list[list[float]]:
    try:
        if settings.llm_provider == "openai":
            response = get_openai_client().embeddings.create(
                model=settings.openai_embed_model,
                input=texts,
            )
            vectors = [
                list(item.embedding)
                for item in sorted(response.data, key=lambda item: item.index)
            ]
        else:
            response = _client().embed(model=settings.embed_model, input=texts)
            vectors = [list(vector) for vector in response.embeddings]
    except OpenAIConfigurationError as exc:
        raise EmbeddingError(str(exc)) from exc
    except AuthenticationError as exc:
        raise EmbeddingError("The OpenAI API key is invalid or does not have API access.") from exc
    except APIError as exc:
        log.exception("OpenAI embedding request failed")
        raise EmbeddingError(f"The OpenAI embedding request failed: {exc.message}") from exc
    except ollama.ResponseError as exc:
        if exc.status_code == 404:
            raise EmbeddingError(
                f"Embedding model '{settings.embed_model}' is not installed. "
                f"Run:  ollama pull {settings.embed_model}"
            ) from exc
        raise EmbeddingError(f"The embedding model returned an error: {exc}") from exc
    except Exception as exc:
        raise EmbeddingError("Could not reach Ollama. Is it running?") from exc

    if len(vectors) != len(texts):
        raise EmbeddingError("The embedding model returned an unexpected number of results.")
    return vectors


def embed_documents(texts: list[str], progress: ProgressFn | None = None) -> list[list[float]]:
    """Embed document chunks (in small batches so we can show progress)."""
    prefix, _ = _prefixes()
    prepared = [prefix + text[:MAX_EMBED_CHARS] for text in texts]
    vectors: list[list[float]] = []
    with timer(f"embedding {len(texts)} chunks"):
        for start in range(0, len(prepared), BATCH_SIZE):
            report(progress, start / max(len(prepared), 1), f"Indexing {start + 1}-{min(start + BATCH_SIZE, len(prepared))} of {len(prepared)}")
            vectors.extend(_embed(prepared[start : start + BATCH_SIZE]))
    report(progress, 1.0, "Indexing done")
    return vectors


def embed_query(question: str) -> list[float]:
    """Embed a user question."""
    _, prefix = _prefixes()
    return _embed([prefix + question[:MAX_EMBED_CHARS]])[0]
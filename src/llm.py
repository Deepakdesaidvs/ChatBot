"""Talking to the local language model through Ollama (streaming, with a concurrency limit)."""
from __future__ import annotations

import threading
import time
from functools import lru_cache
from typing import Iterator

import ollama
from openai import APIError, AuthenticationError

from src.config import settings
from src.openai_client import OpenAIConfigurationError, get_openai_client
from src.utils import get_logger

log = get_logger(__name__)

KEEP_ALIVE = "15m"               # keep the model in memory between questions (faster replies)
QUEUE_TIMEOUT_SECONDS = 180      # how long a request waits for its turn

_slots = threading.BoundedSemaphore(settings.max_concurrent_llm)


class LLMError(Exception):
    """Language-model problem, worded so it can be shown to the user."""


class LLMBusyError(LLMError):
    """Too many requests at once."""


@lru_cache(maxsize=1)
def _client() -> ollama.Client:
    return ollama.Client(host=settings.ollama_host, timeout=300)


def _options(temperature: float | None, max_tokens: int | None) -> dict:
    return {
        "temperature": settings.temperature if temperature is None else temperature,
        "num_ctx": settings.num_ctx,
        "num_predict": settings.max_answer_tokens if max_tokens is None else max_tokens,
        "top_p": 0.9,
        # 1.0 = no penalty. The default penalty can damage exact copying of numbers and ids.
        "repeat_penalty": 1.0,
    }


def stream_chat(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Iterator[str]:
    """Yield the configured model's answer piece by piece."""
    model = model or (
        settings.openai_chat_model if settings.llm_provider == "openai" else settings.llm_model
    )
    if not _slots.acquire(timeout=QUEUE_TIMEOUT_SECONDS):
        raise LLMBusyError("The assistant is busy with other requests. Please try again in a minute.")
    try:
        started = time.perf_counter()
        tokens = 0
        try:
            if settings.llm_provider == "openai":
                stream = get_openai_client().chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    stream_options={"include_usage": True},
                    temperature=settings.temperature if temperature is None else temperature,
                    max_tokens=settings.max_answer_tokens if max_tokens is None else max_tokens,
                    top_p=0.9,
                )
                for part in stream:
                    if part.usage:
                        tokens = part.usage.completion_tokens
                    if part.choices:
                        piece = part.choices[0].delta.content or ""
                        if piece:
                            yield piece
            else:
                stream = _client().chat(
                    model=model,
                    messages=messages,
                    stream=True,
                    options=_options(temperature, max_tokens),
                    keep_alive=KEEP_ALIVE,
                )
                for part in stream:
                    piece = part.message.content or ""
                    if piece:
                        yield piece
                    if part.done:
                        tokens = part.eval_count or 0
        except OpenAIConfigurationError as exc:
            raise LLMError(str(exc)) from exc
        except AuthenticationError as exc:
            raise LLMError("The OpenAI API key is invalid or does not have API access.") from exc
        except APIError as exc:
            log.exception("OpenAI chat request failed")
            raise LLMError(f"The OpenAI API request failed: {exc.message}") from exc
        except ollama.ResponseError as exc:
            if exc.status_code == 404:
                raise LLMError(
                    f"The language model '{model}' is not installed. Run:  ollama pull {model}"
                ) from exc
            raise LLMError(f"The language model returned an error: {exc}") from exc
        except LLMError:
            raise
        except Exception as exc:
            log.exception("LLM request failed")
            raise LLMError("Could not reach the language model. Is Ollama running?") from exc

        elapsed = max(time.perf_counter() - started, 0.001)
        log.info("Generated %d tokens in %.1fs (%.1f tokens/s)", tokens, elapsed, tokens / elapsed)
    finally:
        _slots.release()


def chat(messages: list[dict[str, str]], **kwargs) -> str:
    """Get the complete answer as one string."""
    return "".join(stream_chat(messages, **kwargs)).strip()
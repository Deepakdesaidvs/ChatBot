"""Shared OpenAI client configured from the server-side API key."""
from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

from src.config import settings


class OpenAIConfigurationError(Exception):
    """The hosted model provider is not configured."""


@lru_cache(maxsize=1)
def _client_for_key(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, timeout=300)


def get_openai_client() -> OpenAI:
    """Return a cached client without exposing the API key to the browser."""
    if not settings.openai_api_key:
        raise OpenAIConfigurationError(
            "OPENAI_API_KEY is not configured on the server."
        )
    return _client_for_key(settings.openai_api_key)

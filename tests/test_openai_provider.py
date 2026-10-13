"""Offline checks for OpenAI chat and embedding provider adapters."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from src.embeddings import _embed
from src.llm import stream_chat


def test_openai_chat_stream() -> None:
    stream = [
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Answer"))],
            usage=None,
        ),
        SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(completion_tokens=1),
        ),
    ]
    chat_options = {}

    def create_chat(**kwargs):
        chat_options.update(kwargs)
        return iter(stream)

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create_chat)
        )
    )
    provider_settings = SimpleNamespace(
        llm_provider="openai",
        openai_chat_model="test-chat-model",
        temperature=0.1,
        max_answer_tokens=50,
    )

    with (
        patch("src.llm.settings", provider_settings),
        patch("src.llm.get_openai_client", return_value=client),
    ):
        assert list(stream_chat([{"role": "user", "content": "Question"}])) == ["Answer"]
    assert chat_options["model"] == "test-chat-model"
    assert chat_options["stream"] is True


def test_openai_embeddings_are_returned_in_input_order() -> None:
    response = SimpleNamespace(
        data=[
            SimpleNamespace(index=1, embedding=[0.2, 0.3]),
            SimpleNamespace(index=0, embedding=[0.1, 0.2]),
        ]
    )
    embedding_options = {}

    def create_embeddings(**kwargs):
        embedding_options.update(kwargs)
        return response

    client = SimpleNamespace(embeddings=SimpleNamespace(create=create_embeddings))
    provider_settings = SimpleNamespace(
        llm_provider="openai",
        openai_embed_model="test-embedding-model",
    )

    with (
        patch("src.embeddings.settings", provider_settings),
        patch("src.embeddings.get_openai_client", return_value=client),
    ):
        assert _embed(["first", "second"]) == [[0.1, 0.2], [0.2, 0.3]]
    assert embedding_options == {
        "model": "test-embedding-model",
        "input": ["first", "second"],
    }


def main() -> None:
    test_openai_chat_stream()
    test_openai_embeddings_are_returned_in_input_order()
    print("2 OpenAI provider checks passed.")


if __name__ == "__main__":
    main()

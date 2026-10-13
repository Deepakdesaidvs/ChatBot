"""Optional image understanding with a local vision model through Ollama.

Slow on CPU (tens of seconds per image) and small models make mistakes, so it
is OFF by default (ENABLE_VISION=false). OCR is used for any text in the image.
"""
from __future__ import annotations

import io
import threading

import ollama
from PIL import Image

from src.config import settings
from src.ocr import prepare_image
from src.utils import get_logger

log = get_logger(__name__)

_vision_lock = threading.Lock()  # one vision job at a time
VISION_MAX_SIDE_PX = 1024        # smaller image = faster on CPU

DEFAULT_PROMPT = (
    "Describe this image factually. Transcribe any visible text exactly. "
    "Mention numbers, labels, and what any chart, table or diagram shows. "
    "Do not guess about things you cannot see."
)


class VisionError(Exception):
    """Vision problem, worded so it can be shown to the user."""


def describe_image(image: Image.Image, question: str | None = None) -> str:
    """Ask the vision model to describe the image (or answer a question about it)."""
    if not settings.enable_vision:
        raise VisionError("Image understanding is turned off. Set ENABLE_VISION=true in .env.")

    buffer = io.BytesIO()
    prepare_image(image, max_side=VISION_MAX_SIDE_PX).save(buffer, format="PNG")
    prompt = question.strip() if question and question.strip() else DEFAULT_PROMPT

    client = ollama.Client(host=settings.ollama_host, timeout=300)
    try:
        with _vision_lock:
            response = client.chat(
                model=settings.vision_model,
                messages=[{"role": "user", "content": prompt, "images": [buffer.getvalue()]}],
                options={"temperature": 0.1, "num_ctx": 2048},
            )
    except ollama.ResponseError as exc:
        if exc.status_code == 404:
            raise VisionError(
                f"Vision model '{settings.vision_model}' is not installed. "
                f"Run:  ollama pull {settings.vision_model}"
            ) from exc
        raise VisionError(f"The vision model returned an error: {exc}") from exc
    except Exception as exc:
        log.exception("Vision request failed")
        raise VisionError("Could not reach the vision model. Is Ollama running?") from exc

    return (response.message.content or "").strip()
"""OCR (reading text from images) with RapidOCR: pip-only, no extra programs needed."""
from __future__ import annotations

import threading
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from src.config import settings
from src.utils import get_logger

log = get_logger(__name__)

# Protect against "decompression bomb" images (tiny file, gigantic pixel count).
Image.MAX_IMAGE_PIXELS = 60_000_000

MIN_CONFIDENCE = 0.5  # ignore OCR guesses the engine is unsure about
MAX_SIDE_PX = 2000    # bigger images are shrunk: faster OCR, similar accuracy

_engine: Any = None
_engine_lock = threading.Lock()  # protects creating the engine
_ocr_lock = threading.Lock()     # one OCR job at a time, so the CPU is not swamped


def _get_engine() -> Any:
    """Create the OCR engine on first use (loading takes a few seconds)."""
    global _engine
    with _engine_lock:
        if _engine is None:
            from rapidocr_onnxruntime import RapidOCR

            log.info("Loading OCR engine...")
            _engine = RapidOCR()
    return _engine


def prepare_image(image: Image.Image, max_side: int = MAX_SIDE_PX) -> Image.Image:
    """Return an RGB copy: rotated upright, transparency flattened onto white, resized."""
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, "white")
        background.paste(rgba, mask=rgba.split()[-1])
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")

    longest = max(image.size)
    if longest > max_side:
        scale = max_side / longest
        new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image


def _lines_to_text(result: list) -> str:
    """Turn OCR boxes into text in reading order (top to bottom, left to right)."""
    items: list[tuple[float, float, float, str]] = []  # (center_y, height, left_x, text)
    for box, text, score in result:
        try:
            if float(score) < MIN_CONFIDENCE or not str(text).strip():
                continue
            ys = [point[1] for point in box]
            xs = [point[0] for point in box]
            items.append(((min(ys) + max(ys)) / 2, max(ys) - min(ys), min(xs), str(text).strip()))
        except (TypeError, ValueError, IndexError):
            continue

    items.sort(key=lambda item: item[0])
    rows: list[list[tuple[float, float, float, str]]] = []
    row_centers: list[float] = []
    for item in items:
        if rows and abs(item[0] - row_centers[-1]) < 0.5 * max(item[1], 1.0):
            rows[-1].append(item)  # same visual line
        else:
            rows.append([item])
            row_centers.append(item[0])

    lines = [" ".join(part[3] for part in sorted(row, key=lambda p: p[2])) for row in rows]
    return "\n".join(lines).strip()


def ocr_image(image: Image.Image) -> str:
    """Read the text in a PIL image. Returns "" if OCR is off, fails, or finds nothing."""
    if not settings.ocr_enabled:
        return ""
    try:
        prepared = prepare_image(image)
        array = np.asarray(prepared)[:, :, ::-1].copy()  # RGB -> BGR, what OpenCV tools expect
        engine = _get_engine()
        with _ocr_lock:
            result, _ = engine(array)
    except Exception:  # OCR must never crash the whole upload
        log.exception("OCR failed for one image")
        return ""
    return _lines_to_text(result or [])
"""Image loader (PNG/JPG/JPEG/WEBP): OCR text, plus an optional vision-model description."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.config import settings
from src.loaders.base import DocElement, LoadedDocument, LoaderError, ProgressFn, report
from src.ocr import ocr_image
from src.utils import get_logger
from src.vision import VisionError, describe_image

log = get_logger(__name__)


def load_image(path: Path, display_name: str, progress: ProgressFn | None = None) -> LoadedDocument:
    try:
        with Image.open(path) as opened:
            opened.load()
            image = opened.copy()
    except Exception as exc:
        raise LoaderError("This image could not be opened. It may be damaged or too large.") from exc

    doc = LoadedDocument(filename=display_name, unit_count=1, unit_name="images")

    if settings.ocr_enabled:
        report(progress, 0.1, "Reading text in the image (OCR)")
        text = ocr_image(image)
        if text:
            doc.elements.append(DocElement(text, display_name, "Image (text found by OCR)", "ocr"))

    if settings.enable_vision:
        report(progress, 0.5, "Describing the image with the vision model (slow on CPU)")
        try:
            caption = describe_image(image)
            if caption:
                doc.elements.append(DocElement(caption, display_name, "Image (AI description)", "image_caption"))
        except VisionError as exc:
            doc.warnings.append(str(exc))

    report(progress, 1.0, "Done")
    return doc
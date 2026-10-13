"""One entry point for all file types:  load_document(path)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.loaders.base import MAX_TOTAL_CHARS, LoadedDocument, LoaderError, ProgressFn
from src.loaders.docx_loader import load_docx
from src.loaders.excel_loader import load_csv, load_excel
from src.loaders.image_loader import load_image
from src.loaders.pdf_loader import load_pdf
from src.loaders.pptx_loader import load_pptx
from src.loaders.txt_loader import load_txt
from src.utils import get_logger, timer

log = get_logger(__name__)

Loader = Callable[[Path, str, "ProgressFn | None"], LoadedDocument]

_LOADERS: dict[str, Loader] = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".xlsx": load_excel,
    ".xls": load_excel,
    ".csv": load_csv,
    ".txt": load_txt,
    ".pptx": load_pptx,
    ".png": load_image,
    ".jpg": load_image,
    ".jpeg": load_image,
    ".webp": load_image,
}

SUPPORTED_EXTENSIONS = frozenset(_LOADERS)

_LEGACY_HELP = {
    ".doc": "Old .doc files are not supported. Open it in Word and use Save As > .docx.",
    ".ppt": "Old .ppt files are not supported. Open it in PowerPoint and use Save As > .pptx.",
    ".docm": "Macro-enabled files are not accepted. Save a copy as .docx.",
    ".xlsm": "Macro-enabled files are not accepted. Save a copy as .xlsx.",
    ".pptm": "Macro-enabled files are not accepted. Save a copy as .pptx.",
}


def _empty_message(ext: str) -> str:
    if ext in {".png", ".jpg", ".jpeg", ".webp"}:
        return (
            "No readable text was found in this image. To describe pictures without text, "
            "set ENABLE_VISION=true in .env (slow on CPU)."
        )
    if ext == ".pdf":
        return (
            "No text was found. If this is a scanned document, make sure OCR_ENABLED=true in .env."
        )
    return "No readable content was found in this file."

def unsupported_reason(ext: str) -> str | None:
    """None if the extension is supported, otherwise a friendly explanation."""
    if ext in SUPPORTED_EXTENSIONS:
        return None
    if ext in _LEGACY_HELP:
        return _LEGACY_HELP[ext]
    supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    return f"Files of type '{ext or 'without an extension'}' are not supported. Supported: {supported}"

def load_document(
    path: Path, display_name: str | None = None, progress: ProgressFn | None = None
) -> LoadedDocument:
    """Read any supported file. Raises LoaderError with a user-friendly message on failure."""
    name = display_name or path.name
    ext = path.suffix.lower()

    try:
        if ext in _LEGACY_HELP:
            raise LoaderError(_LEGACY_HELP[ext])
        loader = _LOADERS.get(ext)
        if loader is None:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise LoaderError(f"Files of type '{ext}' are not supported. Supported: {supported}")

        with timer(f"load {name}"):
            doc = loader(path, name, progress)

        if not doc.elements:
            raise LoaderError(_empty_message(ext))
        if doc.total_chars > MAX_TOTAL_CHARS:
            raise LoaderError("This file contains too much text for a PC to process comfortably.")
    except LoaderError as exc:
        raise LoaderError(f"{name}: {exc}") from exc
    except Exception as exc:
        log.exception("Unexpected error while loading %s", name)
        raise LoaderError(
            f"{name}: could not be read ({type(exc).__name__}). The file may be damaged."
        ) from exc

    log.info("Loaded %s: %d pieces, %d characters", name, len(doc.elements), doc.total_chars)
    return doc


__all__ = ["load_document", "LoadedDocument", "LoaderError", "SUPPORTED_EXTENSIONS", "unsupported_reason"]
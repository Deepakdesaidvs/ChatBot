"""PDF loader: text and tables with PyMuPDF; OCR for scanned pages and big embedded images."""
from __future__ import annotations

import io
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from src.config import settings
from src.loaders.base import (
    MIN_PAGE_TEXT_CHARS,
    DocElement,
    LoadedDocument,
    LoaderError,
    OcrBudget,
    ProgressFn,
    report,
    rows_to_markdown,
)
from src.ocr import MAX_SIDE_PX, ocr_image
from src.utils import get_logger

log = get_logger(__name__)

MAX_IMAGES_PER_PAGE = 3
MIN_EMBEDDED_IMAGE_PX = 300   # skip icons and logos
MIN_EMBEDDED_OCR_CHARS = 20   # ignore OCR results that are probably noise


def _render_page(page: "fitz.Page") -> Image.Image:
    """Draw a PDF page as an image (about 2000 px on the long side) for OCR."""
    longest = max(page.rect.width, page.rect.height, 1)
    zoom = min(4.0, MAX_SIDE_PX / longest)
    pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def _page_tables(page: "fitz.Page") -> list[str]:
    """Find tables on a page and return each as Markdown. Never raises."""
    found: list[str] = []
    try:
        for table in page.find_tables().tables:
            markdown = rows_to_markdown(table.extract())
            if markdown:
                found.append(markdown)
    except Exception:
        log.debug("Table detection failed on one page", exc_info=True)
    return found


def load_pdf(path: Path, display_name: str, progress: ProgressFn | None = None) -> LoadedDocument:
    try:
        pdf = fitz.open(str(path))
    except Exception as exc:
        raise LoaderError("This PDF could not be opened. It may be damaged.") from exc

    try:
        if not pdf.is_pdf:
            raise LoaderError("This file is not a real PDF.")
        if pdf.needs_pass:
            raise LoaderError("This PDF is password-protected. Remove the password and upload it again.")
        total = pdf.page_count
        if total == 0:
            raise LoaderError("This PDF has no pages.")
        if total > settings.max_pages:
            raise LoaderError(f"This PDF has {total} pages; the limit is {settings.max_pages}.")

        doc = LoadedDocument(filename=display_name, unit_count=total, unit_name="pages")
        budget = OcrBudget()
        seen_images: set[int] = set()
        skipped_pages: list[int] = []

        for index in range(total):
            number = index + 1
            location = f"Page {number}"
            report(progress, index / total, f"Reading page {number} of {total}")
            page = pdf[index]
            text = page.get_text("text", sort=True).strip()

            if text:
                doc.elements.append(DocElement(text, display_name, location))
                for markdown in _page_tables(page):
                    doc.elements.append(DocElement(markdown, display_name, f"{location} (table)", "table"))

            if len(text) < MIN_PAGE_TEXT_CHARS:
                # Probably a scan (a photo of a page): read it with OCR.
                if settings.ocr_enabled and budget.take():
                    report(progress, index / total, f"Running OCR on page {number} of {total}")
                    ocr_text = ocr_image(_render_page(page))
                    if ocr_text:
                        doc.elements.append(DocElement(ocr_text, display_name, location, "ocr"))
                elif settings.ocr_enabled:
                    skipped_pages.append(number)
            elif settings.ocr_enabled:
                # Normal text page: also read large embedded pictures (screenshots, scanned inserts).
                for info in page.get_images(full=True)[:MAX_IMAGES_PER_PAGE]:
                    xref = info[0]
                    if xref in seen_images:
                        continue
                    seen_images.add(xref)
                    try:
                        extracted = pdf.extract_image(xref)
                        if min(extracted["width"], extracted["height"]) < MIN_EMBEDDED_IMAGE_PX:
                            continue
                        if not budget.take():
                            break
                        image = Image.open(io.BytesIO(extracted["image"]))
                        image.load()
                    except Exception:
                        continue
                    ocr_text = ocr_image(image)
                    if len(ocr_text) >= MIN_EMBEDDED_OCR_CHARS:
                        doc.elements.append(
                            DocElement(ocr_text, display_name, f"{location} (embedded image)", "ocr")
                        )

        if skipped_pages:
            shown = ", ".join(str(n) for n in skipped_pages[:10])
            more = "..." if len(skipped_pages) > 10 else ""
            doc.warnings.append(
                f"OCR limit reached: pages {shown}{more} look scanned but were not read. "
                "Split the PDF into smaller files to read them."
            )
        report(progress, 1.0, "Done")
        return doc
    finally:
        pdf.close()
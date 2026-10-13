"""Word (.docx) loader: text grouped by heading, tables as Markdown, OCR on embedded images."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Iterator, Union

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from PIL import Image

from src.config import settings
from src.loaders.base import (
    DocElement,
    LoadedDocument,
    LoaderError,
    OcrBudget,
    ProgressFn,
    report,
    rows_to_markdown,
)
from src.ocr import ocr_image
from src.utils import get_logger

log = get_logger(__name__)

MIN_IMAGE_PX = 200
MIN_IMAGE_OCR_CHARS = 20


def _iter_blocks(doc) -> Iterator[Union[Paragraph, Table]]:
    """Walk paragraphs and tables in the order they appear in the document."""
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            yield Table(child, doc)


def _style_name(paragraph: Paragraph) -> str:
    try:
        return paragraph.style.name or ""
    except Exception:
        return ""


def _table_rows(table: Table) -> list[list[str]]:
    """Table cells as text; merged cells are only counted once per row."""
    rows: list[list[str]] = []
    for row in table.rows:
        seen: list[object] = []
        cells: list[str] = []
        for cell in row.cells:
            if any(cell._tc is previous for previous in seen):
                continue
            seen.append(cell._tc)
            cells.append(cell.text)
        rows.append(cells)
    return rows


def load_docx(path: Path, display_name: str, progress: ProgressFn | None = None) -> LoadedDocument:
    try:
        word = Document(str(path))
    except Exception as exc:
        raise LoaderError(
            "This Word file could not be opened. It may be damaged or password-protected."
        ) from exc

    doc = LoadedDocument(filename=display_name, unit_name="sections")
    blocks = list(_iter_blocks(word))
    heading = "start of document"
    buffer: list[str] = []
    sections = 0

    def flush(current_heading: str) -> None:
        nonlocal sections
        if buffer:
            doc.elements.append(DocElement("\n".join(buffer), display_name, f"Section: {current_heading}"))
            buffer.clear()
            sections += 1

    for index, block in enumerate(blocks):
        if index % 50 == 0:
            report(progress, index / max(len(blocks), 1) * 0.8, "Reading text")
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue
            style = _style_name(block)
            if style == "Title" or style.startswith("Heading"):
                flush(heading)           # close the previous section
                heading = text[:80]
                buffer.append(text)
            elif style.startswith("List"):
                buffer.append("- " + text)
            else:
                buffer.append(text)
        else:
            markdown = rows_to_markdown(_table_rows(block))
            if markdown:
                doc.elements.append(DocElement(markdown, display_name, f"Section: {heading} (table)", "table"))
    flush(heading)
    doc.unit_count = sections

    # Pictures inside the document (scanned inserts, screenshots, charts with text).
    if settings.ocr_enabled:
        budget = OcrBudget()
        parts = [
            rel.target_part
            for rel in word.part.rels.values()
            if not rel.is_external and rel.reltype.endswith("/image")
        ]
        for number, part in enumerate(parts, start=1):
            report(progress, 0.8 + 0.2 * number / max(len(parts), 1), f"Reading image {number} of {len(parts)}")
            try:
                image = Image.open(io.BytesIO(part.blob))
                image.load()
            except Exception:
                continue  # unsupported picture format (e.g. EMF/WMF vector art)
            if min(image.size) < MIN_IMAGE_PX:
                continue
            if not budget.take():
                doc.warnings.append("OCR limit reached: some images in this document were not read.")
                break
            text = ocr_image(image)
            if len(text) >= MIN_IMAGE_OCR_CHARS:
                doc.elements.append(DocElement(text, display_name, f"Embedded image {number}", "ocr"))

    report(progress, 1.0, "Done")
    return doc
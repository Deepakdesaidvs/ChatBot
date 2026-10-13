"""Shared building blocks for every document loader."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# progress(fraction_from_0_to_1, message): used by the UI progress bar.
ProgressFn = Callable[[float, str], None]

MAX_OCR_PER_DOCUMENT = 40    # OCR is slow on CPU, so cap it per file
MIN_PAGE_TEXT_CHARS = 30     # a PDF page with less text than this is treated as scanned
MAX_TOTAL_CHARS = 1_500_000  # refuse documents with an absurd amount of text


class LoaderError(Exception):
    """A problem with a file, worded so it can be shown directly to the user."""


@dataclass
class DocElement:
    """One piece of extracted content plus where it came from (for citations)."""

    text: str
    source: str            # file name shown to the user
    location: str          # "Page 3", "Sheet: Sales (data rows 1-25)", "Slide 2", ...
    kind: str = "text"     # text | table | ocr | image_caption


@dataclass
class LoadedDocument:
    """Everything extracted from one file."""

    filename: str
    elements: list[DocElement] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    unit_count: int = 0           # number of pages / slides / sheets
    unit_name: str = "pages"

    @property
    def total_chars(self) -> int:
        return sum(len(element.text) for element in self.elements)


class OcrBudget:
    """Counts OCR calls so one huge file cannot keep the CPU busy for an hour."""

    def __init__(self, limit: int = MAX_OCR_PER_DOCUMENT) -> None:
        self.limit = limit
        self.used = 0

    def take(self) -> bool:
        """Return True (and use one credit) if OCR is still allowed."""
        if self.used >= self.limit:
            return False
        self.used += 1
        return True


def report(progress: ProgressFn | None, fraction: float, message: str) -> None:
    """Call the progress callback safely."""
    if progress is not None:
        progress(min(max(fraction, 0.0), 1.0), message)


def clean_cell(value: object) -> str:
    """One table cell as safe single-line text."""
    text = "" if value is None else str(value)
    return " ".join(text.replace("|", "\\|").split())


def rows_to_markdown(rows: list[list[object]]) -> str:
    """Convert rows (first row = header) to a Markdown table. Returns "" if empty."""
    cleaned = [[clean_cell(cell) for cell in row] for row in rows]
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return ""
    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]
    header, body = cleaned[0], cleaned[1:]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)
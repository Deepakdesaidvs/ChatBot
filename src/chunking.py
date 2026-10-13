"""Split loaded content into chunks that are easy to search and fit in the model's context."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from src.config import settings
from src.loaders.base import LoadedDocument

TABLE_MAX_FACTOR = 1.5          # tables up to 1.5x the chunk size are kept in one piece
MIN_WINDOW_FRACTION = 0.6       # look for a natural cut point in the last 40% of a chunk
_BREAKS = ("\n\n", "\n", ". ", "; ", ", ", " ")  # best cut points first


def source_key(source: str) -> str:
    """Short stable id for a file name (used to build unique chunk ids)."""
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class Chunk:
    """One searchable piece of a document."""

    text: str
    source: str     # file name
    location: str   # "Page 3", "Sheet: Sales (data rows 1-25)", ...
    kind: str       # text | table | ocr | image_caption
    index: int      # position inside its document (keeps reading order)

    @property
    def id(self) -> str:
        return f"{source_key(self.source)}-{self.index}"

    @property
    def embed_text(self) -> str:
        """What we turn into a vector: the text plus where it came from."""
        return f"{self.source} | {self.location}\n{self.text}"


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Cut text into pieces of at most `size` characters, repeating `overlap` characters."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    overlap = max(0, min(overlap, size // 2))  # an overlap above half a chunk makes no sense
    chunks: list[str] = []
    start, length = 0, len(text)
    while start < length:
        end = min(start + size, length)
        if end < length:
            window_start = start + int(size * MIN_WINDOW_FRACTION)
            for separator in _BREAKS:
                cut = text.rfind(separator, window_start, end)
                if cut != -1:
                    end = cut + len(separator)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= length:
            break
        next_start = max(end - overlap, start + 1)  # always move forward
        if not text[next_start - 1].isspace():      # start the overlap at a word boundary
            space = text.find(" ", next_start, end)
            if space != -1:
                next_start = space + 1
        start = next_start
    return chunks


def split_table(markdown: str, size: int) -> list[str]:
    """Keep a Markdown table whole if it is small; otherwise split by rows, repeating the header."""
    lines = markdown.splitlines()
    if len(markdown) <= size * TABLE_MAX_FACTOR or len(lines) < 4:
        return [markdown]

    header = "\n".join(lines[:2])  # column names + the |---| line
    pieces: list[str] = []
    current: list[str] = []
    length = len(header)
    for row in lines[2:]:
        if current and length + len(row) + 1 > size:
            pieces.append(header + "\n" + "\n".join(current))
            current, length = [], len(header)
        current.append(row)
        length += len(row) + 1
    if current:
        pieces.append(header + "\n" + "\n".join(current))
    return pieces


def chunk_document(
    doc: LoadedDocument, size: int | None = None, overlap: int | None = None
) -> list[Chunk]:
    """Turn a loaded document into chunks. Sizes default to the values in .env."""
    size = size or settings.chunk_size
    overlap = settings.chunk_overlap if overlap is None else overlap

    chunks: list[Chunk] = []
    for element in doc.elements:
        if element.kind == "table":
            pieces = split_table(element.text, size)
        else:
            pieces = split_text(element.text, size, overlap)
        for piece in pieces:
            chunks.append(Chunk(piece, element.source, element.location, element.kind, len(chunks)))
    return chunks
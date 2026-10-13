"""Plain text loader. Splits text into blocks and records line numbers for citations."""
from __future__ import annotations

from pathlib import Path

from src.config import settings
from src.loaders.base import DocElement, LoadedDocument, LoaderError, ProgressFn, report

SOFT_BLOCK_CHARS = 2500   # close a block at the next blank line after this many characters
HARD_BLOCK_CHARS = 4000   # close a block no matter what


def decode_bytes(data: bytes) -> str:
    """Decode text trying common Windows encodings. latin-1 never fails, so this always returns."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def load_txt(path: Path, display_name: str, progress: ProgressFn | None = None) -> LoadedDocument:
    try:
        data = path.read_bytes()[: settings.max_file_bytes]
    except OSError as exc:
        raise LoaderError("This text file could not be read.") from exc
    if b"\x00" in data[:4096] and not data.startswith((b"\xff\xfe", b"\xfe\xff")):
        raise LoaderError("This does not look like a text file (it contains binary data).")

    lines = decode_bytes(data).splitlines()
    doc = LoadedDocument(filename=display_name, unit_name="blocks")
    buffer: list[str] = []
    start, size = 1, 0

    def flush(end: int) -> None:
        nonlocal start, size
        body = "\n".join(buffer).strip()
        if body:
            doc.elements.append(DocElement(body, display_name, f"Lines {start}-{end}"))
        buffer.clear()
        start, size = end + 1, 0

    for number, line in enumerate(lines, start=1):
        buffer.append(line)
        size += len(line) + 1
        if (size >= SOFT_BLOCK_CHARS and not line.strip()) or size >= HARD_BLOCK_CHARS:
            flush(number)
    flush(len(lines))

    doc.unit_count = len(doc.elements)
    report(progress, 1.0, "Done")
    return doc
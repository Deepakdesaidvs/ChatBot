"""Excel (.xlsx/.xls) and CSV loader. Rows are grouped in small blocks, header repeated in each."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.loaders.base import (
    DocElement,
    LoadedDocument,
    LoaderError,
    ProgressFn,
    report,
    rows_to_markdown,
)
from src.utils import get_logger

log = get_logger(__name__)

MAX_ROWS_PER_SHEET = 5000
MAX_SHEETS = 30
ROWS_PER_BLOCK = 25
CSV_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


def _frame_to_elements(df: pd.DataFrame, source: str, label: str, warnings: list[str]) -> list[DocElement]:
    """Turn one table into a summary element plus blocks of rows."""
    df = df.fillna("").astype(str)
    df = df[(df != "").any(axis=1)].reset_index(drop=True)  # drop completely empty rows
    df.columns = [
        (str(name).strip() if str(name).strip() else f"Column {i + 1}") for i, name in enumerate(df.columns)
    ]
    if df.empty:
        return []

    if len(df) > MAX_ROWS_PER_SHEET:
        warnings.append(f"{label}: only the first {MAX_ROWS_PER_SHEET} of {len(df)} rows were read.")
        df = df.head(MAX_ROWS_PER_SHEET)

    columns = list(df.columns)
    elements = [
        DocElement(
            f"{label}: {len(df)} data rows. Columns: {', '.join(columns)}",
            source,
            label,
            "table",
        )
    ]
    for start in range(0, len(df), ROWS_PER_BLOCK):
        block = df.iloc[start : start + ROWS_PER_BLOCK]
        markdown = rows_to_markdown([columns] + block.values.tolist())
        if markdown:
            location = f"{label} (data rows {start + 1}-{start + len(block)})"
            elements.append(DocElement(markdown, source, location, "table"))
    return elements


def load_excel(path: Path, display_name: str, progress: ProgressFn | None = None) -> LoadedDocument:
    engine = "xlrd" if path.suffix.lower() == ".xls" else "openpyxl"
    try:
        sheets = pd.read_excel(path, sheet_name=None, dtype=str, keep_default_na=False, engine=engine)
    except Exception as exc:
        raise LoaderError(
            "This spreadsheet could not be read. It may be damaged or password-protected."
        ) from exc

    doc = LoadedDocument(filename=display_name, unit_count=len(sheets), unit_name="sheets")
    names = list(sheets)
    if len(names) > MAX_SHEETS:
        doc.warnings.append(f"Only the first {MAX_SHEETS} of {len(names)} sheets were read.")
        names = names[:MAX_SHEETS]

    for index, name in enumerate(names):
        report(progress, index / len(names), f"Reading sheet {name}")
        doc.elements.extend(_frame_to_elements(sheets[name], display_name, f"Sheet: {name}", doc.warnings))

    # if any(doc.total_chars == 0 for _ in [0]) and not doc.elements:
    if not doc.elements:
        doc.warnings.append(
            "No values found. If the cells contain formulas, open and save the file in Excel first "
            "so the calculated results are stored."
        )
    report(progress, 1.0, "Done")
    return doc


def _try_read_csv(path: Path, encoding: str) -> pd.DataFrame | None:
    """Read a CSV with a given encoding; guess the separator, then fall back to a comma."""
    for separator in (None, ","):
        try:
            return pd.read_csv(
                path,
                sep=separator,
                engine="python",
                dtype=str,
                keep_default_na=False,
                encoding=encoding,
                nrows=MAX_ROWS_PER_SHEET + 1,
                on_bad_lines="skip",
            )
        except UnicodeDecodeError:
            raise
        except pd.errors.EmptyDataError as exc:
            raise LoaderError("This CSV file is empty.") from exc
        except Exception:
            continue
    return None


def load_csv(path: Path, display_name: str, progress: ProgressFn | None = None) -> LoadedDocument:
    df: pd.DataFrame | None = None
    for encoding in CSV_ENCODINGS:
        try:
            df = _try_read_csv(path, encoding)
        except UnicodeDecodeError:
            continue
        if df is not None:
            break
    if df is None:
        raise LoaderError("This CSV file could not be read. Check that it is a normal comma-separated file.")

    doc = LoadedDocument(filename=display_name, unit_count=1, unit_name="tables")
    doc.elements.extend(_frame_to_elements(df, display_name, "CSV data", doc.warnings))
    report(progress, 1.0, "Done")
    return doc
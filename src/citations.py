"""Citation numbers: read [1] markers in answers, link them to excerpts, sanity-check figures."""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.vector_store import Hit

# Matches [1], [2, 3], [1;2]
_MARKER_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
# Matches 1,250.00 / 50,000 / 2025 / 3.14
_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
MAX_FIGURE_WARNINGS = 5


@dataclass(frozen=True)
class Citation:
    """One source excerpt behind an answer."""

    number: int
    source: str
    location: str
    kind: str
    text: str
    similarity: float
    cited: bool = True  # False = shown only because the model did not cite anything

    @property
    def label(self) -> str:
        return f"{self.source} - {self.location}"


def _numbers_in(match: re.Match) -> list[int]:
    return [int(part) for part in re.split(r"\s*[,;]\s*", match.group(1))]


def clean_markers(answer: str, max_number: int) -> str:
    """Remove citation markers that point to excerpts which do not exist.

    '[7]' is dropped when only 3 excerpts exist; '[2, 9]' becomes '[2]'.
    Numbers of 100 or more are left alone (they are probably years like [2024]).
    """
    changed = False

    def fix(match: re.Match) -> str:
        nonlocal changed
        numbers = _numbers_in(match)
        if any(n >= 100 for n in numbers):
            return match.group(0)
        valid = [n for n in dict.fromkeys(numbers) if 1 <= n <= max_number]
        fixed = "".join(f"[{n}]" for n in valid)
        if fixed != match.group(0):
            changed = True
        return fixed

    result = _MARKER_RE.sub(fix, answer)
    if changed:
        result = re.sub(r"[ \t]{2,}", " ", result)
        result = re.sub(r" +([.,;:!?])", r"\1", result)
    return result


def strip_markers(text: str) -> str:
    """Remove all [n] markers (used for chat history and checks)."""

    def drop(match: re.Match) -> str:
        return match.group(0) if any(n >= 100 for n in _numbers_in(match)) else ""

    result = _MARKER_RE.sub(drop, text)
    result = re.sub(r"[ \t]{2,}", " ", result)
    return re.sub(r" +([.,;:!?])", r"\1", result).strip()


def cited_numbers(answer: str, max_number: int) -> list[int]:
    """Valid excerpt numbers used in the answer, in order of first use."""
    found: list[int] = []
    for match in _MARKER_RE.finditer(answer):
        for number in _numbers_in(match):
            if 1 <= number <= max_number and number not in found:
                found.append(number)
    return found


def build_citations(answer: str, hits: list[Hit]) -> list[Citation]:
    """Link the answer to its excerpts. If nothing was cited, list every consulted excerpt."""
    numbers = sorted(cited_numbers(answer, len(hits)))
    cited = bool(numbers)
    chosen = numbers or list(range(1, len(hits) + 1))
    return [
        Citation(
            number=n,
            source=hits[n - 1].source,
            location=hits[n - 1].location,
            kind=hits[n - 1].kind,
            text=hits[n - 1].text,
            similarity=hits[n - 1].similarity,
            cited=cited,
        )
        for n in chosen
    ]


def unsupported_figures(answer: str, hits: list[Hit], question: str = "") -> list[str]:
    """Numbers (3+ digits) in the answer that appear nowhere in the retrieved text.

    A cheap hallucination alarm, not a proof: a number the model calculated itself
    or wrote in another format may be flagged even though it is correct.
    """
    haystack = " ".join(hit.text for hit in hits) + " " + question
    haystack = re.sub(r"(?<=\d),(?=\d{3})", "", haystack)  # 1,250 -> 1250

    flagged: list[str] = []
    for token in _NUMBER_RE.findall(strip_markers(answer)):
        normal = token.replace(",", "")
        if len(normal.split(".")[0]) < 3:
            continue  # short numbers (list numbers, "6 members") are too ambiguous to check
        if normal not in haystack and token not in flagged:
            flagged.append(token)
    return flagged[:MAX_FIGURE_WARNINGS]
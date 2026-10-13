"""Conversation memory: what was asked and answered in this session."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.citations import Citation, strip_markers

MAX_STORED_TURNS = 50


def shorten(text: str, limit: int) -> str:
    """Cut text to `limit` characters, adding an ellipsis if it was cut."""
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


@dataclass
class Turn:
    """One question and its answer."""

    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    mode: str = "factual"        # factual | summary | compare | chat
    found: bool = True
    note: str = ""
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0
    created: datetime = field(default_factory=datetime.now)


class ConversationMemory:
    """Holds the chat history of ONE session."""

    def __init__(self) -> None:
        self._turns: list[Turn] = []

    @property
    def turns(self) -> list[Turn]:
        return list(self._turns)

    def __len__(self) -> int:
        return len(self._turns)

    def add(self, turn: Turn) -> None:
        self._turns.append(turn)
        del self._turns[:-MAX_STORED_TURNS]  # keep only the newest turns

    def clear(self) -> None:
        self._turns.clear()

    def recent_pairs(self, max_turns: int = 3, answer_chars: int = 500) -> list[tuple[str, str]]:
        """(question, short answer) pairs for the prompt. Citation tags are removed because
        their numbers only made sense for the excerpts of that earlier question."""
        pairs: list[tuple[str, str]] = []
        for turn in reversed(self._turns):
            if turn.mode == "chat":
                continue
            pairs.append((turn.question, shorten(strip_markers(turn.answer), answer_chars)))
            if len(pairs) >= max_turns:
                break
        pairs.reverse()
        return pairs

    def to_markdown(self, title: str = "Document chat") -> str:
        """The whole conversation as Markdown (for the download button)."""
        lines = [f"# {title}", "", f"_Exported {datetime.now():%Y-%m-%d %H:%M}_", ""]
        for number, turn in enumerate(self._turns, start=1):
            lines += [f"## Question {number}", "", turn.question, "", "**Answer**", "", turn.answer, ""]
            if turn.note:
                lines += [f"_{turn.note}_", ""]
            for warning in turn.warnings:
                lines += [f"> Warning: {warning}", ""]
            if turn.citations:
                lines.append("**Sources**")
                for citation in turn.citations:
                    suffix = "" if citation.cited else " (consulted, not cited)"
                    lines.append(f"- [{citation.number}] {citation.label}{suffix}")
                lines.append("")
        return "\n".join(lines).strip() + "\n"
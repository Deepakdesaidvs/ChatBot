"""Helpers that make text safe to show in the browser."""
from __future__ import annotations

import re

# ![alt](url) and ![alt][ref]: images load automatically, which could leak data to a stranger's server.
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)|!\[[^\]]*\]\[[^\]]*\]")
# [text](url "title") -> we show the address openly
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)[^)]*\)")
_MD_SPECIAL_RE = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$:])")

_MODE_LABELS = {"summary": "Summary (sampled)", "compare": "Comparison", "factual": "Answer"}


def safe_markdown(text: str) -> str:
    """Prepare model or user text for st.markdown: remove images, make links visible."""
    text = _IMAGE_RE.sub("[image removed]", text)
    return _LINK_RE.sub(lambda match: f"{match.group(1)} ({match.group(2)})", text)


def escape_md(text: str) -> str:
    """Show text literally: stops file or sheet names from being read as Markdown."""
    return _MD_SPECIAL_RE.sub(r"\\\1", text)


def mode_label(mode: str) -> str:
    return _MODE_LABELS.get(mode, "")
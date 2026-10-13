"""The question-answering engine: retrieval + prompt + local LLM + citations + memory."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator

from src.citations import (
    Citation,
    build_citations,
    clean_markers,
    strip_markers,
    unsupported_figures,
)
from src.embeddings import EmbeddingError
from src.llm import LLMError, chat, stream_chat
from src.memory import ConversationMemory, Turn, shorten
from src.prompts import (
    NO_DOCUMENTS_MESSAGE,
    NOT_FOUND_MESSAGE,
    REWRITE_SYSTEM_PROMPT,
    SMALL_TALK_MESSAGE,
    build_messages,
)
from src.retrieval import Retriever, detect_mode
from src.utils import get_logger
from src.vector_store import Hit, SessionStore, VectorStoreError

log = get_logger(__name__)

# The UI should catch these and show str(error) to the user.
ENGINE_ERRORS = (LLMError, EmbeddingError, VectorStoreError)

MAX_QUESTION_CHARS = 1500
REWRITE_FOLLOWUPS = True  # set False to skip the extra model call for follow-up questions

_SMALL_TALK_RE = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|thx|ok|okay|bye|good (morning|afternoon|evening))\W*$",
    re.IGNORECASE,
)
_FOLLOWUP_RE = re.compile(
    r"^\s*(and|also|then|so|but|what about|how about|why)\b"
    r"|\b(it|its|they|them|their|that|those|these|he|she|his|her|former|latter|same|"
    r"previous|above|earlier|first one|second one|third one|this one)\b",
    re.IGNORECASE,
)
_SOURCE_NAME_INTENT_RE = re.compile(r"\b(?:file\s*name|filename|name|called)\b", re.IGNORECASE)
_IMAGE_SOURCE_RE = re.compile(r"\b(?:image|picture|photo|screenshot)\b", re.IGNORECASE)
_PDF_SOURCE_RE = re.compile(r"\bpdf\b", re.IGNORECASE)
_FILE_SOURCE_RE = re.compile(
    r"\b(?:file|document|report|presentation|spreadsheet)\b", re.IGNORECASE
)
_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp"})


def small_talk_reply(question: str) -> str | None:
    """A friendly canned reply for 'hi' / 'thanks' (no model call needed)."""
    return SMALL_TALK_MESSAGE if _SMALL_TALK_RE.match(question) else None


def looks_like_followup(question: str, has_history: bool) -> bool:
    """Does this question probably depend on the earlier conversation?"""
    if not has_history:
        return False
    if detect_mode(question, 2) != "factual":  # summaries and comparisons stand on their own
        return False
    return len(question.split()) <= 2 or bool(_FOLLOWUP_RE.search(question))


def _source_name_target(question: str) -> str | None:
    """Return which uploaded source names a question requests, if any."""
    if not _SOURCE_NAME_INTENT_RE.search(question):
        return None
    if _IMAGE_SOURCE_RE.search(question):
        return "image"
    if _PDF_SOURCE_RE.search(question):
        return "pdf"
    if _FILE_SOURCE_RE.search(question):
        return "file"
    return None


def _is_refusal(text: str) -> bool:
    """Did the model just say 'not found' (rather than answer part of the question)?"""
    plain = strip_markers(text)
    return NOT_FOUND_MESSAGE.lower().rstrip(".") in plain.lower() and len(plain) < len(NOT_FOUND_MESSAGE) + 40


@dataclass
class Answer:
    """The finished result of one question."""

    text: str
    found: bool = True
    mode: str = "factual"
    citations: list[Citation] = field(default_factory=list)
    note: str = ""
    warnings: list[str] = field(default_factory=list)
    retrieval_query: str = ""
    seconds: float = 0.0


class AnswerStream:
    """Iterate to receive the answer text piece by piece. When finished, `.answer` is the
    complete Answer. Show `.answer.text` at the end: it has invalid citation tags removed."""

    def __init__(self, producer: Callable[["AnswerStream"], Iterator[str]]) -> None:
        self.answer: Answer | None = None
        self._iterator = producer(self)

    def __iter__(self) -> Iterator[str]:
        return self._iterator

    def consume(self) -> Answer:
        for _ in self:
            pass
        assert self.answer is not None
        return self.answer


class ChatEngine:
    """Answers questions about ONE session's documents."""

    def __init__(self, store: SessionStore, memory: ConversationMemory | None = None) -> None:
        self.store = store
        self.memory = memory or ConversationMemory()
        self.retriever = Retriever(store)

    # ---- public API ------------------------------------------------------
    def ask_stream(self, question: str) -> AnswerStream:
        return AnswerStream(lambda stream: self._run(stream, question))

    def ask(self, question: str) -> Answer:
        return self.ask_stream(question).consume()

    # ---- internals -------------------------------------------------------
    def _canned(
        self,
        stream: AnswerStream,
        question: str,
        text: str,
        *,
        found: bool,
        mode: str,
        started: float,
        note: str = "",
        remember: bool = True,
    ) -> Iterator[str]:
        stream.answer = Answer(text=text, found=found, mode=mode, note=note,
                               seconds=time.perf_counter() - started)
        if remember:
            self.memory.add(Turn(question=question, answer=text, mode=mode, found=found, note=note))
        yield text

    def _standalone_question(self, question: str, history: list[tuple[str, str]]) -> str:
        """Turn a follow-up like 'And its budget?' into a self-contained question."""
        if not REWRITE_FOLLOWUPS or not looks_like_followup(question, bool(history)):
            return question
        transcript = "\n".join(
            f"User: {old_q}\nAssistant: {shorten(old_a, 300)}" for old_q, old_a in history[-2:]
        )
        messages = [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Conversation:\n{transcript}\nLast question: {question}\nStandalone question:",
            },
        ]
        try:
            rewritten = chat(messages, temperature=0.0, max_tokens=80)
        except LLMError:
            log.warning("Follow-up rewrite failed; using a simple fallback", exc_info=True)
            rewritten = ""
        lines = [line.strip() for line in rewritten.splitlines() if line.strip()]
        cleaned = re.sub(r"^(standalone question|question)\s*:\s*", "", lines[0], flags=re.I).strip(" \"'") if lines else ""
        if 5 <= len(cleaned) <= 300:
            return cleaned
        return f"{history[-1][0]} {question}"  # fallback: previous question + new one

    def _run(self, stream: AnswerStream, raw_question: str) -> Iterator[str]:
        started = time.perf_counter()
        question = raw_question.strip()[:MAX_QUESTION_CHARS]

        if not question:
            yield from self._canned(stream, question, "Please type a question.", found=False,
                                    mode="chat", started=started, remember=False)
            return
        reply = small_talk_reply(question)
        if reply:
            yield from self._canned(stream, question, reply, found=True, mode="chat", started=started)
            return
        if self.store.count() == 0:
            yield from self._canned(stream, question, NO_DOCUMENTS_MESSAGE, found=False,
                                    mode="chat", started=started)
            return

        source_target = _source_name_target(question)
        if source_target:
            source_hits: dict[str, Hit] = {}
            for hit in self.store.all_chunks():
                source = hit.source
                if source_target == "image" and not source.lower().endswith(tuple(_IMAGE_EXTENSIONS)):
                    continue
                if source_target == "pdf" and not source.lower().endswith(".pdf"):
                    continue
                source_hits.setdefault(source, hit)

            if not source_hits:
                noun = "image files" if source_target == "image" else "PDF files"
                text = f"There are no uploaded {noun}."
                citations: list[Citation] = []
            else:
                matches = list(source_hits.values())
                noun = "image file" if source_target == "image" else (
                    "PDF file" if source_target == "pdf" else "uploaded file"
                )
                if len(matches) == 1:
                    text = f"The {noun} name is {matches[0].source} [1]."
                else:
                    text = f"The {noun} names are " + ", ".join(
                        f"{hit.source} [{index}]" for index, hit in enumerate(matches, start=1)
                    ) + "."
                citations = build_citations(text, matches)

            answer = Answer(
                text=text,
                found=True,
                mode="intro",
                citations=citations,
                seconds=time.perf_counter() - started,
            )
            stream.answer = answer
            self.memory.add(Turn(
                question=question,
                answer=answer.text,
                citations=answer.citations,
                mode=answer.mode,
                found=answer.found,
                seconds=answer.seconds,
            ))
            log.info("Answered uploaded source-name question: sources=%d %.1fs",
                     len(source_hits), answer.seconds)
            yield text
            return

        history = self.memory.recent_pairs()
        retrieval_query = self._standalone_question(question, history)
        result = self.retriever.retrieve(retrieval_query)

        if not result.found:  # nothing relevant: answer without spending CPU on the model
            yield from self._canned(stream, question, NOT_FOUND_MESSAGE, found=False,
                                    mode=result.mode, started=started, note=result.note)
            return

        prompt_question = question
        if retrieval_query != question:
            prompt_question = f"{question}\n(In context, this means: {retrieval_query})"
        messages, used_hits = build_messages(prompt_question, result.hits, result.mode, history, result.note)

        parts: list[str] = []
        for piece in stream_chat(messages):  # LLMError propagates to the UI
            parts.append(piece)
            yield piece

        raw = "".join(parts).strip()
        if not raw:
            raise LLMError("The model returned an empty answer. Please try again.")
        text = clean_markers(raw, len(used_hits))

        warnings: list[str] = []
        citations: list[Citation] = []
        found = True
        if _is_refusal(text):
            text, found = NOT_FOUND_MESSAGE, False
        else:
            citations = build_citations(text, used_hits)
            if citations and not citations[0].cited:
                warnings.append("The model did not cite specific excerpts, so all consulted excerpts are listed.")
            odd = unsupported_figures(text, used_hits, question)
            if odd:
                warnings.append(
                    "These figures were not found in the retrieved text: "
                    + ", ".join(odd) + ". Please verify them in the source."
                )

        stream.answer = Answer(
            text=text, found=found, mode=result.mode, citations=citations, note=result.note,
            warnings=warnings, retrieval_query=retrieval_query, seconds=time.perf_counter() - started,
        )
        self.memory.add(Turn(
            question=question, answer=text, citations=citations, mode=result.mode, found=found,
            note=result.note, warnings=warnings, seconds=stream.answer.seconds,
        ))
        log.info("Answered: mode=%s found=%s citations=%d %.1fs", result.mode, found, len(citations),
                 stream.answer.seconds)
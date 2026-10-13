"""Find the chunks that answer a question: hybrid search plus special handling for
'summarize' and 'compare' questions."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field, replace

from src.config import settings
from src.embeddings import embed_query
from src.utils import get_logger
from src.vector_store import Hit, SessionStore

log = get_logger(__name__)

RRF_K = 60                   # standard constant for Reciprocal Rank Fusion
KEYWORD_COVERAGE_MIN = 0.6   # a chunk with 60%+ of the question's key words counts as relevant
CONTEXT_FILL = 0.55          # share of the model's context window used for document excerpts
CHARS_PER_TOKEN = 3.5        # rough size of a token in characters
MIN_AVG_CHUNK_CHARS = 200

STOPWORDS = frozenset(
    """a an the and or but if of to in on at by for with from as is are was were be been being
    it its this that these those what which who whom whose when where why how do does did can could
    should would will shall may might must i you he she we they me my your our their there here
    about into over under than then so not no yes any all some each per also have has had""".split()
)

_SUMMARY_PATTERNS = [
    r"\bsummar(?:y|ise|ize|ies)\b",
    r"\boverview\b",
    r"\btl;?dr\b",
    r"\bgist\b",
    r"\bmain (?:points?|ideas?|topics?)\b",
    r"\bkey (?:points?|takeaways?|findings?)\b",
    r"\bwhat\b.{0,30}\b(?:document|file|report|presentation|paper)s?\b.{0,20}\babout\b",
]
_COMPARE_PATTERNS = [
    r"\bcompar(?:e|es|ed|ing|ison)\b",
    r"\bdifferen(?:ce|ces)\b",
    r"\bdiffer(?:s)?\b",
    r"\bversus\b",
    r"\bvs\.?(?=\s)",
    r"\bsimilarit(?:y|ies)\b",
    r"\bcontrast\b",
]
# Questions about the document itself (title, author) cannot be matched by meaning,
# because the answer is simply "what is written at the top".
_INTRO_PATTERNS = [
    r"\b(?:title|titile|tittle|heading|headline|author|authors|name)\b.{0,25}"
    r"\b(?:document|file|report|paper|presentation|pdf|slides?|doc)\b",
    r"\bwho\s+(?:wrote|authored|created|prepared)\s+(?:this|the|it)\b",
    r"\bwhat\s+is\s+(?:this|the)\s+(?:document|file|report|paper|presentation)\s+called\b",
]
_SUMMARY_RE = re.compile("|".join(_SUMMARY_PATTERNS), re.IGNORECASE)
_COMPARE_RE = re.compile("|".join(_COMPARE_PATTERNS), re.IGNORECASE)
_INTRO_RE = re.compile("|".join(_INTRO_PATTERNS), re.IGNORECASE)
INTRO_CHUNKS = 2  # how many chunks from the start of each document an 'intro' question reads
_TOKEN_RE = re.compile(r"\w+(?:[-./,:]\w+)*", re.UNICODE)


# ---------------------------------------------------------------- helpers
def _normalise(token: str) -> str:
    """Very light plural stripping so 'budgets' matches 'budget' (English only)."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lower-case words. 'INV-2024-001' gives the whole id AND its parts."""
    tokens: list[str] = []
    for match in _TOKEN_RE.findall(text.lower()):
        tokens.append(_normalise(match))
        if re.search(r"[-./,:]", match):
            tokens.extend(_normalise(part) for part in re.split(r"[-./,:]", match) if part)
    return tokens


def distinctive_tokens(question: str) -> set[str]:
    """The words of a question that actually carry meaning."""
    return {
        token
        for token in tokenize(question)
        if token not in STOPWORDS and (len(token) >= 3 or any(ch.isdigit() for ch in token))
    }


def detect_mode(question: str, document_count: int) -> str:
    """'compare' | 'summary' | 'intro' | 'factual'"""
    if document_count >= 2 and _COMPARE_RE.search(question):
        return "compare"
    if _SUMMARY_RE.search(question):
        return "summary"
    if _INTRO_RE.search(question):
        return "intro"
    return "factual"


def mentioned_sources(question: str, sources: list[str]) -> list[str]:
    """Files the user named in the question (full name, or name without extension)."""
    lowered = question.lower()
    found: list[str] = []
    for source in sources:
        name = source.lower()
        stem = name.rsplit(".", 1)[0]
        if name in lowered or (
            len(stem) >= 4 and re.search(rf"(?<!\w){re.escape(stem)}(?!\w)", lowered)
        ):
            found.append(source)
    return found


def even_sample(items: list[Hit], k: int) -> list[Hit]:
    """Pick k items spread evenly across the list (always including the first and last)."""
    if k >= len(items):
        return list(items)
    if k <= 1:
        return items[:1]
    positions = sorted({round(i * (len(items) - 1) / (k - 1)) for i in range(k)})
    return [items[p] for p in positions]


def _cut_tail(hits: list[Hit], budget: int) -> list[Hit]:
    """Keep the best-ranked hits that fit in the character budget."""
    kept: list[Hit] = []
    total = 0
    for hit in hits:
        if kept and total + len(hit.text) > budget:
            break
        kept.append(hit)
        total += len(hit.text)
    return kept


def _trim_even(hits: list[Hit], budget: int) -> list[Hit]:
    """Shrink to the budget by removing from whichever file has the most chunks."""
    hits = list(hits)
    while len(hits) > 1 and sum(len(h.text) for h in hits) > budget:
        counts = Counter(h.source for h in hits)
        biggest = counts.most_common(1)[0][0]
        for i in range(len(hits) - 1, -1, -1):
            if hits[i].source == biggest:
                del hits[i]
                break
    return hits


# ---------------------------------------------------------------- keyword search (BM25)
class _KeywordIndex:
    """Classic BM25 keyword scoring over all chunks of a session."""

    K1 = 1.5
    B = 0.75

    def __init__(self, chunks: list[Hit]) -> None:
        self.chunks = chunks
        self.position = {chunk.id: i for i, chunk in enumerate(chunks)}
        self.text_tokens = [set(tokenize(chunk.text)) for chunk in chunks]
        self.term_counts = [
            Counter(tokenize(f"{chunk.source} {chunk.location} {chunk.text}")) for chunk in chunks
        ]
        self.lengths = [sum(counts.values()) for counts in self.term_counts]
        self.avg_length = (sum(self.lengths) / len(self.lengths)) or 1.0
        self.doc_freq: Counter[str] = Counter()
        for counts in self.term_counts:
            self.doc_freq.update(counts.keys())

    def _idf(self, term: str) -> float:
        containing = self.doc_freq.get(term, 0)
        return math.log(1 + (len(self.chunks) - containing + 0.5) / (containing + 0.5))

    def search(self, query: str, scope: set[str] | None, limit: int) -> list[Hit]:
        terms = distinctive_tokens(query)
        if not terms:
            return []
        scored: list[tuple[float, int]] = []
        for i, counts in enumerate(self.term_counts):
            if scope is not None and self.chunks[i].source not in scope:
                continue
            norm = self.K1 * (1 - self.B + self.B * self.lengths[i] / self.avg_length)
            score = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if tf:
                    score += self._idf(term) * tf * (self.K1 + 1) / (tf + norm)
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [self.chunks[i] for _, i in scored[:limit]]

    def coverage(self, chunk_id: str, wanted: set[str]) -> float:
        """Share of the wanted words that appear in this chunk's text."""
        if not wanted:
            return 0.0
        return len(wanted & self.text_tokens[self.position[chunk_id]]) / len(wanted)


# ---------------------------------------------------------------- the retriever
@dataclass
class RetrievalResult:
    hits: list[Hit] = field(default_factory=list)
    found: bool = False            # False = nothing relevant, answer "not found" without the LLM
    mode: str = "factual"          # factual | summary | compare
    best_similarity: float = 0.0   # best meaning-score seen (useful for tuning MIN_RELEVANCE)
    sources: list[str] = field(default_factory=list)
    note: str = ""                 # e.g. "based on a sample of the document"


class Retriever:
    """Retrieval for one session's documents."""

    def __init__(self, store: SessionStore) -> None:
        self.store = store
        self.top_k = settings.top_k
        self._index: _KeywordIndex | None = None
        self._index_version = -1

    def _keyword_index(self) -> _KeywordIndex | None:
        if self._index_version != self.store.version:
            chunks = self.store.all_chunks()
            self._index = _KeywordIndex(chunks) if chunks else None
            self._index_version = self.store.version
        return self._index

    # -- normal questions: hybrid search --------------------------------
    def _factual(
        self, question: str, query_vector: list[float], scope: list[str] | None, limit: int
    ) -> tuple[list[Hit], float]:
        """Meaning search + keyword search, merged with RRF, filtered by relevance."""
        pool = max(self.top_k * 3, 12)
        vector_hits = self.store.query(query_vector, pool, sources=scope)
        index = self._keyword_index()
        keyword_hits = index.search(question, set(scope) if scope else None, pool) if index else []
        best = max((hit.similarity for hit in vector_hits), default=0.0)

        scores: dict[str, float] = {}
        by_id: dict[str, Hit] = {}
        for hits in (vector_hits, keyword_hits):
            for rank, hit in enumerate(hits):
                scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (RRF_K + rank + 1)
                by_id.setdefault(hit.id, hit)  # vector hit is first, so it keeps its similarity

        wanted = distinctive_tokens(question)
        relevant: list[Hit] = []
        for hit_id in sorted(by_id, key=lambda i: scores[i], reverse=True):
            coverage = index.coverage(hit_id, wanted) if index else 0.0
            hit = replace(by_id[hit_id], keyword_coverage=coverage)
            if hit.similarity >= settings.min_relevance or coverage >= KEYWORD_COVERAGE_MIN:
                relevant.append(hit)
                if len(relevant) >= limit:
                    break
        return relevant, best

    # -- compare: make sure every document is represented ----------------
    def _compare(
        self, question: str, query_vector: list[float], scope: list[str], budget: int
    ) -> tuple[list[Hit], float, str]:
        chunks = [c for c in self.store.all_chunks() if c.source in scope]
        average = max(MIN_AVG_CHUNK_CHARS, sum(len(c.text) for c in chunks) / max(len(chunks), 1))
        per_source = max(1, int(budget // average) // len(scope))

        hits: list[Hit] = []
        best = 0.0
        sampled: list[str] = []
        for source in scope:
            relevant, source_best = self._factual(question, query_vector, [source], per_source)
            best = max(best, source_best)
            if not relevant:  # e.g. "compare the two documents" has no topic words
                relevant = even_sample([c for c in chunks if c.source == source], per_source)
                sampled.append(source)
            hits.extend(relevant)
        hits.sort(key=lambda h: (h.source, h.chunk_index))
        hits = _trim_even(hits, budget)
        note = "Documents compared: " + ", ".join(scope) + "."
        if sampled:
            note += " For " + ", ".join(sampled) + " a sample of sections was used."
        return hits, best, note

    # -- summaries: sample the whole document ----------------------------
    def _summary(
        self, question: str, query_vector: list[float], scope: list[str], budget: int
    ) -> tuple[list[Hit], float, str]:
        chunks = [c for c in self.store.all_chunks() if c.source in scope]
        average = max(MIN_AVG_CHUNK_CHARS, sum(len(c.text) for c in chunks) / max(len(chunks), 1))
        slots = max(len(scope), int(budget // average))

        relevant, best = self._factual(question, query_vector, scope, min(3, self.top_k))
        per_source = max(1, (slots - len(relevant)) // len(scope))

        chosen: dict[str, Hit] = {}
        for source in scope:
            for hit in even_sample([c for c in chunks if c.source == source], per_source):
                chosen[hit.id] = hit
        for hit in relevant:
            chosen[hit.id] = hit  # the relevant version carries its similarity score

        hits = sorted(chosen.values(), key=lambda h: (h.source, h.chunk_index))
        hits = _trim_even(hits, budget)
        note = ""
        if len(hits) < len(chunks):
            note = (
                f"Based on {len(hits)} of {len(chunks)} text sections spread across "
                f"{', '.join(scope)}. Very long documents are sampled, not read in full."
            )
        return hits, best, note

    # -- title / author questions: read the start of each document ---------
    def _intro(self, scope: list[str], budget: int) -> tuple[list[Hit], float, str]:
        chunks = self.store.all_chunks()
        hits: list[Hit] = []
        for source in scope:
            hits.extend([c for c in chunks if c.source == source][:INTRO_CHUNKS])
        hits = _trim_even(hits, budget)
        return hits, 0.0, "Based on the beginning of " + ", ".join(scope) + "."
    
    # -- public entry point ------------------------------------------------
    def retrieve(self, question: str) -> RetrievalResult:
        """Find the chunks to show the language model for this question."""
        question = question.strip()
        chunks = self.store.all_chunks()
        if not chunks:
            return RetrievalResult(note="No documents are loaded.")
        if not question:
            return RetrievalResult()

        sources = sorted({c.source for c in chunks})
        named = mentioned_sources(question, sources)
        scope = named or sources
        mode = detect_mode(question, len(scope))
        query_vector = embed_query(question)
        budget = int(settings.num_ctx * CONTEXT_FILL * CHARS_PER_TOKEN)

        note = ""
        if mode == "compare":
            hits, best, note = self._compare(question, query_vector, scope, budget)
        elif mode == "summary":
            hits, best, note = self._summary(question, query_vector, scope, budget)
        elif mode == "intro":
            hits, best, note = self._intro(scope, budget)
        else:
            hits, best = self._factual(question, query_vector, named or None, self.top_k)
            if not hits and named:  # the named file had nothing: look everywhere
                hits, best = self._factual(question, query_vector, None, self.top_k)
            hits = _cut_tail(hits, budget)

        log.info("Retrieval: mode=%s hits=%d best_similarity=%.2f", mode, len(hits), best)
        return RetrievalResult(
            hits=hits,
            found=bool(hits),
            mode=mode,
            best_similarity=best,
            sources=sorted({h.source for h in hits}),
            note=note,
        )
"""Phase 5 check: prompts, citations, memory (offline) + a live question/answer run."""
from __future__ import annotations

import sys
import time

from src.chunking import chunk_document
from src.citations import build_citations, cited_numbers, clean_markers, strip_markers, unsupported_figures
from src.config import settings
from src.embeddings import embed_documents
from src.memory import ConversationMemory, Turn
from src.prompts import NOT_FOUND_MESSAGE, build_messages, estimate_tokens
from src.qa import ChatEngine, looks_like_followup, small_talk_reply
from src.utils import check_ollama, new_session_id, setup_logging
from src.vector_store import Hit, SessionStore
from tests.test_phase4 import FALCON, INVOICE, make_doc

results: list[bool] = []


def check(label: str, ok: bool) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    results.append(ok)


def hit(n: int, text: str, source: str = "contract.pdf", location: str = "Page 2") -> Hit:
    return Hit(id=f"x-{n}", text=text, source=source, location=location, kind="text", chunk_index=n)


def offline_tests() -> None:
    print("\n--- 1. Prompts")
    hits = [hit(0, "The notice period is 30 days."), hit(1, "Fees are 100 EUR.", location="Page 3")]
    messages, used = build_messages("How long is the notice period?", hits, "factual", [("Hi?", "Hello")])
    check("system prompt comes first", messages[0]["role"] == "system")
    check("history is included", messages[1]["content"] == "Hi?")
    last = messages[-1]["content"]
    check("excerpts are numbered with source and location", "[1] contract.pdf | Page 2" in last and "[2] contract.pdf | Page 3" in last)
    long_history = [(f"Question {i} " * 20, "Answer text " * 60) for i in range(40)]
    messages, used = build_messages("How long?", hits, "factual", long_history)
    limit = settings.num_ctx - settings.max_answer_tokens
    check("oversized history is trimmed to fit the context window", estimate_tokens(messages) <= limit)

    print("\n--- 2. Citations")
    check("invalid markers are removed",
          clean_markers("A [1] B [7] C [2, 9] D [2024]", 3) == "A [1] B C [2] D [2024]")
    check("cited numbers found in order", cited_numbers("Yes [2] and [1][2].", 3) == [2, 1])
    check("strip_markers removes tags", strip_markers("The total is 5 [1].") == "The total is 5.")
    three = [hit(0, "a"), hit(1, "b"), hit(2, "c")]
    only = build_citations("Fact. [2]", three)
    check("only the cited excerpt is listed", [c.number for c in only] == [2] and only[0].cited)
    none = build_citations("Fact without a tag.", three)
    check("no citation tags -> all excerpts listed as 'consulted'", len(none) == 3 and not none[0].cited)
    ctx = [hit(0, "Total amount due: 1,250.00 USD. Invoice INV-2024-001.")]
    check("correct figures raise no warning",
          unsupported_figures("The total is 1,250.00 USD for INV-2024-001. [1]", ctx) == [])
    check("invented figure is flagged", unsupported_figures("The total is 9,999.00 USD. [1]", ctx) == ["9,999.00"])

    print("\n--- 3. Memory and follow-ups")
    memory = ConversationMemory()
    for i in range(4):
        memory.add(Turn(question=f"Q{i}", answer=f"Answer {i} [1]"))
    pairs = memory.recent_pairs(max_turns=2)
    check("recent_pairs returns the newest turns, tags removed", pairs == [("Q2", "Answer 2"), ("Q3", "Answer 3")])
    check("markdown export contains the questions", "## Question 1" in memory.to_markdown())
    memory.clear()
    check("clear empties the memory", len(memory) == 0)
    check("'And the due date?' is a follow-up", looks_like_followup("And the due date?", True))
    check("'What about it?' is a follow-up", looks_like_followup("What about it?", True))
    check("a full question is not a follow-up", not looks_like_followup("What is the budget of Project Falcon?", True))
    check("no history -> never a follow-up", not looks_like_followup("And the due date?", False))
    check("'summarize' is never rewritten", not looks_like_followup("Summarize it", True))
    check("'thanks!' gets a canned reply", small_talk_reply("thanks!") is not None)
    check("real questions are not small talk", small_talk_reply("What is the budget?") is None)

    print("\n--- 4. Uploaded source names")

    class SourceStore:
        def __init__(self) -> None:
            self.chunks = [hit(0, "OCR text from the image", source="Screenshot (56).png")]

        def count(self) -> int:
            return len(self.chunks)

        def all_chunks(self):
            return self.chunks

    answer = ChatEngine(SourceStore()).ask("what is the image name")
    check("image-name question returns the uploaded filename without an LLM call",
          answer.found and "Screenshot (56).png" in answer.text and
          len(answer.citations) == 1 and answer.citations[0].source == "Screenshot (56).png")


def live_tests() -> None:
    print("\n--- 4. Live questions (slow: expect 1-2 minutes per model answer on a CPU)")
    status = check_ollama([settings.llm_model, settings.embed_model])
    if not status["running"] or status["missing"]:
        print("  Skipped:", status["error"] or f"missing model(s): {status['missing']}")
        return

    store = SessionStore(new_session_id())
    try:
        for name, text in (("falcon_plan.txt", FALCON), ("invoice.txt", INVOICE)):
            chunks = chunk_document(make_doc(name, text), size=300, overlap=50)
            store.add_document(name, chunks, embed_documents([c.embed_text for c in chunks]))
        engine = ChatEngine(store)

        def ask(question: str):
            print(f"\n  Q: {question}\n  A: ", end="", flush=True)
            started = time.perf_counter()
            stream = engine.ask_stream(question)
            for piece in stream:
                print(piece, end="", flush=True)
            answer = stream.answer
            print(f"\n     [{time.perf_counter() - started:.0f}s | mode={answer.mode} | found={answer.found}]")
            for c in answer.citations:
                print(f"     source [{c.number}] {c.label}" + ("" if c.cited else " (consulted, not cited)"))
            for warning in answer.warnings:
                print("     WARNING:", warning)
            return answer

        a = ask("Who is the owner of Project Falcon and what is the budget?")
        check("(model-dependent) answer names Maria Lopez and 50,000", "Maria Lopez" in a.text and "50,000" in a.text)
        check("answer cites falcon_plan.txt", any(c.source == "falcon_plan.txt" for c in a.citations))

        a = ask("And when is the go-live date?")
        check("(model-dependent) follow-up understood: 30 September 2025", "30 September 2025" in a.text)
        print(f"     (search used the question: {a.retrieval_query!r})")

        started = time.perf_counter()
        a = ask("What is the capital of France?")
        check("off-topic question -> exact 'not found' message", a.text == NOT_FOUND_MESSAGE and not a.found)
        check("off-topic answer is instant (no model call)", time.perf_counter() - started < 5)

        a = ask("Summarize invoice.txt")
        check("summary mode used and an answer was written", a.mode == "summary" and len(a.text) > 50)
        check("memory holds all four questions", len(engine.memory) == 4)
    finally:
        store.delete()


def main() -> None:
    setup_logging()
    offline_tests()
    if "--offline" not in sys.argv:
        live_tests()
    print(f"\n{sum(results)} of {len(results)} checks passed.")


if __name__ == "__main__":
    main()
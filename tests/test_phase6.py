"""Phase 6 check: display helpers, ingestion, exports and a start-up test of the Streamlit app."""
from __future__ import annotations

import shutil

from streamlit.testing.v1 import AppTest

from src.citations import Citation
from src.config import settings
from src.display import escape_md, safe_markdown
from src.exports import answer_markdown, conversation_docx, conversation_markdown, conversation_text
from src.ingest import UploadRejected, ingest_file, safe_filename
from src.memory import ConversationMemory, Turn
from src.utils import check_ollama, new_session_id, session_dir, setup_logging
from src.vector_store import SessionStore

results: list[bool] = []


def check(label: str, ok: bool) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    results.append(ok)


def rejected(name: str, data: bytes, sid: str, store: SessionStore) -> str | None:
    try:
        ingest_file(store, sid, name, data, {})
    except UploadRejected as exc:
        return str(exc)
    return None


def main() -> None:
    setup_logging()

    print("\n--- 1. Display helpers")
    cleaned = safe_markdown("see ![x](http://evil.test/a.png?d=secret) now")
    check("markdown images are removed", "evil.test" not in cleaned and "[image removed]" in cleaned)
    check("links show their address", safe_markdown("[click](http://a.test)") == "click (http://a.test)")
    check("escape_md shows text literally", escape_md("a_b*c") == "a\\_b\\*c")

    print("\n--- 2. File names and upload rules")
    check("path tricks are removed", safe_filename("../../evil<script>.PDF") == "evil_script_.pdf")
    check("Windows paths are reduced to the name", safe_filename("C:\\Users\\me\\report 2024.docx") == "report 2024.docx")

    status = check_ollama([settings.embed_model])
    if not status["running"] or status["missing"]:
        print("  Skipping upload tests: Ollama or the embedding model is not available.")
    else:
        sid = new_session_id()
        store = SessionStore(sid)
        try:
            check(".exe is rejected", rejected("virus.exe", b"MZ", sid, store) is not None)
            check("empty file is rejected", rejected("empty.txt", b"", sid, store) is not None)
            check("oversized file is rejected",
                  rejected("big.txt", b"x" * (settings.max_file_bytes + 1), sid, store) is not None)
            old = rejected("old.doc", b"abc", sid, store)
            check("old .doc gets a 'Save As' hint", old is not None and "Save As" in old)

            text = b"Project Falcon budget is 50,000 EUR. The owner is Maria Lopez."
            info = ingest_file(store, sid, "notes.txt", text, {})
            check("good file is ingested", info.chunks >= 1 and "notes.txt" in store.list_documents())
            leftovers = list((session_dir(sid) / "uploads").glob("*"))
            check("original file was deleted after reading", leftovers == [])
        finally:
            store.delete()
            shutil.rmtree(session_dir(sid), ignore_errors=True)

    print("\n--- 3. Exports")
    memory = ConversationMemory()
    citation = Citation(1, "notes.txt", "Lines 1-1", "text", "Budget is 50,000 EUR.", 0.8)
    memory.add(Turn(question="What is the budget?", answer="**50,000 EUR** [1]", citations=[citation]))
    check("markdown export has the question", "What is the budget?" in conversation_markdown(memory))
    check("text export has no Markdown bold", "**" not in conversation_text(memory))
    check("Word export is a real .docx file", conversation_docx(memory)[:2] == b"PK")
    check("single answer export lists sources", "notes.txt" in answer_markdown(memory.turns[0], 1))

    print("\n--- 4. The Streamlit app starts")
    app = AppTest.from_file("app.py", default_timeout=90)
    app.run()
    check("app runs without an error", not app.exception)
    check("chat box is present", len(app.chat_input) == 1)

    print(f"\n{sum(results)} of {len(results)} checks passed.")


if __name__ == "__main__":
    main()
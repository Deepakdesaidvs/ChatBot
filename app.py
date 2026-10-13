"""Local Document AI Chatbot: the Streamlit user interface.

Start it with:   streamlit run app.py
"""
from __future__ import annotations

import shutil
import hmac
import time

import streamlit as st

from src.config import settings
from src.display import escape_md, mode_label, safe_markdown
from src.embeddings import EmbeddingError
from src.exports import answer_markdown, conversation_docx, conversation_markdown, conversation_text
from src.ingest import DocInfo, UploadRejected, ingest_file
from src.loaders import SUPPORTED_EXTENSIONS
from src.loaders.base import LoaderError
from src.memory import Turn
from src.qa import ENGINE_ERRORS, ChatEngine
from src.utils import check_ollama, get_logger, human_size, new_session_id, setup_logging
from src.vector_store import SessionStore, VectorStoreError, purge_all_sessions

# st.set_page_config must be the first Streamlit command.
st.set_page_config(page_title="Local Document AI", page_icon="📄", layout="wide")
setup_logging()
log = get_logger("app")

CSS = """
<style>
.block-container {max-width: 980px; padding-top: 2rem;}
div[data-testid="stExpander"] {border-radius: 10px;}
</style>
"""
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


# ------------------------------------------------------------------ start-up and state
@st.cache_resource
def startup() -> bool:
    """Runs once per server start: wipe data left behind by earlier runs (privacy)."""
    removed = purge_all_sessions()
    shutil.rmtree(settings.sessions_dir, ignore_errors=True)
    log.info("Start-up cleanup: removed %d old collection(s)", removed)
    return True


def init_state() -> None:
    """Create this browser tab's private session (own ID, vector store, chat memory)."""
    ss = st.session_state
    if "session_id" in ss:
        return
    try:
        session_id = new_session_id()
        store = SessionStore(session_id)
    except VectorStoreError as exc:
        st.error(str(exc))
        st.stop()
    ss.session_id = session_id
    ss.store = store
    ss.engine = ChatEngine(store)
    ss.docs = {}               # file name -> DocInfo
    ss.uploader_key = 0        # changing it resets the upload box
    ss.report = []             # messages from the last processing run
    ss.pending_question = None # set by the example-question buttons
    ss.ollama = None


def ollama_status(force: bool = False) -> dict:
    """Is Ollama running and are the models installed? (re-checked at most every 20 seconds)"""
    ss = st.session_state
    status = ss.ollama
    if force or status is None or time.time() - status["checked"] > 20:
        if settings.llm_provider == "openai":
            configured = bool(settings.openai_api_key)
            status = {
                "running": configured,
                "installed": [],
                "missing": [] if configured else ["OPENAI_API_KEY"],
                "error": "" if configured else "Set OPENAI_API_KEY in the server environment.",
            }
        else:
            status = check_ollama()
        status["checked"] = time.time()
        ss.ollama = status
    return status


def require_app_password() -> None:
    """Gate public access when an application password is configured."""
    if not settings.app_password or st.session_state.get("authenticated", False):
        return

    st.title("Local Document AI")
    with st.form("app_password_form"):
        password = st.text_input("Application password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        if hmac.compare_digest(password.encode("utf-8"), settings.app_password.encode("utf-8")):
            st.session_state.authenticated = True
            st.rerun()
        st.error("Incorrect password.")
    st.stop()


# ------------------------------------------------------------------ actions
def process_uploads(files: list) -> None:
    """Read, chunk, embed and store every selected file, with a progress bar."""
    ss = st.session_state
    report: list[tuple[str, str]] = []
    total = len(files)

    with st.status("Processing documents...", expanded=True) as status:
        bar = st.progress(0.0, text="Starting...")
        for index, upload in enumerate(files):

            def progress(fraction: float, message: str, _index: int = index, _name: str = upload.name) -> None:
                bar.progress(min((_index + fraction) / total, 1.0), text=f"{_name}: {message}")

            try:
                info = ingest_file(ss.store, ss.session_id, upload.name, upload.getvalue(), ss.docs, progress)
            except (UploadRejected, LoaderError, EmbeddingError, VectorStoreError) as exc:
                prefix = "" if isinstance(exc, LoaderError) else f"{upload.name}: "
                report.append(("error", prefix + str(exc)))
                continue
            except Exception:
                log.exception("Unexpected error while processing an upload")
                report.append(("error", f"{upload.name}: unexpected problem while reading this file."))
                continue

            replaced = info.name in ss.docs
            ss.docs[info.name] = info
            suffix = " (replaced the earlier copy)" if replaced else ""
            report.append(("success", f"{info.name}: ready - {info.unit_count} {info.unit_name}, {info.chunks} chunks{suffix}"))
            report += [("warning", f"{info.name}: {warning}") for warning in info.warnings]

        bar.progress(1.0, text="Done")
        status.update(label="Finished", state="complete", expanded=False)

    ss.report = report
    ss.uploader_key += 1  # empties the upload box; the document list is now the source of truth
    st.rerun()


def remove_document(name: str) -> None:
    ss = st.session_state
    ss.store.remove_document(name)
    ss.docs.pop(name, None)


def remove_all_documents() -> None:
    ss = st.session_state
    ss.store.clear()
    ss.docs.clear()


def answer_question(question: str) -> None:
    """Show the question, stream the answer, then redraw the page from the saved history."""
    engine: ChatEngine = st.session_state.engine
    with st.chat_message("user"):
        st.markdown(safe_markdown(question))
    with st.chat_message("assistant"):
        stream = engine.ask_stream(question)
        iterator = iter(stream)
        try:
            with st.spinner("Searching your documents and reading the relevant parts (can take a minute on a CPU)..."):
                first = next(iterator, None)  # retrieval and prompt reading happen here

            def pieces():
                if first is not None:
                    yield first
                yield from iterator

            st.write_stream(pieces())
        except ENGINE_ERRORS as exc:
            st.error(str(exc))
            return
        except Exception:
            log.exception("Unexpected error while answering")
            st.error("Something went wrong while answering. Please try again; details are in data/logs/app.log.")
            return
    st.rerun()


# ------------------------------------------------------------------ sidebar
def render_sidebar(status: dict, ready: bool) -> None:
    ss = st.session_state
    with st.sidebar:
        st.title("📄 Local Document AI")
        if settings.llm_provider == "openai":
            st.caption("Answers use OpenAI. Extracted document text and questions are sent to its API.")
        else:
            st.caption("Runs on this computer. Your files are never sent to an online AI service.")

        if settings.llm_provider == "openai" and status["missing"]:
            st.error("Set OPENAI_API_KEY in the server environment to enable answers.")
        elif not status["running"]:
            st.error("Ollama is not running. Start it from the Start menu (llama icon in the system tray).")
        elif settings.llm_provider == "ollama" and status["missing"]:
            st.error("A model is missing. Run this in PowerShell, then re-check:")
            st.code("\n".join(f"ollama pull {name}" for name in status["missing"]), language="powershell")
        elif settings.llm_provider == "openai":
            st.success(f"Ready · OpenAI model: {settings.openai_chat_model}")
        else:
            st.success(f"Ready · model: {settings.llm_model}")
        if not ready and st.button("Re-check Ollama", use_container_width=True):
            ollama_status(force=True)
            st.rerun()

        st.subheader("1. Add documents")
        files = st.file_uploader(
            "Choose files",
            type=sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS),
            accept_multiple_files=True,
            key=f"uploader_{ss.uploader_key}",
            label_visibility="collapsed",
        )
        st.caption(
            f"Up to {settings.max_files} files, {settings.max_file_mb} MB each, "
            f"{settings.max_pages} pages/slides each. Password-protected files are not supported."
        )
        if st.button("Process documents", type="primary", use_container_width=True,
                     disabled=not files or not ready):
            process_uploads(files)

        for level, text in ss.report[:12]:
            {"success": st.success, "warning": st.warning, "error": st.error}[level](escape_md(text))
        if ss.report and st.button("Dismiss messages", use_container_width=True):
            ss.report = []
            st.rerun()

        st.subheader("2. Your documents")
        if not ss.docs:
            st.caption("No documents yet.")
        for name, info in list(ss.docs.items()):
            left, right = st.columns([5, 1])
            with left:
                st.markdown(f"**{escape_md(name)}**")
                st.caption(f"{info.unit_count} {info.unit_name} · {info.chunks} chunks · {human_size(info.size_bytes)}")
                for warning in info.warnings:
                    st.caption(f"⚠️ {escape_md(warning)}")
            with right:
                if st.button("✕", key=f"remove_{name}", help=f"Remove {escape_md(name)}"):
                    remove_document(name)
                    st.rerun()
        if ss.docs and st.button("Remove all documents", use_container_width=True):
            remove_all_documents()
            st.rerun()

        st.subheader("3. Conversation")
        turns = ss.engine.memory.turns
        if st.button("Clear conversation", use_container_width=True, disabled=not turns):
            ss.engine.memory.clear()
            st.rerun()
        if turns:
            memory = ss.engine.memory
            st.download_button("Download (.docx)", conversation_docx(memory), "conversation.docx",
                               DOCX_MIME, use_container_width=True, key="dl_docx")
            st.download_button("Download (.md)", conversation_markdown(memory), "conversation.md",
                               "text/markdown", use_container_width=True, key="dl_md")
            st.download_button("Download (.txt)", conversation_text(memory), "conversation.txt",
                               "text/plain", use_container_width=True, key="dl_txt")

        with st.expander("Privacy and limits"):
            privacy = (
                "- Extracted document text and questions are sent to the OpenAI API.\n"
                if settings.llm_provider == "openai"
                else "- Everything is processed on this computer by a local model.\n"
            )
            st.markdown(
                privacy
                + "- Original files are deleted right after reading; only extracted text is kept for this session.\n"
                + "- Refreshing the page or closing the tab ends your session.\n"
                + "- Answers can contain mistakes. Open **Sources** to check the exact text.\n"
                + "- Scanned pages are read with OCR, which can misread poor scans."
            )


# ------------------------------------------------------------------ main area
def render_sources(turn: Turn) -> None:
    all_cited = all(c.cited for c in turn.citations)
    heading = f"**Sources** ({len(turn.citations)})"
    st.markdown(heading if all_cited else heading + " - excerpts consulted (the model did not cite specific ones)")
    for citation in turn.citations:
        match = f" · match {citation.similarity:.0%}" if citation.similarity > 0 else ""
        with st.expander(escape_md(f"[{citation.number}] {citation.label}") + match):
            st.text(citation.text)  # plain text: exactly what the document says


def render_turn(index: int, turn: Turn) -> None:
    with st.chat_message("user"):
        st.markdown(safe_markdown(turn.question))
        if not turn.found and turn.mode != "chat":
            st.caption("Tip: use words that appear in your document, or try asking for a summary.")
    with st.chat_message("assistant"):
        st.markdown(safe_markdown(turn.answer))
        if turn.note:
            st.caption(safe_markdown(turn.note))
        for warning in turn.warnings:
            st.warning(safe_markdown(warning), icon="⚠️")
        if turn.citations:
            render_sources(turn)
        if turn.mode != "chat":
            st.caption(" · ".join(part for part in (mode_label(turn.mode), f"{turn.seconds:.0f}s") if part))
            st.download_button("Download this answer", answer_markdown(turn, index + 1),
                               f"answer_{index + 1}.md", "text/markdown", key=f"dl_answer_{index}")


def render_suggestions() -> None:
    ss = st.session_state
    st.markdown("**Try asking:**")
    questions = ["Summarize the documents", "List all names, dates and amounts"]
    if len(ss.docs) >= 2:
        questions.append("Compare the documents")
    for column, text in zip(st.columns(len(questions)), questions):
        if column.button(text, key=f"suggest_{text}", use_container_width=True):
            ss.pending_question = text
            st.rerun()


def render_main(ready: bool) -> None:
    ss = st.session_state
    st.title("Ask your documents")
    model_name = settings.openai_chat_model if settings.llm_provider == "openai" else settings.llm_model
    provider_label = "OpenAI API" if settings.llm_provider == "openai" else "runs on this computer"
    st.caption(f"Answers come only from your uploaded files · model: {model_name} · {provider_label}")

    turns = ss.engine.memory.turns
    for index, turn in enumerate(turns):
        render_turn(index, turn)

    if not ss.docs:
        st.info("👈 Add one or more documents in the sidebar, then click **Process documents**.")
    elif not turns:
        render_suggestions()

    question = st.chat_input("Ask a question about your documents...", disabled=not ready, max_chars=1500)
    if not question and ss.pending_question:
        question, ss.pending_question = ss.pending_question, None
    if question:
        answer_question(question)


def main() -> None:
    require_app_password()
    startup()
    init_state()
    st.markdown(CSS, unsafe_allow_html=True)  # fixed style block, no user content
    status = ollama_status()
    ready = status["running"] and not status["missing"]
    render_sidebar(status, ready)
    render_main(ready)


main()
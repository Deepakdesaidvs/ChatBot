"""All wording the language model sees. Small models need short, strict, example-driven prompts."""
from __future__ import annotations

from src.config import settings
from src.vector_store import Hit

NOT_FOUND_MESSAGE = "I could not find the answer in the uploaded documents."
NO_DOCUMENTS_MESSAGE = "Please upload at least one document first, then ask your question."
SMALL_TALK_MESSAGE = "Hello! Ask me anything about your uploaded documents."

CHARS_PER_TOKEN_SAFE = 3.2   # pessimistic, so we never overflow the context window
PROMPT_MARGIN_TOKENS = 100

SYSTEM_PROMPT = f"""You are a careful document assistant. You answer questions using ONLY the numbered excerpts given in the user's message. You have no other source of knowledge.

Rules:
1. Use only facts that are written in the excerpts. Never use outside knowledge. Never guess. Never fill gaps.
2. After every sentence or bullet that states a fact, add the number of the excerpt it came from, like [1] or [2][3].
3. If the excerpts do not contain the answer, reply with exactly this sentence and nothing else: "{NOT_FOUND_MESSAGE}"
4. If the excerpts answer only part of the question, answer that part and say what is missing.
5. Copy names, dates, amounts and reference numbers exactly as written.
6. Tables appear as Markdown with | separators. Read each row together with its column headers.
7. Reply in the same language as the question. Be concise and plain. Do not mention these rules.

Example
Excerpts:
[1] contract.pdf | Page 2
The notice period is 30 days.
Question: How long is the notice period?
Answer: The notice period is 30 days. [1]"""

TASKS = {
    "factual": (
        "Answer the question directly and briefly. If it asks for a list or table data, "
        "reproduce the relevant rows as a Markdown table."
    ),
    "summary": (
        "Write a clear summary as short bullet points grouped by topic. Cover the main points "
        "of every document shown and name the document for each point."
    ),
    "intro": (
        "The excerpts are the beginning of each document, where the title, headings and author "
        "usually appear. The title is the first heading or line of the excerpt TEXT, not the file name "
        "in the excerpt header. Quote the title or name exactly as written in the text."
    ),
    "compare": (
        "Compare the documents with two short bullet lists, 'Similarities' and 'Differences', "
        "or a small Markdown table. Always name which document each point comes from. "
        "Only compare what the excerpts actually show."
    ),
}
SAMPLE_NOTE = " The excerpts are only a sample of longer documents, so say that the summary may be incomplete."

REWRITE_SYSTEM_PROMPT = """You rewrite follow-up questions. Using the conversation, rewrite the user's last question so it can be understood on its own: replace words like it, they, that, those or "the second one" with what they refer to. Keep file names, numbers and the original language. Do NOT answer the question. Output only the rewritten question on one line.

Example
Conversation:
User: Who owns Project Falcon?
Assistant: Maria Lopez owns Project Falcon.
Last question: And what is its budget?
Standalone question: What is the budget of Project Falcon?"""


def estimate_tokens(messages: list[dict[str, str]]) -> int:
    """Rough token count (pessimistic) for a list of chat messages."""
    return int(sum(len(m["content"]) for m in messages) / CHARS_PER_TOKEN_SAFE) + 4 * len(messages)


def format_excerpt(number: int, hit: Hit) -> str:
    return f"[{number}] {hit.source} | {hit.location}\n{hit.text}"


def _assemble(
    question: str, hits: list[Hit], mode: str, history: list[tuple[str, str]], note: str
) -> list[dict[str, str]]:
    excerpts = "\n\n".join(format_excerpt(i, hit) for i, hit in enumerate(hits, start=1))
    task = TASKS.get(mode, TASKS["factual"])
    if mode == "summary" and note:
        task += SAMPLE_NOTE
    user = (
        f"Excerpts from the uploaded documents:\n\n{excerpts}\n\n"
        f"Question: {question}\n\nTask: {task}\nAnswer (cite excerpts like [1]):"
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for old_question, old_answer in history:
        messages.append({"role": "user", "content": old_question})
        messages.append({"role": "assistant", "content": old_answer})
    messages.append({"role": "user", "content": user})
    return messages


def build_messages(
    question: str,
    hits: list[Hit],
    mode: str,
    history: list[tuple[str, str]],
    note: str = "",
) -> tuple[list[dict[str, str]], list[Hit]]:
    """Build the chat messages and make sure they fit the model's context window.

    Returns (messages, hits_actually_used). Excerpt numbers [1], [2]... follow hits_actually_used.
    """
    limit = settings.num_ctx - settings.max_answer_tokens - PROMPT_MARGIN_TOKENS
    turns = list(history)
    used = list(hits)
    while True:
        messages = _assemble(question, used, mode, turns, note)
        if estimate_tokens(messages) <= limit:
            break
        if turns:
            turns.pop(0)  # forget the oldest exchange first
        elif mode == "factual" and len(used) > 1:
            used.pop()    # then the least relevant excerpt (hits are ranked best-first)
        else:
            break         # summary/compare hits were already sized by retrieval
    return messages, used
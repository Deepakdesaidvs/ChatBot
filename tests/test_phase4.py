"""Phase 4 check: chunking, embeddings, ChromaDB storage and retrieval."""
from __future__ import annotations

from src.chunking import chunk_document, split_table, split_text
from src.config import settings
from src.embeddings import EmbeddingError, embed_documents
from src.loaders.base import DocElement, LoadedDocument
from src.retrieval import Retriever
from src.utils import check_ollama, new_session_id, setup_logging
from src.vector_store import SessionStore

FALCON = """Project Falcon is a plan to move the company's customer database to a new cloud platform. The project owner is Maria Lopez and the sponsor is the Chief Technology Officer, Henrik Larsen.

Budget: The approved budget for Project Falcon is 50,000 EUR. The budget covers licences, consulting and staff training. Any spending above 55,000 EUR requires board approval.

Timeline: Phase 1 (analysis) runs from January to March 2025. Phase 2 (migration) runs from April to August 2025. The final go-live date is 30 September 2025.

Risks: The main risks are data loss during migration, delays caused by vendor support, and low user adoption. Mitigation includes nightly backups, a dedicated vendor contact, and two training workshops per department.

Team: The team has six members: two engineers, one data analyst, one security specialist, one trainer and one project coordinator."""

INVOICE = """INVOICE INV-2024-001. Supplier: Acme Office Supplies GmbH, Hamburg. Customer: Northwind Traders Ltd.

Invoice date: 12 November 2024. Payment due: 12 December 2024. Payment terms: 30 days net.

Items: 20 ergonomic office chairs at 45.00 USD each (900.00 USD); 10 standing desks at 30.00 USD each (300.00 USD); shipping 50.00 USD.

Total amount due: 1,250.00 USD. Bank: Hamburger Sparkasse, IBAN DE89 3704 0044 0532 0130 00. Please quote the invoice number INV-2024-001 with your payment.

Late payments incur interest of 2 percent per month. Questions about this invoice can be sent to billing at acme-office.example."""

results: list[bool] = []


def check(label: str, ok: bool) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    results.append(ok)


def make_doc(name: str, text: str) -> LoadedDocument:
    return LoadedDocument(filename=name, elements=[DocElement(text, name, "Lines 1-12")])


def main() -> None:
    setup_logging()

    print("\n--- 1. Chunker")
    long_text = " ".join(f"Sentence number {i} is here." for i in range(200))
    parts = split_text(long_text, 300, 50)
    check(f"long text split into {len(parts)} chunks", len(parts) > 3)
    check("every chunk is at most 300 characters", all(len(p) <= 300 for p in parts))
    check("neighbouring chunks overlap", parts[1][:20] in parts[0])
    rows = ["| Item | Price |", "| --- | --- |"] + [f"| Product {i} | {i * 3} |" for i in range(60)]
    pieces = split_table("\n".join(rows), 300)
    check(f"big table split into {len(pieces)} pieces", len(pieces) > 1)
    check("every table piece repeats the header", all(p.startswith("| Item | Price |") for p in pieces))

    print("\n--- 2. Ollama and embeddings")
    status = check_ollama([settings.embed_model])
    if status["missing"] or not status["running"]:
        print("  Cannot continue:", status["error"] or f"missing model(s): {status['missing']}")
        return

    store = SessionStore(new_session_id())
    try:
        for name, text in (("falcon_plan.txt", FALCON), ("invoice.txt", INVOICE)):
            chunks = chunk_document(make_doc(name, text), size=300, overlap=50)
            vectors = embed_documents([c.embed_text for c in chunks])
            store.add_document(name, chunks, vectors)
            print(f"  {name}: {len(chunks)} chunks, vector length {len(vectors[0])}")
        check("both documents stored", set(store.list_documents()) == {"falcon_plan.txt", "invoice.txt"})

        print("\n--- 3. Retrieval")
        retriever = Retriever(store)

        def ask(question: str):
            result = retriever.retrieve(question)
            top = result.hits[0] if result.hits else None
            where = f"{top.source} [{top.location}]" if top else "-"
            print(f"  Q: {question}")
            print(f"     mode={result.mode} found={result.found} best_sim={result.best_similarity:.2f} "
                  f"hits={len(result.hits)} top={where}")
            if result.note:
                print("     note:", result.note)
            return result

        on_topic: list[float] = []
        r = ask("Who is the owner of Project Falcon?")
        check("owner question finds falcon_plan.txt", r.found and r.hits[0].source == "falcon_plan.txt")
        on_topic.append(r.best_similarity)
        r = ask("When is the go-live date?")
        check("go-live question finds falcon_plan.txt", r.found and r.hits[0].source == "falcon_plan.txt")
        on_topic.append(r.best_similarity)
        r = ask("What is the total amount due?")
        check("total question finds invoice.txt", r.found and r.hits[0].source == "invoice.txt")
        on_topic.append(r.best_similarity)
        r = ask("What are the payment terms?")
        check("payment-terms question finds invoice.txt", r.found and r.hits[0].source == "invoice.txt")
        on_topic.append(r.best_similarity)

        r = ask("Find INV-2024-001")
        check("exact reference number is found by keyword search", r.found and r.hits[0].source == "invoice.txt")

        off_topic: list[float] = []
        for question in ("What is the capital of France?", "How do I bake sourdough bread?",
                         "Who won the 2018 football World Cup?"):
            r = ask(question)
            check("off-topic question is reported as NOT found", not r.found)
            off_topic.append(r.best_similarity)

        r = ask("Summarize invoice.txt")
        check("summary mode, only the named file", r.mode == "summary" and r.sources == ["invoice.txt"])
        r = ask("Give me an overview of everything")
        check("summary of all documents covers both", r.mode == "summary" and len(r.sources) == 2)
        r = ask("Compare the two documents")
        check("compare mode uses both documents", r.mode == "compare" and len(r.sources) == 2)

        print("\n--- 4. Calibrating MIN_RELEVANCE")
        print(f"  Lowest on-topic best_sim : {min(on_topic):.2f}")
        print(f"  Highest off-topic best_sim: {max(off_topic):.2f}")
        print(f"  Current MIN_RELEVANCE     : {settings.min_relevance:.2f}")
        if min(on_topic) > max(off_topic):
            suggestion = round((min(on_topic) + max(off_topic)) / 2, 2)
            print(f"  Suggested MIN_RELEVANCE   : {suggestion:.2f}  (edit it in your .env file)")
        else:
            print("  The ranges overlap, so no single value separates them perfectly. Keep the")
            print("  value just above the highest off-topic number; we also use keyword matching.")
    except EmbeddingError as exc:
        print("  Embedding problem:", exc)
    finally:
        store.delete()

    print(f"\n{sum(results)} of {len(results)} checks passed.")


if __name__ == "__main__":
    main()
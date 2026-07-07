import sys
from app.rag import RAGPipeline


def main():
    if len(sys.argv) < 2:
        print('Usage: python ask.py "your question here"')
        return

    question = sys.argv[1]

    rag = RAGPipeline()

    try:
        result = rag.ask(question)
    except ValueError as exc:
        print(exc)
        return 1

    print("\nQuestion:")
    print(result["question"])

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")
    for source in result["sources"]:
        metadata = source.get("metadata") or {}
        source_name = metadata.get("source_file", source["id"])
        page = metadata.get("page")
        page_text = f" page {page}" if page is not None else ""
        print(f"- {source_name}{page_text} | score: {source['score']:.4f}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
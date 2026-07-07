import sys
from pathlib import Path

from app.chroma_vector_store import ChromaVectorStore
from app.indexer import DocumentIndexer


SUPPORTED_SUFFIXES = {".pdf", ".txt"}


def collect_documents(paths: list[str]) -> list[str]:
    documents = []

    for raw_path in paths:
        path = Path(raw_path)

        if path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES:
                    documents.append(str(candidate))
            continue

        if path.is_file():
            if path.suffix.lower() in SUPPORTED_SUFFIXES:
                documents.append(str(path))
            continue

        raise FileNotFoundError(f"File or directory not found: {raw_path}")

    return documents


def main() -> int:
    input_paths = sys.argv[1:] or ["data"]
    documents = collect_documents(input_paths)

    if not documents:
        print("No supported documents found to index.")
        return 1

    indexer = DocumentIndexer()
    vector_store = ChromaVectorStore()
    vector_store.clear()

    indexed_chunks = indexer.build_many(documents)

    for chunk in indexed_chunks:
        vector_store.add(
            chunk_id=chunk["id"],
            text=chunk["text"],
            embedding=chunk["embedding"],
            metadata=chunk["metadata"],
        )

    print(f"Indexed {len(documents)} document(s) and {len(indexed_chunks)} chunk(s).")
    for document in documents:
        print(f"- {document}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

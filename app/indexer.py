import re
from pathlib import Path

from app.embeddings import EmbeddingService
from app.loader import load_document_pages
from app.splitter import split_text

class DocumentIndexer:
    def __init__(self, embedding_service: EmbeddingService | None = None):
        self.embedding_service = embedding_service or EmbeddingService()

    def _slugify(self, value: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
        return slug or "document"

    def _build_document_chunks(self, file_path: str) -> list[dict]:
        path = Path(file_path)
        page_blocks = load_document_pages(file_path)

        indexed_chunks = []

        for page_block in page_blocks:
            chunks = split_text(page_block["text"])
            chunks_texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embedding_service.embed_texts(chunks_texts)

            for chunk_number, (chunk, embedding) in enumerate(
                zip(chunks, embeddings),
                start=1,
            ):
                indexed_chunks.append(
                    {
                        "id": f"{self._slugify(path.stem)}_p{page_block['page']}_c{chunk_number}",
                        "text": chunk["text"],
                        "embedding": embedding,
                        "metadata": {
                            "source_file": path.name,
                            "source_path": str(path.resolve()),
                            "page": page_block["page"],
                            "chunk": chunk_number,
                        },
                    }
                )

        return indexed_chunks

    def build(self, file_path: str) -> list[dict]:
        return self._build_document_chunks(file_path)

    def build_many(self, file_paths: list[str]) -> list[dict]:
        indexed_chunks = []

        for file_path in file_paths:
            indexed_chunks.extend(self._build_document_chunks(file_path))

        return indexed_chunks

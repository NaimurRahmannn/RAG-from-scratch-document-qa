from app.loader import load_text_file
from app.splitter import split_text
from app.embeddings import EmbeddingService
from app.loader import load_document



class DocumentIndexer:
    def __init__(self):
        self.embedding_service = EmbeddingService()

    def build(self, file_path: str) -> list[dict]:
        text = load_document(file_path)
        chunks = split_text(text)
        chunks_texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_service.embed_texts(chunks_texts)
        indexed_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            indexed_chunks.append(
                {
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "embedding": embedding,
                }
            )

        return indexed_chunks

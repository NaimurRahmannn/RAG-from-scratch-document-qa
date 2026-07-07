from app.embeddings import EmbeddingService
from app.vector_store import SimpleVectorStore


class Retriever:
    """
    Responsible for retrieving the most relevant document chunks
    for a user's question.
    """

    def __init__(
        self,
        vector_store: SimpleVectorStore,
    ):
        self.vector_store = vector_store
        self.embedding_service = EmbeddingService()

    def retrieve(
        self,
        question: str,
        top_k: int,
    ) -> list[dict]:

        question_embedding = self.embedding_service.embed_text(question)

        return self.vector_store.search(
            query_embedding=question_embedding,
            top_k=top_k,
        )
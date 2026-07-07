import chromadb

from app.base_vector_store import VectorStore


class ChromaVectorStore(VectorStore):

    def __init__(self):
        client = chromadb.PersistentClient(
            path="./chroma_db"
        )

        self.collection = client.get_or_create_collection(
            name="documents"
        )

    def add(
        self,
        chunk_id: str,
        text: str,
        embedding: list[float],
    ) -> None:

        self.collection.add(
            ids=[chunk_id],
            documents=[text],
            embeddings=[embedding],
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict]:

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        retrieved_chunks = []

        for chunk_id, text in zip(
            results["ids"][0],
            results["documents"][0],
        ):

            retrieved_chunks.append(
                {
                    "id": chunk_id,
                    "text": text,
                }
            )

        return retrieved_chunks
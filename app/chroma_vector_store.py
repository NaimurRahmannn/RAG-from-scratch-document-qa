import chromadb

from app.base_vector_store import VectorStore


class ChromaVectorStore(VectorStore):

    def __init__(self, path: str = "./chroma_db", collection_name: str = "documents"):
        self.path = path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def clear(self) -> None:
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        chunk_id: str,
        text: str,
        embedding: list[float],
        metadata: dict | None = None,
    ) -> None:

        self.collection.upsert(
            ids=[chunk_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata or {}],
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict]:

        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved_chunks = []

        for chunk_id, text, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):

            retrieved_chunks.append(
                {
                    "id": chunk_id,
                    "text": text,
                    "metadata": metadata or {},
                    "score": 1.0 - float(distance),
                }
            )

        return retrieved_chunks
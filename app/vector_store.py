import numpy as np

from app.base_vector_store import VectorStore


class SimpleVectorStore(VectorStore):

    def __init__(self):
        self.items = []

    def add(
        self,
        chunk_id: str,
        text: str,
        embedding: list[float],
    ) -> None:

        self.items.append(
            {
                "id": chunk_id,
                "text": text,
                "embedding": embedding,
            }
        )

    def _cosine_similarity(
        self,
        vector_a: list[float],
        vector_b: list[float],
    ) -> float:

        a = np.array(vector_a)
        b = np.array(vector_b)

        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0

        return float(
            np.dot(a, b)
            / (np.linalg.norm(a) * np.linalg.norm(b))
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> list[dict]:

        scored_items = []

        for item in self.items:

            score = self._cosine_similarity(
                query_embedding,
                item["embedding"],
            )

            scored_items.append(
                {
                    "id": item["id"],
                    "text": item["text"],
                    "embedding": item["embedding"],
                    "score": score,
                }
            )

        scored_items.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return scored_items[:top_k]
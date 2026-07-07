from abc import ABC, abstractmethod


class VectorStore(ABC):
    """
    Base class (interface) for all vector stores.
    """

    @abstractmethod
    def add(
        self,
        chunk_id: str,
        text: str,
        embedding: list[float],
    ) -> None:
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict]:
        pass
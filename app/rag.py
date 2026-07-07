from app.chroma_vector_store import ChromaVectorStore
from app.prompt import build_rag_prompt
from app.llm import LLMService
from app.config import TOP_K
from app.retriever import Retriever


class RAGPipeline:
    """
    Full RAG pipeline.

    Steps:
    1. Load document
    2. Split into chunks
    3. Embed chunks
    4. Store vectors
    5. Embed user question
    6. Retrieve relevant chunks
    7. Build prompt
    8. Generate answer
    """

    def __init__(self):
        self.vector_store = ChromaVectorStore()
        self.llm_service = LLMService()
        self.retriever = Retriever(self.vector_store)

    def ask(self, question: str, top_k: int = TOP_K) -> dict:
        """
        Ask a question against the document.
        """
        retrieved_chunks = self.retriever.retrieve(
            question=question,
            top_k=top_k,
        )

        if not retrieved_chunks:
            raise ValueError(
                "No indexed documents were found. Run `python index.py` first."
            )

        prompt = build_rag_prompt(question=question, retrieved_chunks=retrieved_chunks)

        answer = self.llm_service.generate_answer(prompt)

        return {"question": question, "answer": answer, "sources": retrieved_chunks}

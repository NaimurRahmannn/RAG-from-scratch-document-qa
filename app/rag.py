from app.loader import load_text_file
from app.splitter import split_text
from app.embeddings import EmbeddingService
from app.vectore_store import SimpleVectorStore
from app.prompt import build_rag_prompt
from app.llm import LLMService
from app.indexer import DocumentIndexer


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

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.embedding_service = EmbeddingService()
        self.vector_store = SimpleVectorStore()
        self.llm_service = LLMService()
        indexer = DocumentIndexer()
        indexed_chunks = indexer.build(file_path)
        for chunk in indexed_chunks:
            self.vector_store.add(
                chunk_id=chunk["id"], text=chunk["text"], embedding=chunk["embedding"]
            )

    def ask(self, question: str, top_k: int = 3) -> dict:
        """
        Ask a question against the document.
        """
        question_embedding = self.embedding_service.embed_text(question)

        retrieved_chunks = self.vector_store.search(
            query_embedding=question_embedding, top_k=top_k
        )

        prompt = build_rag_prompt(question=question, retrieved_chunks=retrieved_chunks)

        answer = self.llm_service.generate_answer(prompt)

        return {"question": question, "answer": answer, "sources": retrieved_chunks}

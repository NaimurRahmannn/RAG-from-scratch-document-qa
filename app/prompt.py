def _format_source_label(chunk: dict) -> str:
    metadata = chunk.get("metadata") or {}
    source_file = metadata.get("source_file")
    page = metadata.get("page")

    if source_file and page:
        return f"{source_file} (page {page})"

    if source_file:
        return source_file

    return chunk.get("id", "unknown source")


def build_rag_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Build a grounded RAG prompt.

    Terms:
    - Prompt: instruction sent to the LLM.
    - Context: retrieved document chunks.
    - Grounding: forcing answer to come from the provided context.
    """

    context = "\n\n".join(
        [
            f"Source: {_format_source_label(chunk)}\n{chunk['text']}"
            for chunk in retrieved_chunks
        ]
    )

    prompt = f"""
You are a helpful assistant.

Answer the user's question using only the context below.

If the answer is not found in the context, say:
"I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer:
""".strip()

    return prompt
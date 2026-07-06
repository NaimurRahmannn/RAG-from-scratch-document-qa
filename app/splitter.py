def split_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    if chunk_size <= overlap:
        raise ValueError("chunk size must be greater than overlap")
    chunks = []
    start = 0
    chunk_id = 1
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({"id": f"chunk_{chunk_id}", "text": chunk_text})
            chunk_id += 1
        start += chunk_size - overlap
    return chunks

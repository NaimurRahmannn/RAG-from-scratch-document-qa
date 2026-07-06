# Basic Document Q&A RAG App (No LangChain)

This project is a simple Retrieval-Augmented Generation (RAG) application built from scratch in Python, without LangChain.

It takes a local text document, splits it into chunks, embeds those chunks, retrieves the most relevant chunks for a question using cosine similarity, and asks Gemini to generate a grounded answer.

## Features

- End-to-end RAG pipeline implemented manually.
- Uses `sentence-transformers` for embeddings.
- Uses an in-memory vector store with cosine similarity (`numpy`).
- Uses Gemini (`google-genai`) for answer generation.
- Returns retrieved source chunk IDs and scores.

## Project Structure

```text
ask.py
app/
	embeddings.py      # Embedding service (SentenceTransformer)
	llm.py             # Gemini client wrapper
	loader.py          # Document loader (.txt)
	prompt.py          # Grounded RAG prompt builder
	rag.py             # Full pipeline orchestration
	retriever.py       # (currently unused/empty)
	splitter.py        # Chunking logic with overlap
	vectore_store.py   # In-memory vector store + cosine search
data/
	sample.txt
requirements.txt
```

## How It Works

1. Load document text from a local `.txt` file.
2. Split text into overlapping chunks.
3. Embed all chunks.
4. Store chunk embeddings in memory.
5. Embed the user question.
6. Retrieve top-k similar chunks.
7. Build a prompt with retrieved context.
8. Ask Gemini to answer using only that context.

## Requirements

- Python 3.10+
- Gemini API key

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

## Run

```bash
python ask.py "Tell me about RAG"
```

Example output includes:

- The question
- The generated answer
- Retrieved source chunks with similarity scores

## Key Modules

- `app/rag.py`: orchestrates indexing and Q&A flow.
- `app/splitter.py`: character-based chunking (`chunk_size=500`, `overlap=100` by default).
- `app/embeddings.py`: wraps `all-MiniLM-L6-v2` embeddings model.
- `app/vectore_store.py`: brute-force top-k cosine retrieval.
- `app/prompt.py`: builds a grounded prompt and fallback instruction.
- `app/llm.py`: sends prompt to Gemini and returns text.

## Notes and Current Limitations

- Supports local `.txt` documents (single file input).
- Vector store is in-memory (no persistence).
- Retrieval is brute-force; suitable for small documents.
- `app/retriever.py` is currently empty and unused.
- In `app/loader.py`, the error message says `.txtx` (typo), but the code actually validates `.txt` correctly.

## Troubleshooting

### `httpcore.ReadError: [Errno 104] Connection reset by peer`

This commonly happens when `HTTP_PROXY`/`HTTPS_PROXY` env vars route Gemini requests through a proxy that resets the connection.

Quick test/workaround:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
	python ask.py "What is this document about?"
```

If that works, your proxy is the cause. Use a proxy allowlist/bypass (if allowed in your environment), or keep running without those proxy vars for this script.

## Future Improvements

- Add persistent vector storage (FAISS, Chroma, SQLite-backed store).
- Support multiple documents and metadata filtering.
- Add configurable chunking strategies.
- Add retry/backoff and graceful network error handling.
- Add unit tests for splitter, vector search, and prompt building.


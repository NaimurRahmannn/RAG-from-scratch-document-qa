# RAG From Scratch: Document Q&A

A small Retrieval-Augmented Generation (RAG) project built in Python without LangChain. It indexes local documents, stores their embeddings in ChromaDB, retrieves the most relevant chunks for a question, and asks Gemini to answer using only the retrieved context.

The current version supports `.txt` and `.pdf` files, keeps page/source metadata, and uses a two-step workflow:

1. Index documents into a persistent local Chroma collection.
2. Ask questions against the saved index.

## Features

- RAG pipeline implemented with small, readable Python modules.
- Local document indexing for files and directories.
- Supports `.txt` and text-based `.pdf` documents.
- Page-level PDF loading with source metadata.
- Character chunking with overlap.
- Sentence Transformer embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- Persistent ChromaDB vector storage in `./chroma_db`.
- Gemini answer generation through `google-genai`.
- Grounded prompt that tells the model to answer only from retrieved context.
- Source output with file name, page number, and similarity score.
- Includes a simple in-memory vector store implementation for learning/comparison.

## Project Structure

```text
.
+-- ask.py                     # Ask a question against the persisted Chroma index
+-- index.py                   # Index .txt/.pdf files or directories into Chroma
+-- requirements.txt           # Python dependencies
+-- data/
|   +-- sample.pdf             # Example document
+-- app/
|   +-- base_vector_store.py   # Vector store interface
|   +-- chroma_vector_store.py # Persistent ChromaDB vector store
|   +-- config.py              # Chunking, retrieval, and embedding defaults
|   +-- embeddings.py          # SentenceTransformer wrapper
|   +-- indexer.py             # Document -> pages -> chunks -> embeddings
|   +-- llm.py                 # Gemini client wrapper
|   +-- loader.py              # .txt and .pdf loading helpers
|   +-- prompt.py              # Grounded RAG prompt builder
|   +-- rag.py                 # Query-time RAG orchestration
|   +-- retriever.py           # Question embedding + vector search
|   +-- splitter.py            # Chunking logic with overlap
|   +-- vector_store.py        # Simple in-memory vector store
+-- test_chroma.py             # Chroma smoke test
+-- test_pdf_loader.py         # PDF loader smoke test
```

## How It Works

### Indexing

`index.py` collects supported documents from the paths you pass in. If no path is provided, it indexes the `data/` directory.

For each document:

1. Load text from `.txt` or `.pdf`.
2. Preserve page metadata for PDFs.
3. Split page text into overlapping chunks.
4. Embed chunks with Sentence Transformers.
5. Store chunk text, embeddings, and metadata in ChromaDB.

Important: `index.py` clears the existing Chroma collection before adding the new index.

### Asking

`ask.py` loads the persisted Chroma collection and runs the query-time pipeline:

1. Embed the user question.
2. Retrieve the top matching chunks from ChromaDB.
3. Build a grounded prompt with the retrieved context.
4. Send the prompt to Gemini.
5. Print the answer and source list.

If the index is empty, the app asks you to run `python index.py` first.

## Requirements

- Python 3.10+
- Gemini API key
- Internet access the first time Sentence Transformers downloads the embedding model

## Installation

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS/Linux

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

`app/llm.py` requires both values. If either one is missing, the app raises a clear error.

## Usage

### 1. Index Documents

Index the default `data/` directory:

```bash
python index.py
```

Index a specific file:

```bash
python index.py data/sample.pdf
```

Index multiple files or directories:

```bash
python index.py data docs/example.txt docs/reports
```

Example output:

```text
Indexed 1 document(s) and 8 chunk(s).
- data\sample.pdf
```

### 2. Ask Questions

```bash
python ask.py "What is this document about?"
```

Example output format:

```text
Question:
What is this document about?

Answer:
...

Sources:
- sample.pdf page 1 | score: 0.7421
```

## Configuration

Defaults live in `app/config.py`.

| Setting | Default | Purpose |
| --- | --- | --- |
| `CHUNK_SIZE` | `500` | Maximum characters per chunk |
| `CHUNK_OVERLAP` | `100` | Characters shared between adjacent chunks |
| `TOP_K` | `3` | Number of chunks retrieved per question |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model used for documents and questions |

ChromaDB settings are defined in `app/chroma_vector_store.py`:

| Setting | Default |
| --- | --- |
| Persist directory | `./chroma_db` |
| Collection name | `documents` |
| Distance space | cosine |

## Key Modules

- `app/indexer.py`: builds searchable chunks from one or more documents.
- `app/loader.py`: loads text files and text-based PDFs.
- `app/splitter.py`: creates overlapping chunks.
- `app/embeddings.py`: wraps the Sentence Transformer model.
- `app/chroma_vector_store.py`: stores and searches embeddings with ChromaDB.
- `app/retriever.py`: embeds questions and retrieves matching chunks.
- `app/prompt.py`: formats sources into a grounded prompt.
- `app/llm.py`: sends prompts to Gemini.
- `app/rag.py`: coordinates query-time retrieval and generation.

## Smoke Tests

These scripts are lightweight checks rather than a full test suite.

```bash
python test_chroma.py
python test_pdf_loader.py
```

You can also run a syntax check:

```bash
python -m py_compile index.py ask.py app/*.py
```

On PowerShell, use:

```powershell
python -m py_compile index.py ask.py (Get-ChildItem app\*.py)
```

## Troubleshooting

### `No indexed documents were found. Run python index.py first.`

The Chroma collection is empty. Run:

```bash
python index.py
```

Then ask your question again.

### `Gemini API key is missing in .env`

Create a `.env` file in the project root and set:

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

### PDF text is empty or incomplete

This project uses `pypdf` text extraction. It works for PDFs that contain selectable text. Scanned image PDFs need OCR first.

### `httpcore.ReadError` or connection reset errors

This can happen when proxy environment variables route Gemini requests through a proxy that resets the connection.

On macOS/Linux, try:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  python ask.py "What is this document about?"
```

On PowerShell, temporarily remove proxy variables for the current session:

```powershell
Remove-Item Env:HTTP_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:HTTPS_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:http_proxy -ErrorAction SilentlyContinue
Remove-Item Env:https_proxy -ErrorAction SilentlyContinue
python ask.py "What is this document about?"
```

## Current Limitations

- The active query pipeline uses ChromaDB; the in-memory vector store is kept as a simple reference implementation.
- Re-running `index.py` clears and rebuilds the Chroma collection.
- PDF support depends on text extraction from `pypdf`; there is no OCR.
- There is no web UI or API server yet.
- There is no retry/backoff layer around Gemini requests.
- The smoke tests are not a complete automated test suite.

## Roadmap Ideas

- Add a CLI option for `top_k`.
- Add incremental indexing without clearing the whole collection.
- Add metadata filters for source file or page ranges.
- Add OCR support for scanned PDFs.
- Add unit tests for splitting, indexing, retrieval, and prompt generation.
- Add a minimal web UI or API endpoint.

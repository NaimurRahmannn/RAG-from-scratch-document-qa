from pathlib import Path
from pypdf import PdfReader


def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_pdf(path: Path) -> str:
    reader = PdfReader(path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def load_pdf_pages(path: Path) -> list[dict]:
    reader = PdfReader(path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        if page_text.strip():
            pages.append(
                {
                    "page": page_number,
                    "text": page_text,
                }
            )

    return pages


def load_document(file_path: str) -> str:
    """
    Load a supported document and return its text.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".txt":
        return load_text_file(path)

    elif suffix == ".pdf":
        return load_pdf(path)

    raise ValueError(f"Unsupported file type: {suffix}")


def load_document_pages(file_path: str) -> list[dict]:
    """
    Load a supported document and return page-level text blocks.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".txt":
        return [
            {
                "page": 1,
                "text": load_text_file(path),
            }
        ]

    if suffix == ".pdf":
        return load_pdf_pages(path)

    raise ValueError(f"Unsupported file type: {suffix}")
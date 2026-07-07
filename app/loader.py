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
from pathlib import Path

def load_text_file(file_path: str)->str:
    path=Path(file_path)
    if not path.exists():
        raise FileExistsError(f"File not found: {file_path}")
    if path.suffix!=".txt":
        raise ValueError("For version 0, only .txtx files")
    return path.read_text(encoding="utf-8")

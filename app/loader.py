from pathlib import Path
from app.config import SUPPORTED_EXTENSIONS
def load_text_file(file_path: str)->str:
    path=Path(file_path)
    if not path.exists():
        raise FileExistsError(f"File not found: {file_path}")
    if path.suffix!=SUPPORTED_EXTENSIONS:
        raise ValueError("For version 0, only .txtx files")
    return path.read_text(encoding="utf-8")

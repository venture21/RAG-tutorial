"""OS-independent paths; importing this module creates no files or clients."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def data_directory() -> Path:
    value = os.getenv("RAG_DATA_DIR", "data")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else BASE_DIR / path).resolve()

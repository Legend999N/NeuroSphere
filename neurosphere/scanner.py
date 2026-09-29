from pathlib import Path
from typing import List

from .config import SUPPORTED_EXTENSIONS


def scan_directory(directory: Path) -> List[Path]:
    """
    Recursively find supported files inside a directory.
    """

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory does not exist: {directory}"
        )

    if not directory.is_dir():
        raise NotADirectoryError(
            f"Not a directory: {directory}"
        )

    files = []

    for path in directory.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)

    return sorted(files)
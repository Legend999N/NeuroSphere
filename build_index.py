from pathlib import Path

from neurosphere.config import ensure_directories
from neurosphere.indexer import build_index


def main():
    ensure_directories()

    print("=== NeuroSphere Index Builder ===")
    print()

    folder = input(
        "Enter the folder to index: "
    ).strip()

    if not folder:
        print("No folder supplied.")
        return

    source_directory = Path(folder)

    if not source_directory.exists():
        print("Folder does not exist.")
        return

    build_index(
        source_directory,
        use_embeddings=True,
    )


if __name__ == "__main__":
    main()
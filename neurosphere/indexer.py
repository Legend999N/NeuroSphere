from pathlib import Path
import time

from .config import INDEX_DIR
from .database import Database
from .scanner import scan_directory
from .extractors import extract_text
from .chunker import chunk_text
from .embeddings import EmbeddingEngine, save_embeddings


def build_index(
    source_directory: Path,
    use_embeddings: bool = True,
) -> None:

    source_directory = source_directory.resolve()

    db_path = INDEX_DIR / "neurosphere.db"

    database = Database(db_path)

    print("\nClearing old index...")
    database.clear()

    files = scan_directory(source_directory)

    print(f"Found {len(files)} supported files.")

    all_chunk_ids = []
    all_chunk_texts = []

    start_time = time.perf_counter()

    for file_number, file_path in enumerate(files, start=1):

        print(
            f"[{file_number}/{len(files)}] "
            f"Processing: {file_path}"
        )

        text = extract_text(file_path)

        if not text:
            print("  -> No text extracted")
            continue

        chunks = chunk_text(text)

        if not chunks:
            continue

        stat = file_path.stat()

        try:
            file_id = database.add_file(
                path=str(file_path),
                name=file_path.name,
                extension=file_path.suffix.lower(),
                size=stat.st_size,
                modified=stat.st_mtime,
            )
        except Exception as exc:
            print(f"  -> Database error: {exc}")
            continue

        for index, chunk in enumerate(chunks):

            chunk_id = database.add_chunk(
                file_id=file_id,
                chunk_index=index,
                content=chunk,
            )

            all_chunk_ids.append(chunk_id)
            all_chunk_texts.append(chunk)

    database.commit()

    print(
        f"\nIndexed {len(all_chunk_texts)} chunks."
    )

    if use_embeddings and all_chunk_texts:

        print("\nCreating semantic embeddings...")

        embedding_engine = EmbeddingEngine()

        embeddings = embedding_engine.encode_documents(
            all_chunk_texts,
            batch_size=8,
        )

        save_embeddings(
            embeddings,
            all_chunk_ids,
            INDEX_DIR,
        )

        print(
            f"Saved embeddings: {embeddings.shape}"
        )

    elapsed = time.perf_counter() - start_time

    print(
        f"\nIndexing completed in {elapsed:.2f} seconds."
    )

    database.close()
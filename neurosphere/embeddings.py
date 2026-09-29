from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL


class EmbeddingEngine:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        print(f"Loading embedding model: {model_name}")

        self.model = SentenceTransformer(
            model_name,
            device="cpu",
        )

    def encode_documents(
        self,
        texts: List[str],
        batch_size: int = 8,
    ) -> np.ndarray:

        prepared = [
            f"passage: {text}"
            for text in texts
        ]

        embeddings = self.model.encode(
            prepared,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:

        embedding = self.model.encode(
            [f"query: {query}"],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding[0].astype(np.float32)


def save_embeddings(
    embeddings: np.ndarray,
    chunk_ids: List[int],
    index_dir: Path,
) -> None:

    index_dir.mkdir(parents=True, exist_ok=True)

    np.save(
        index_dir / "embeddings.npy",
        embeddings,
    )

    np.save(
        index_dir / "chunk_ids.npy",
        np.array(chunk_ids, dtype=np.int64),
    )


def load_embeddings(
    index_dir: Path,
) -> Tuple[np.ndarray, np.ndarray]:

    embeddings = np.load(
        index_dir / "embeddings.npy"
    )

    chunk_ids = np.load(
        index_dir / "chunk_ids.npy"
    )

    return embeddings, chunk_ids
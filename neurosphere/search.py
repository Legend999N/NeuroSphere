from pathlib import Path
from typing import List, Dict

import numpy as np

from .config import INDEX_DIR, DEFAULT_TOP_K, RRF_K
from .database import Database
from .embeddings import EmbeddingEngine, load_embeddings


class SearchEngine:

    def __init__(self):
        db_path = INDEX_DIR / "neurosphere.db"

        if not db_path.exists():
            raise FileNotFoundError(
                "Index does not exist. Build the index first."
            )

        self.database = Database(db_path)

        self.embedding_engine = None

        self.embeddings = None
        self.embedding_chunk_ids = None

        embeddings_path = INDEX_DIR / "embeddings.npy"
        ids_path = INDEX_DIR / "chunk_ids.npy"

        if embeddings_path.exists() and ids_path.exists():
            self.embeddings, self.embedding_chunk_ids = (
                load_embeddings(INDEX_DIR)
            )

    def _load_embedding_engine(self):
        if self.embedding_engine is None:
            self.embedding_engine = EmbeddingEngine()

    def bm25(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Dict]:

        return self.database.bm25_search(
            query,
            limit=top_k,
        )

    def dense(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Dict]:

        if self.embeddings is None:
            raise RuntimeError(
                "Semantic index does not exist. "
                "Rebuild the index with embeddings."
            )

        self._load_embedding_engine()

        query_embedding = (
            self.embedding_engine.encode_query(query)
        )

        # Embeddings are normalized, so dot product = cosine similarity.
        similarities = self.embeddings @ query_embedding

        top_indices = np.argsort(
            similarities
        )[::-1][:top_k]

        chunk_ids = self.embedding_chunk_ids[top_indices]

        return self._get_chunks_by_ids(
            chunk_ids,
            similarities[top_indices],
        )

    def hybrid(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Dict]:

        lexical_results = self.bm25(
            query,
            top_k=top_k * 3,
        )

        dense_results = self.dense(
            query,
            top_k=top_k * 3,
        )

        scores = {}

        result_lookup = {}

        for rank, result in enumerate(
            lexical_results,
            start=1,
        ):
            chunk_id = result["chunk_id"]

            scores[chunk_id] = scores.get(
                chunk_id,
                0.0,
            ) + (
                1.0 / (RRF_K + rank)
            )

            result_lookup[chunk_id] = result

        for rank, result in enumerate(
            dense_results,
            start=1,
        ):
            chunk_id = result["chunk_id"]

            scores[chunk_id] = scores.get(
                chunk_id,
                0.0,
            ) + (
                1.0 / (RRF_K + rank)
            )

            result_lookup[chunk_id] = result

        ranked_ids = sorted(
            scores,
            key=scores.get,
            reverse=True,
        )

        results = []

        for chunk_id in ranked_ids[:top_k]:
            result = dict(result_lookup[chunk_id])
            result["score"] = scores[chunk_id]
            result["retrieval"] = "hybrid"
            results.append(result)

        return results

    def _get_chunks_by_ids(
        self,
        chunk_ids,
        similarity_scores,
    ) -> List[Dict]:

        chunk_id_list = [
            int(chunk_id)
            for chunk_id in chunk_ids
        ]

        if not chunk_id_list:
            return []

        placeholders = ",".join(
            "?" for _ in chunk_id_list
        )

        cursor = self.database.connection.cursor()

        cursor.execute(
            f"""
            SELECT
                c.chunk_id,
                c.content,
                f.file_id,
                f.path,
                f.name,
                f.extension
            FROM chunks c
            JOIN files f
                ON f.file_id = c.file_id
            WHERE c.chunk_id IN ({placeholders})
            """,
            chunk_id_list,
        )

        rows = {
            row[0]: row
            for row in cursor.fetchall()
        }

        results = []

        for chunk_id, score in zip(
            chunk_id_list,
            similarity_scores,
        ):

            row = rows.get(chunk_id)

            if row is None:
                continue

            results.append(
                {
                    "chunk_id": row[0],
                    "content": row[1],
                    "file_id": row[2],
                    "path": row[3],
                    "name": row[4],
                    "extension": row[5],
                    "score": float(score),
                    "retrieval": "dense",
                }
            )

        return results

    def close(self):
        self.database.close()
import sqlite3
from pathlib import Path
from typing import List, Dict


class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.connection = sqlite3.connect(str(db_path))

        self.connection.execute("PRAGMA journal_mode=WAL;")
        self.connection.execute("PRAGMA synchronous=NORMAL;")

        self._create_tables()

    def _create_tables(self) -> None:
        cursor = self.connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                file_id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                extension TEXT,
                size INTEGER,
                modified REAL
            );
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                FOREIGN KEY(file_id) REFERENCES files(file_id)
            );
            """
        )

        cursor.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
            USING fts5(
                content,
                chunk_id UNINDEXED
            );
            """
        )

        self.connection.commit()

    def clear(self) -> None:
        cursor = self.connection.cursor()

        cursor.execute("DELETE FROM chunks_fts;")
        cursor.execute("DELETE FROM chunks;")
        cursor.execute("DELETE FROM files;")

        self.connection.commit()

    def add_file(
        self,
        path: str,
        name: str,
        extension: str,
        size: int,
        modified: float,
    ) -> int:

        cursor = self.connection.cursor()

        cursor.execute(
            """
            INSERT INTO files
                (path, name, extension, size, modified)
            VALUES (?, ?, ?, ?, ?)
            """,
            (path, name, extension, size, modified),
        )

        return cursor.lastrowid

    def add_chunk(
        self,
        file_id: int,
        chunk_index: int,
        content: str,
    ) -> int:

        cursor = self.connection.cursor()

        cursor.execute(
            """
            INSERT INTO chunks
                (file_id, chunk_index, content)
            VALUES (?, ?, ?)
            """,
            (file_id, chunk_index, content),
        )

        chunk_id = cursor.lastrowid

        cursor.execute(
            """
            INSERT INTO chunks_fts
                (rowid, content, chunk_id)
            VALUES (?, ?, ?)
            """,
            (chunk_id, content, chunk_id),
        )

        return chunk_id

    def commit(self) -> None:
        self.connection.commit()

    def bm25_search(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Dict]:

        cursor = self.connection.cursor()

        query = query.strip()

        if not query:
            return []

        # First try the complete query as a phrase.
        phrase_query = f'"{query.replace(chr(34), " ")}"'

        cursor.execute(
            """
            SELECT
                c.chunk_id,
                c.content,
                f.file_id,
                f.path,
                f.name,
                f.extension,
                bm25(chunks_fts) AS bm25_score
            FROM chunks_fts
            JOIN chunks c
                ON c.chunk_id = chunks_fts.chunk_id
            JOIN files f
                ON f.file_id = c.file_id
            WHERE chunks_fts MATCH ?
            ORDER BY bm25_score ASC
            LIMIT ?
            """,
            (phrase_query, limit),
        )

        rows = cursor.fetchall()

        # If the phrase produces no results,
        # fall back to individual query terms.
        if not rows:

            tokens = [
                token.replace('"', "").strip()
                for token in query.split()
                if token.strip()
            ]

            if not tokens:
                return []

            token_query = " OR ".join(
                f'"{token}"'
                for token in tokens
            )

            cursor.execute(
                """
                SELECT
                    c.chunk_id,
                    c.content,
                    f.file_id,
                    f.path,
                    f.name,
                    f.extension,
                    bm25(chunks_fts) AS bm25_score
                FROM chunks_fts
                JOIN chunks c
                    ON c.chunk_id = chunks_fts.chunk_id
                JOIN files f
                    ON f.file_id = c.file_id
                WHERE chunks_fts MATCH ?
                ORDER BY bm25_score ASC
                LIMIT ?
                """,
                (token_query, limit),
            )

            rows = cursor.fetchall()

        results = []

        for row in rows:
            (
                chunk_id,
                content,
                file_id,
                path,
                name,
                extension,
                score,
            ) = row

            results.append(
                {
                    "chunk_id": chunk_id,
                    "file_id": file_id,
                    "content": content,
                    "path": path,
                    "name": name,
                    "extension": extension,
                    "score": float(-score),
                    "retrieval": "bm25",
                }
            )

        return results

    def get_all_chunks(self) -> List[Dict]:
        cursor = self.connection.cursor()

        cursor.execute(
            """
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
            ORDER BY c.chunk_id
            """
        )

        rows = cursor.fetchall()

        return [
            {
                "chunk_id": row[0],
                "content": row[1],
                "file_id": row[2],
                "path": row[3],
                "name": row[4],
                "extension": row[5],
            }
            for row in rows
        ]

    def close(self) -> None:
        self.connection.close()
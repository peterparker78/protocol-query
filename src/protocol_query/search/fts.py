"""Full-text search using SQLite FTS5."""

from dataclasses import dataclass
from typing import Optional

from protocol_query.core.database import Database


@dataclass
class FTSResult:
    """A full-text search result."""

    chunk_id: int
    document_id: int
    protocol_id: Optional[str]
    chunk_text: str
    section_type: Optional[str]
    score: float


class FTSSearch:
    """FTS5 full-text search implementation."""

    def __init__(self, db: Database):
        self.db = db

    def search(
        self,
        query: str,
        limit: int = 10,
        protocol_ids: Optional[list[str]] = None,
        section_types: Optional[list[str]] = None,
    ) -> list[FTSResult]:
        """
        Search using FTS5 with BM25 ranking.
        """
        # Escape special FTS5 characters and build query
        fts_query = self._build_fts_query(query)

        sql = """
            SELECT
                c.id as chunk_id,
                c.document_id,
                d.protocol_id,
                c.chunk_text,
                s.section_type,
                bm25(chunks_fts) as score
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            JOIN documents d ON d.id = c.document_id
            LEFT JOIN sections s ON s.id = c.section_id
            WHERE chunks_fts MATCH ?
        """
        params: list = [fts_query]

        if protocol_ids:
            placeholders = ",".join("?" * len(protocol_ids))
            sql += f" AND d.protocol_id IN ({placeholders})"
            params.extend(protocol_ids)

        if section_types:
            placeholders = ",".join("?" * len(section_types))
            sql += f" AND s.section_type IN ({placeholders})"
            params.extend(section_types)

        sql += " ORDER BY bm25(chunks_fts) LIMIT ?"
        params.append(limit)

        results = []
        with self.db.cursor() as cur:
            cur.execute(sql, params)
            for row in cur.fetchall():
                results.append(
                    FTSResult(
                        chunk_id=row["chunk_id"],
                        document_id=row["document_id"],
                        protocol_id=row["protocol_id"],
                        chunk_text=row["chunk_text"],
                        section_type=row["section_type"],
                        score=abs(row["score"]),  # BM25 returns negative scores
                    )
                )

        return results

    def _build_fts_query(self, query: str) -> str:
        """
        Build FTS5 query string.

        Handles special characters and creates an OR query for multiple terms.
        """
        # Remove special FTS5 characters
        special_chars = '"*^:(){}[]'
        for char in special_chars:
            query = query.replace(char, " ")

        # Split into words and filter
        words = [w.strip() for w in query.split() if w.strip()]

        if not words:
            return '""'  # Empty query

        # Create OR query for better recall
        # Each word is prefix-matched with *
        query_parts = []
        for word in words:
            if len(word) >= 2:
                query_parts.append(f'"{word}"*')

        if not query_parts:
            return f'"{query}"'

        return " OR ".join(query_parts)

"""Vector similarity search."""

import struct
from dataclasses import dataclass
from typing import Optional
import math

from protocol_query.core.database import Database
from protocol_query.embeddings.local import LocalEmbeddings


@dataclass
class VectorResult:
    """A vector search result."""

    chunk_id: int
    document_id: int
    protocol_id: Optional[str]
    chunk_text: str
    section_type: Optional[str]
    score: float  # Cosine similarity (0-1)


class VectorSearch:
    """Vector similarity search using embeddings."""

    def __init__(self, db: Database, embedder: Optional[LocalEmbeddings] = None):
        self.db = db
        self.embedder = embedder or LocalEmbeddings()

    def search(
        self,
        query: str,
        limit: int = 10,
        protocol_ids: Optional[list[str]] = None,
        section_types: Optional[list[str]] = None,
    ) -> list[VectorResult]:
        """
        Search using cosine similarity.

        Note: This is a brute-force implementation. For larger datasets,
        consider using sqlite-vec or a dedicated vector database.
        """
        # Generate query embedding
        query_embedding = self.embedder.embed(query)

        # Build SQL to fetch candidates
        sql = """
            SELECT
                c.id as chunk_id,
                c.document_id,
                d.protocol_id,
                c.chunk_text,
                s.section_type,
                e.embedding
            FROM embeddings e
            JOIN chunks c ON c.id = e.chunk_id
            JOIN documents d ON d.id = c.document_id
            LEFT JOIN sections s ON s.id = c.section_id
            WHERE 1=1
        """
        params: list = []

        if protocol_ids:
            placeholders = ",".join("?" * len(protocol_ids))
            sql += f" AND d.protocol_id IN ({placeholders})"
            params.extend(protocol_ids)

        if section_types:
            placeholders = ",".join("?" * len(section_types))
            sql += f" AND s.section_type IN ({placeholders})"
            params.extend(section_types)

        # Fetch all candidates and compute similarity
        results = []
        with self.db.cursor() as cur:
            cur.execute(sql, params)
            for row in cur.fetchall():
                # Decode embedding from blob
                embedding_blob = row["embedding"]
                dim = len(embedding_blob) // 4  # 4 bytes per float
                embedding = list(struct.unpack(f"{dim}f", embedding_blob))

                # Compute cosine similarity
                similarity = self._cosine_similarity(query_embedding, embedding)

                results.append(
                    VectorResult(
                        chunk_id=row["chunk_id"],
                        document_id=row["document_id"],
                        protocol_id=row["protocol_id"],
                        chunk_text=row["chunk_text"],
                        section_type=row["section_type"],
                        score=similarity,
                    )
                )

        # Sort by similarity and return top results
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

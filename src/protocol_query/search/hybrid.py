"""Hybrid search combining FTS5 and vector search."""

from dataclasses import dataclass
from typing import Optional

from protocol_query.core.database import Database
from protocol_query.core.config import get_config
from protocol_query.search.fts import FTSSearch
from protocol_query.search.vector import VectorSearch
from protocol_query.embeddings.local import LocalEmbeddings


@dataclass
class SearchResult:
    """A hybrid search result."""

    chunk_id: int
    document_id: int
    protocol_id: Optional[str]
    chunk_text: str
    section_type: Optional[str]
    score: float
    source: str  # 'fts', 'vector', or 'hybrid'


class HybridSearch:
    """
    Combines FTS5 full-text search with vector similarity search
    using Reciprocal Rank Fusion (RRF).

    RRF Score = sum(1 / (k + rank_i)) for each result list
    """

    RRF_K = 60  # Ranking constant

    def __init__(
        self,
        db: Database,
        embedder: Optional[LocalEmbeddings] = None,
    ):
        self.db = db
        self.config = get_config()
        self.embedder = embedder or LocalEmbeddings()
        self.fts_search = FTSSearch(db)
        self.vector_search = VectorSearch(db, self.embedder)

    def search(
        self,
        query: str,
        limit: int = 10,
        protocol_ids: Optional[list[str]] = None,
        section_types: Optional[list[str]] = None,
        mode: str = "hybrid",
    ) -> list[SearchResult]:
        """
        Execute search with specified mode.

        Args:
            query: Search query text
            limit: Maximum results to return
            protocol_ids: Filter by specific protocols
            section_types: Filter by section types
            mode: 'fts', 'vector', or 'hybrid'
        """
        if mode == "fts":
            fts_results = self.fts_search.search(
                query, limit, protocol_ids, section_types
            )
            return [
                SearchResult(
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    protocol_id=r.protocol_id,
                    chunk_text=r.chunk_text,
                    section_type=r.section_type,
                    score=r.score,
                    source="fts",
                )
                for r in fts_results
            ]

        elif mode == "vector":
            vector_results = self.vector_search.search(
                query, limit, protocol_ids, section_types
            )
            return [
                SearchResult(
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    protocol_id=r.protocol_id,
                    chunk_text=r.chunk_text,
                    section_type=r.section_type,
                    score=r.score,
                    source="vector",
                )
                for r in vector_results
            ]

        else:
            return self._hybrid_search(query, limit, protocol_ids, section_types)

    def _hybrid_search(
        self,
        query: str,
        limit: int,
        protocol_ids: Optional[list[str]],
        section_types: Optional[list[str]],
    ) -> list[SearchResult]:
        """
        Combine FTS and vector results using Reciprocal Rank Fusion.
        """
        # Get more results from each source for better fusion
        fetch_limit = limit * 3

        fts_results = self.fts_search.search(
            query, fetch_limit, protocol_ids, section_types
        )
        vector_results = self.vector_search.search(
            query, fetch_limit, protocol_ids, section_types
        )

        # Build rank dictionaries
        fts_ranks = {r.chunk_id: i + 1 for i, r in enumerate(fts_results)}
        vector_ranks = {r.chunk_id: i + 1 for i, r in enumerate(vector_results)}

        # Collect all chunk data
        chunk_data = {}
        for r in fts_results:
            chunk_data[r.chunk_id] = {
                "chunk_id": r.chunk_id,
                "document_id": r.document_id,
                "protocol_id": r.protocol_id,
                "chunk_text": r.chunk_text,
                "section_type": r.section_type,
            }
        for r in vector_results:
            if r.chunk_id not in chunk_data:
                chunk_data[r.chunk_id] = {
                    "chunk_id": r.chunk_id,
                    "document_id": r.document_id,
                    "protocol_id": r.protocol_id,
                    "chunk_text": r.chunk_text,
                    "section_type": r.section_type,
                }

        # Calculate RRF scores
        all_chunk_ids = set(fts_ranks.keys()) | set(vector_ranks.keys())
        rrf_scores = {}

        for chunk_id in all_chunk_ids:
            score = 0.0
            sources = []

            if chunk_id in fts_ranks:
                score += 1.0 / (self.RRF_K + fts_ranks[chunk_id])
                sources.append("fts")

            if chunk_id in vector_ranks:
                score += 1.0 / (self.RRF_K + vector_ranks[chunk_id])
                sources.append("vector")

            rrf_scores[chunk_id] = (score, sources)

        # Sort by RRF score
        sorted_ids = sorted(
            rrf_scores.keys(),
            key=lambda x: rrf_scores[x][0],
            reverse=True,
        )

        # Build final results
        results = []
        for chunk_id in sorted_ids[:limit]:
            score, sources = rrf_scores[chunk_id]
            data = chunk_data[chunk_id]

            source = "hybrid" if len(sources) > 1 else sources[0]

            results.append(
                SearchResult(
                    chunk_id=data["chunk_id"],
                    document_id=data["document_id"],
                    protocol_id=data["protocol_id"],
                    chunk_text=data["chunk_text"],
                    section_type=data["section_type"],
                    score=score,
                    source=source,
                )
            )

        return results

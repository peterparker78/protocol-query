"""Database connection and schema management."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from protocol_query.core.config import Config, get_config


SCHEMA_SQL = """
-- Documents table: stores protocol metadata
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    filepath TEXT NOT NULL UNIQUE,
    file_hash TEXT NOT NULL,
    file_type TEXT NOT NULL,
    title TEXT,
    protocol_id TEXT,
    version TEXT,
    sponsor TEXT,
    indication TEXT,
    phase TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON
);

-- Sections table: protocol sections hierarchy
CREATE TABLE IF NOT EXISTS sections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    section_type TEXT NOT NULL,
    section_number TEXT,
    title TEXT,
    parent_section_id INTEGER REFERENCES sections(id),
    level INTEGER DEFAULT 0,
    start_page INTEGER,
    end_page INTEGER,
    raw_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Chunks table: searchable text chunks
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    section_id INTEGER REFERENCES sections(id) ON DELETE SET NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_type TEXT DEFAULT 'text',
    start_char INTEGER,
    end_char INTEGER,
    page_number INTEGER,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Eligibility criteria extracted table
CREATE TABLE IF NOT EXISTS eligibility_criteria (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    criterion_type TEXT NOT NULL,
    criterion_number INTEGER,
    criterion_text TEXT NOT NULL,
    category TEXT,
    is_required INTEGER DEFAULT 1,
    chunk_id INTEGER REFERENCES chunks(id),
    metadata JSON
);

-- Embeddings table for vector storage
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id INTEGER NOT NULL UNIQUE REFERENCES chunks(id) ON DELETE CASCADE,
    embedding BLOB NOT NULL
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_documents_protocol ON documents(protocol_id);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks(section_id);
CREATE INDEX IF NOT EXISTS idx_sections_document ON sections(document_id);
CREATE INDEX IF NOT EXISTS idx_sections_type ON sections(section_type);
CREATE INDEX IF NOT EXISTS idx_criteria_document ON eligibility_criteria(document_id);
CREATE INDEX IF NOT EXISTS idx_criteria_type ON eligibility_criteria(criterion_type);
CREATE INDEX IF NOT EXISTS idx_embeddings_chunk ON embeddings(chunk_id);
"""

FTS_SCHEMA_SQL = """
-- FTS5 virtual table for full-text search
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_text,
    content='chunks',
    content_rowid='id',
    tokenize='porter unicode61'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, chunk_text) VALUES (new.id, new.chunk_text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, chunk_text)
    VALUES('delete', old.id, old.chunk_text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, chunk_text)
    VALUES('delete', old.id, old.chunk_text);
    INSERT INTO chunks_fts(rowid, chunk_text) VALUES (new.id, new.chunk_text);
END;
"""


class Database:
    """SQLite database connection manager."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self._connection: Optional[sqlite3.Connection] = None

    @property
    def db_path(self) -> Path:
        return self.config.db_path

    def connect(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._connection is None:
            self.config.ensure_db_dir()
            self._connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA journal_mode = WAL")
        return self._connection

    def close(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None

    @contextmanager
    def cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager for database cursor."""
        conn = self.connect()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    def initialize(self) -> None:
        """Initialize database schema."""
        conn = self.connect()
        conn.executescript(SCHEMA_SQL)
        conn.executescript(FTS_SCHEMA_SQL)
        conn.commit()

    def rebuild_fts(self) -> None:
        """Rebuild the FTS index from chunks table."""
        with self.cursor() as cur:
            cur.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")

    def get_document_by_path(self, filepath: str) -> Optional[dict]:
        """Get document by file path."""
        with self.cursor() as cur:
            cur.execute(
                "SELECT * FROM documents WHERE filepath = ?",
                (filepath,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def get_document_by_id(self, doc_id: int) -> Optional[dict]:
        """Get document by ID."""
        with self.cursor() as cur:
            cur.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def get_document_by_protocol_id(self, protocol_id: str) -> Optional[dict]:
        """Get document by protocol ID."""
        with self.cursor() as cur:
            cur.execute(
                "SELECT * FROM documents WHERE protocol_id = ?",
                (protocol_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def list_documents(self) -> list[dict]:
        """List all documents."""
        with self.cursor() as cur:
            cur.execute(
                """
                SELECT d.*,
                    (SELECT COUNT(*) FROM chunks WHERE document_id = d.id) as chunk_count,
                    (SELECT COUNT(*) FROM eligibility_criteria WHERE document_id = d.id) as criteria_count
                FROM documents d
                ORDER BY d.created_at DESC
                """
            )
            return [dict(row) for row in cur.fetchall()]

    def delete_document(self, doc_id: int) -> bool:
        """Delete a document and all related data."""
        with self.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            return cur.rowcount > 0


# Global database instance
_db: Optional[Database] = None


def get_database() -> Database:
    """Get or create the global database instance."""
    global _db
    if _db is None:
        _db = Database()
    return _db

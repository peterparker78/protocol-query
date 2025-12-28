"""Ingest commands for protocol query CLI."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from protocol_query.cli.app import ingest_app
from protocol_query.core.database import get_database

console = Console()


@ingest_app.command("add")
def ingest_add(
    file_path: Path = typer.Argument(..., help="Path to protocol document"),
    protocol_id: Optional[str] = typer.Option(
        None, "--protocol-id", "-p", help="Protocol identifier"
    ),
    title: Optional[str] = typer.Option(
        None, "--title", "-t", help="Protocol title"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Re-process even if already indexed"
    ),
) -> None:
    """Add a single protocol document."""
    from protocol_query.parsers import parse_document
    from protocol_query.parsers.chunker import ProtocolChunker
    from protocol_query.embeddings.local import LocalEmbeddings
    from protocol_query.core.config import get_config

    if not file_path.exists():
        console.print(f"[red]File not found: {file_path}[/red]")
        raise typer.Exit(1)

    db = get_database()
    db.initialize()

    # Check if already indexed
    existing = db.get_document_by_path(str(file_path.resolve()))
    if existing and not force:
        console.print(
            f"[yellow]Document already indexed (ID: {existing['id']})[/yellow]"
        )
        console.print("Use --force to re-process")
        raise typer.Exit(1)

    if existing and force:
        db.delete_document(existing["id"])
        console.print("[dim]Removed existing document[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Parse document
        task = progress.add_task("Parsing document...", total=None)
        doc_data = parse_document(file_path)
        if protocol_id:
            doc_data["protocol_id"] = protocol_id
        if title:
            doc_data["title"] = title
        progress.update(task, description="[green]Parsed document[/green]")

        # Chunk document
        progress.update(task, description="Chunking document...")
        chunker = ProtocolChunker(get_config())
        chunks = chunker.chunk_document(doc_data)
        progress.update(
            task, description=f"[green]Created {len(chunks)} chunks[/green]"
        )

        # Generate embeddings
        progress.update(task, description="Generating embeddings...")
        embedder = LocalEmbeddings()
        embeddings = embedder.embed_batch([c["chunk_text"] for c in chunks])
        progress.update(task, description="[green]Generated embeddings[/green]")

        # Store in database
        progress.update(task, description="Storing in database...")
        doc_id = _store_document(db, doc_data, chunks, embeddings)
        progress.update(task, description="[green]Stored in database[/green]")

    console.print(f"\n[bold green]Document ingested successfully![/bold green]")
    console.print(f"  Document ID: {doc_id}")
    console.print(f"  Protocol ID: {doc_data.get('protocol_id', 'N/A')}")
    console.print(f"  Chunks: {len(chunks)}")


def _store_document(
    db, doc_data: dict, chunks: list[dict], embeddings: list[list[float]]
) -> int:
    """Store document and chunks in database."""
    import json
    import struct

    with db.cursor() as cur:
        # Insert document
        cur.execute(
            """
            INSERT INTO documents (filename, filepath, file_hash, file_type,
                                   title, protocol_id, version, sponsor,
                                   indication, phase, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_data["filename"],
                doc_data["filepath"],
                doc_data["file_hash"],
                doc_data["file_type"],
                doc_data.get("title"),
                doc_data.get("protocol_id"),
                doc_data.get("version"),
                doc_data.get("sponsor"),
                doc_data.get("indication"),
                doc_data.get("phase"),
                json.dumps(doc_data.get("metadata", {})),
            ),
        )
        doc_id = cur.lastrowid

        # Insert sections if present
        section_id_map = {}
        for section in doc_data.get("sections", []):
            cur.execute(
                """
                INSERT INTO sections (document_id, section_type, section_number,
                                     title, level, start_page, end_page, raw_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    section.get("section_type", "other"),
                    section.get("section_number"),
                    section.get("title"),
                    section.get("level", 0),
                    section.get("start_page"),
                    section.get("end_page"),
                    section.get("raw_text"),
                ),
            )
            section_id_map[section.get("index", 0)] = cur.lastrowid

        # Insert chunks and embeddings
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            cur.execute(
                """
                INSERT INTO chunks (document_id, section_id, chunk_index,
                                   chunk_text, chunk_type, page_number, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    section_id_map.get(chunk.get("section_index")),
                    i,
                    chunk["chunk_text"],
                    chunk.get("chunk_type", "text"),
                    chunk.get("page_number"),
                    json.dumps(chunk.get("metadata", {})),
                ),
            )
            chunk_id = cur.lastrowid

            # Store embedding as binary blob
            embedding_blob = struct.pack(f"{len(embedding)}f", *embedding)
            cur.execute(
                "INSERT INTO embeddings (chunk_id, embedding) VALUES (?, ?)",
                (chunk_id, embedding_blob),
            )

            # Store eligibility criteria if this is a criterion chunk
            if chunk.get("chunk_type") == "criterion":
                cur.execute(
                    """
                    INSERT INTO eligibility_criteria
                    (document_id, criterion_type, criterion_number,
                     criterion_text, category, chunk_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc_id,
                        chunk.get("criterion_type", "inclusion"),
                        chunk.get("criterion_number"),
                        chunk["chunk_text"],
                        chunk.get("category"),
                        chunk_id,
                    ),
                )

        return doc_id


@ingest_app.command("list")
def ingest_list(
    format: str = typer.Option(
        "table", "--format", "-f", help="Output format (table/json)"
    ),
) -> None:
    """List ingested documents."""
    import json

    db = get_database()
    docs = db.list_documents()

    if not docs:
        console.print("[dim]No documents ingested yet.[/dim]")
        return

    if format == "json":
        console.print(json.dumps(docs, indent=2, default=str))
        return

    table = Table(title="Ingested Documents")
    table.add_column("ID", style="cyan")
    table.add_column("Protocol ID")
    table.add_column("Title")
    table.add_column("Chunks", justify="right")
    table.add_column("Criteria", justify="right")
    table.add_column("Created")

    for doc in docs:
        table.add_row(
            str(doc["id"]),
            doc.get("protocol_id") or "-",
            (doc.get("title") or doc["filename"])[:40],
            str(doc.get("chunk_count", 0)),
            str(doc.get("criteria_count", 0)),
            str(doc["created_at"])[:10],
        )

    console.print(table)


@ingest_app.command("remove")
def ingest_remove(
    doc_id: int = typer.Argument(..., help="Document ID to remove"),
    confirm: bool = typer.Option(
        False, "--confirm", "-y", help="Skip confirmation"
    ),
) -> None:
    """Remove an ingested document."""
    db = get_database()
    doc = db.get_document_by_id(doc_id)

    if not doc:
        console.print(f"[red]Document not found: {doc_id}[/red]")
        raise typer.Exit(1)

    if not confirm:
        console.print(f"Document: {doc.get('title') or doc['filename']}")
        if not typer.confirm("Are you sure you want to remove this document?"):
            raise typer.Abort()

    db.delete_document(doc_id)
    console.print(f"[green]Document {doc_id} removed[/green]")

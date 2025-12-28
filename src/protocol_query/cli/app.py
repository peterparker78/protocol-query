"""Main CLI application for protocol query."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from protocol_query.core.config import Config, get_config, set_config
from protocol_query.core.database import Database, get_database

app = typer.Typer(
    name="protocol-query",
    help="Interactive clinical protocol querying CLI",
    add_completion=True,
    rich_markup_mode="rich",
)

console = Console()


# Sub-command groups
ingest_app = typer.Typer(help="Ingest and parse protocol documents")
query_app = typer.Typer(help="Query protocols")
compare_app = typer.Typer(help="Compare protocols")
config_app = typer.Typer(help="Configuration management")

app.add_typer(ingest_app, name="ingest")
app.add_typer(query_app, name="query")
app.add_typer(compare_app, name="compare")
app.add_typer(config_app, name="config")


@app.callback()
def main(
    db_path: Optional[Path] = typer.Option(
        None, "--db", help="Path to database file"
    ),
) -> None:
    """Protocol Query - Interactive clinical protocol querying CLI."""
    config = get_config()
    if db_path:
        config.db_path = db_path
        set_config(config)


# Config commands
@config_app.command("init")
def config_init(
    db_path: Optional[Path] = typer.Option(
        None, "--db-path", help="Database location"
    ),
    force: bool = typer.Option(
        False, "--force", help="Overwrite existing database"
    ),
) -> None:
    """Initialize the database."""
    config = get_config()
    if db_path:
        config.db_path = db_path
        set_config(config)

    if config.db_path.exists() and not force:
        console.print(
            f"[yellow]Database already exists at {config.db_path}[/yellow]"
        )
        console.print("Use --force to reinitialize")
        raise typer.Exit(1)

    db = Database(config)
    db.initialize()
    console.print(f"[green]Database initialized at {config.db_path}[/green]")


@config_app.command("show")
def config_show() -> None:
    """Show current configuration."""
    config = get_config()
    console.print("[bold]Current Configuration:[/bold]")
    console.print(f"  Database path: {config.db_path}")
    console.print(f"  Embedding model: {config.embedding_model}")
    console.print(f"  LLM model: {config.llm_model}")
    console.print(f"  Chunk size: {config.chunk_size}")
    console.print(f"  Chunk overlap: {config.chunk_overlap}")
    console.print(f"  Anthropic API key: {'[set]' if config.anthropic_api_key else '[not set]'}")


# Import subcommands (will be implemented later)
from protocol_query.cli import ingest, query, compare  # noqa: E402, F401

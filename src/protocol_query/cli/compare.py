"""Compare commands for protocol query CLI."""

from typing import List

import typer
from rich.console import Console

from protocol_query.cli.app import compare_app
from protocol_query.core.database import get_database

console = Console()


@compare_app.command("protocols")
def compare_protocols(
    protocol_ids: List[str] = typer.Argument(
        ..., help="Protocol IDs to compare (2 or more)"
    ),
    aspect: str = typer.Option(
        "all", "--aspect", "-a", help="Aspect to compare: all, eligibility, design"
    ),
    format: str = typer.Option(
        "table", "--format", "-f", help="Output format: table, json, markdown"
    ),
) -> None:
    """Compare two or more protocols."""
    from protocol_query.analysis.comparison import ProtocolComparer
    from protocol_query.output.formatters import format_comparison

    if len(protocol_ids) < 2:
        console.print("[red]Need at least 2 protocols to compare[/red]")
        raise typer.Exit(1)

    db = get_database()

    # Verify all protocols exist
    docs = []
    for pid in protocol_ids:
        doc = db.get_document_by_protocol_id(pid)
        if not doc:
            console.print(f"[red]Protocol not found: {pid}[/red]")
            raise typer.Exit(1)
        docs.append(doc)

    comparer = ProtocolComparer(db)
    result = comparer.compare(protocol_ids, aspect)

    output = format_comparison(result, format)
    console.print(output)


@compare_app.command("eligibility")
def compare_eligibility(
    protocol_ids: List[str] = typer.Argument(
        ..., help="Protocol IDs to compare"
    ),
    type: str = typer.Option(
        "all", "--type", "-t", help="Criteria type: inclusion, exclusion, all"
    ),
    format: str = typer.Option(
        "table", "--format", "-f", help="Output format: table, json, markdown"
    ),
) -> None:
    """Compare eligibility criteria across protocols."""
    from protocol_query.analysis.comparison import ProtocolComparer
    from protocol_query.output.formatters import format_eligibility_comparison

    if len(protocol_ids) < 2:
        console.print("[red]Need at least 2 protocols to compare[/red]")
        raise typer.Exit(1)

    db = get_database()

    # Verify all protocols exist
    for pid in protocol_ids:
        doc = db.get_document_by_protocol_id(pid)
        if not doc:
            console.print(f"[red]Protocol not found: {pid}[/red]")
            raise typer.Exit(1)

    comparer = ProtocolComparer(db)
    result = comparer.compare_eligibility(protocol_ids, type)

    output = format_eligibility_comparison(result, format)
    console.print(output)

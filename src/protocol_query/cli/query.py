"""Query commands for protocol query CLI."""

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from protocol_query.cli.app import query_app
from protocol_query.core.database import get_database
from protocol_query.core.config import get_config

console = Console()


@query_app.command("search")
def query_search(
    query_text: str = typer.Argument(..., help="Search query"),
    protocol: Optional[str] = typer.Option(
        None, "--protocol", "-p", help="Limit to specific protocol"
    ),
    mode: str = typer.Option(
        "hybrid", "--mode", "-m", help="Search mode: fts, vector, hybrid"
    ),
    limit: int = typer.Option(10, "--limit", "-n", help="Max results"),
    format: str = typer.Option(
        "text", "--format", "-f", help="Output format: text, json, table"
    ),
    section: Optional[str] = typer.Option(
        None, "--section", "-s", help="Filter by section type"
    ),
) -> None:
    """Search across protocols."""
    from protocol_query.search.hybrid import HybridSearch
    from protocol_query.output.formatters import format_search_results

    db = get_database()
    search = HybridSearch(db)

    protocol_ids = [protocol] if protocol else None
    section_types = [section] if section else None

    results = search.search(
        query=query_text,
        limit=limit,
        protocol_ids=protocol_ids,
        section_types=section_types,
        mode=mode,
    )

    if not results:
        console.print("[dim]No results found.[/dim]")
        return

    output = format_search_results(results, format)
    console.print(output)


@query_app.command("what-if")
def query_what_if(
    scenario: str = typer.Argument(..., help="What-if scenario to analyze"),
    protocol: str = typer.Option(
        ..., "--protocol", "-p", help="Protocol to analyze"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed reasoning"
    ),
) -> None:
    """Analyze a what-if scenario using LLM."""
    from protocol_query.analysis.what_if import WhatIfAnalyzer
    from protocol_query.search.hybrid import HybridSearch

    config = get_config()
    if not config.anthropic_api_key:
        console.print(
            "[red]ANTHROPIC_API_KEY not set. Please set it in your environment or .env file.[/red]"
        )
        raise typer.Exit(1)

    db = get_database()

    # Check protocol exists
    doc = db.get_document_by_protocol_id(protocol)
    if not doc:
        console.print(f"[red]Protocol not found: {protocol}[/red]")
        raise typer.Exit(1)

    search = HybridSearch(db)
    analyzer = WhatIfAnalyzer(search, db, config)

    with console.status("Analyzing scenario..."):
        result = analyzer.analyze(scenario, protocol)

    console.print(Panel(f"[bold]Scenario:[/bold] {scenario}", title="What-If Analysis"))
    console.print()
    console.print(Markdown(result.analysis))

    if verbose and result.relevant_chunks:
        console.print("\n[bold]Relevant Protocol Sections:[/bold]")
        for i, chunk in enumerate(result.relevant_chunks[:5], 1):
            console.print(f"\n[dim]{i}. {chunk.get('section_type', 'N/A')}:[/dim]")
            console.print(f"   {chunk.get('chunk_text', '')[:200]}...")


@query_app.command("eligibility")
def query_eligibility(
    protocol: str = typer.Option(
        ..., "--protocol", "-p", help="Protocol identifier"
    ),
    type: str = typer.Option(
        "all", "--type", "-t", help="Criteria type: inclusion, exclusion, all"
    ),
    format: str = typer.Option(
        "table", "--format", "-f", help="Output format: table, json, markdown"
    ),
) -> None:
    """Get eligibility criteria for a protocol."""
    from protocol_query.output.formatters import format_eligibility_criteria
    import json

    db = get_database()

    doc = db.get_document_by_protocol_id(protocol)
    if not doc:
        console.print(f"[red]Protocol not found: {protocol}[/red]")
        raise typer.Exit(1)

    with db.cursor() as cur:
        if type == "all":
            cur.execute(
                """
                SELECT * FROM eligibility_criteria
                WHERE document_id = ?
                ORDER BY criterion_type, criterion_number
                """,
                (doc["id"],),
            )
        else:
            cur.execute(
                """
                SELECT * FROM eligibility_criteria
                WHERE document_id = ? AND criterion_type = ?
                ORDER BY criterion_number
                """,
                (doc["id"], type),
            )
        criteria = [dict(row) for row in cur.fetchall()]

    if not criteria:
        console.print("[dim]No eligibility criteria found.[/dim]")
        return

    if format == "json":
        console.print(json.dumps(criteria, indent=2))
    else:
        output = format_eligibility_criteria(criteria, format)
        console.print(output)


@query_app.command("interactive")
def query_interactive(
    protocol: Optional[str] = typer.Option(
        None, "--protocol", "-p", help="Limit to specific protocol"
    ),
) -> None:
    """Start interactive query mode (REPL)."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from pathlib import Path

    from protocol_query.search.hybrid import HybridSearch
    from protocol_query.analysis.what_if import WhatIfAnalyzer
    from protocol_query.output.formatters import format_search_results

    config = get_config()
    db = get_database()
    search = HybridSearch(db)
    analyzer = None

    if config.anthropic_api_key:
        analyzer = WhatIfAnalyzer(search, db, config)

    history_file = Path.home() / ".protocol_query_history"
    session = PromptSession(history=FileHistory(str(history_file)))

    console.print("[bold]Protocol Query Interactive Mode[/bold]")
    console.print("Commands: /search, /what-if, /eligibility, /help, /quit")
    if protocol:
        console.print(f"Protocol filter: {protocol}")
    console.print()

    while True:
        try:
            user_input = session.prompt(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input.startswith("/quit") or user_input.startswith("/exit"):
            break

        if user_input.startswith("/help"):
            console.print(
                """
Commands:
  /search <query>       - Search protocols
  /what-if <scenario>   - Analyze what-if scenario (requires API key)
  /eligibility          - Show eligibility criteria
  /protocol <id>        - Set protocol filter
  /quit                 - Exit interactive mode

Or just type a query to search.
                """
            )
            continue

        if user_input.startswith("/protocol "):
            protocol = user_input[10:].strip()
            console.print(f"Protocol filter set to: {protocol}")
            continue

        if user_input.startswith("/eligibility"):
            if not protocol:
                console.print("[yellow]Set a protocol first with /protocol <id>[/yellow]")
                continue
            # Reuse eligibility function logic
            query_eligibility.callback(protocol=protocol, type="all", format="table")
            continue

        if user_input.startswith("/what-if "):
            if not analyzer:
                console.print("[red]ANTHROPIC_API_KEY not set[/red]")
                continue
            if not protocol:
                console.print("[yellow]Set a protocol first with /protocol <id>[/yellow]")
                continue
            scenario = user_input[9:].strip()
            query_what_if.callback(scenario=scenario, protocol=protocol, verbose=False)
            continue

        # Default: search
        query_text = user_input
        if user_input.startswith("/search "):
            query_text = user_input[8:].strip()

        protocol_ids = [protocol] if protocol else None
        results = search.search(query=query_text, limit=5, protocol_ids=protocol_ids)
        if results:
            output = format_search_results(results, "text")
            console.print(output)
        else:
            console.print("[dim]No results found.[/dim]")

    console.print("[dim]Goodbye![/dim]")

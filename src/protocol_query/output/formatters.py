"""Output formatters for CLI responses."""

import json
from typing import Union

from rich.table import Table
from rich.console import Console
from rich.panel import Panel
from tabulate import tabulate


def format_search_results(
    results: list,
    format_type: str = "text",
) -> Union[str, Table]:
    """Format search results for display."""
    if not results:
        return "No results found."

    if format_type == "json":
        data = []
        for r in results:
            if hasattr(r, "__dict__"):
                data.append({
                    "chunk_id": r.chunk_id,
                    "protocol_id": r.protocol_id,
                    "section_type": r.section_type,
                    "score": r.score,
                    "chunk_text": r.chunk_text,
                    "source": getattr(r, "source", None),
                })
            else:
                data.append(r)
        return json.dumps(data, indent=2)

    elif format_type == "table":
        table = Table(title="Search Results")
        table.add_column("#", style="cyan", width=3)
        table.add_column("Protocol", style="green")
        table.add_column("Section", style="yellow")
        table.add_column("Score", justify="right")
        table.add_column("Text", max_width=60)

        for i, r in enumerate(results, 1):
            protocol = getattr(r, "protocol_id", None) or r.get("protocol_id", "-")
            section = getattr(r, "section_type", None) or r.get("section_type", "-")
            score = getattr(r, "score", None) or r.get("score", 0)
            text = getattr(r, "chunk_text", None) or r.get("chunk_text", "")

            table.add_row(
                str(i),
                str(protocol) if protocol else "-",
                str(section) if section else "-",
                f"{score:.3f}",
                text[:100] + "..." if len(text) > 100 else text,
            )

        return table

    else:  # text format
        lines = []
        for i, r in enumerate(results, 1):
            protocol = getattr(r, "protocol_id", None) or r.get("protocol_id", "N/A")
            section = getattr(r, "section_type", None) or r.get("section_type", "N/A")
            score = getattr(r, "score", None) or r.get("score", 0)
            text = getattr(r, "chunk_text", None) or r.get("chunk_text", "")
            source = getattr(r, "source", None) or r.get("source", "")

            lines.append(f"\n[{i}] Protocol: {protocol} | Section: {section} | Score: {score:.3f}")
            if source:
                lines.append(f"    Source: {source}")
            lines.append(f"    {text}")

        return "\n".join(lines)


def format_eligibility_criteria(
    criteria: list[dict],
    format_type: str = "table",
) -> Union[str, Table]:
    """Format eligibility criteria for display."""
    if not criteria:
        return "No criteria found."

    if format_type == "json":
        return json.dumps(criteria, indent=2)

    elif format_type == "markdown":
        lines = ["# Eligibility Criteria\n"]

        inclusion = [c for c in criteria if c.get("criterion_type") == "inclusion"]
        exclusion = [c for c in criteria if c.get("criterion_type") == "exclusion"]

        if inclusion:
            lines.append("## Inclusion Criteria\n")
            for c in inclusion:
                num = c.get("criterion_number", "")
                text = c.get("criterion_text", "")
                lines.append(f"{num}. {text}")

        if exclusion:
            lines.append("\n## Exclusion Criteria\n")
            for c in exclusion:
                num = c.get("criterion_number", "")
                text = c.get("criterion_text", "")
                lines.append(f"{num}. {text}")

        return "\n".join(lines)

    else:  # table format
        table = Table(title="Eligibility Criteria")
        table.add_column("Type", style="cyan")
        table.add_column("#", width=3)
        table.add_column("Criterion", max_width=80)
        table.add_column("Category", style="dim")

        for c in criteria:
            ctype = c.get("criterion_type", "")
            style = "green" if ctype == "inclusion" else "red"

            table.add_row(
                f"[{style}]{ctype.title()}[/{style}]",
                str(c.get("criterion_number", "")),
                c.get("criterion_text", "")[:150],
                c.get("category", "") or "-",
            )

        return table


def format_comparison(
    result,
    format_type: str = "table",
) -> Union[str, Table]:
    """Format protocol comparison results."""
    if format_type == "json":
        return json.dumps({
            "protocols": result.protocols,
            "aspect": result.aspect,
            "summary": result.summary,
        }, indent=2)

    elif format_type == "markdown":
        lines = [
            f"# Protocol Comparison: {', '.join(result.protocols)}\n",
            f"**Aspect:** {result.aspect}\n",
            "## Summary\n",
            result.summary,
        ]
        return "\n".join(lines)

    else:  # table/text format
        console = Console()
        panel = Panel(
            result.summary,
            title=f"Comparison: {', '.join(result.protocols)}",
            subtitle=f"Aspect: {result.aspect}",
        )
        return panel


def format_eligibility_comparison(
    result,
    format_type: str = "table",
) -> Union[str, Table]:
    """Format eligibility comparison results."""
    if format_type == "json":
        # Remove embeddings from output
        criteria_clean = {}
        for pid, criteria in result.criteria_by_protocol.items():
            criteria_clean[pid] = [
                {k: v for k, v in c.items() if k != "embedding"}
                for c in criteria
            ]

        return json.dumps({
            "protocols": result.protocols,
            "criteria_by_protocol": criteria_clean,
            "similar_criteria": result.similar_criteria,
            "unique_criteria": result.unique_criteria,
            "summary": result.summary,
        }, indent=2, default=str)

    elif format_type == "markdown":
        lines = [
            f"# Eligibility Comparison: {', '.join(result.protocols)}\n",
        ]

        if result.summary:
            lines.append(f"{result.summary}\n")

        if result.similar_criteria:
            lines.append("## Similar Criteria\n")
            for pair in result.similar_criteria[:10]:
                sim = pair.get("similarity", 0)
                lines.append(f"**Similarity: {sim:.2%}**")
                for c in pair.get("criteria", []):
                    lines.append(f"- {c.get('protocol_id')}: {c.get('criterion_text', '')[:100]}")
                lines.append("")

        if result.unique_criteria:
            lines.append("## Unique Criteria\n")
            for pid, criteria in result.unique_criteria.items():
                lines.append(f"### {pid}")
                for c in criteria[:5]:
                    lines.append(f"- {c.get('criterion_text', '')[:100]}")
                lines.append("")

        return "\n".join(lines)

    else:  # table format
        table = Table(title="Eligibility Comparison")
        table.add_column("Protocol", style="cyan")
        table.add_column("Type")
        table.add_column("Criterion", max_width=70)
        table.add_column("Match")

        # Mark criteria that have matches
        matched_ids = set()
        for pair in result.similar_criteria:
            for c in pair.get("criteria", []):
                matched_ids.add((c.get("protocol_id"), c.get("id")))

        for pid, criteria in result.criteria_by_protocol.items():
            for c in criteria[:10]:  # Limit display
                is_matched = (pid, c.get("id")) in matched_ids
                match_indicator = "[green]Yes[/green]" if is_matched else "[dim]-[/dim]"

                table.add_row(
                    pid,
                    c.get("criterion_type", "")[:3].upper(),
                    c.get("criterion_text", "")[:100],
                    match_indicator,
                )

        return table

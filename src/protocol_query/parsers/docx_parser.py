"""Word document parser."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional

from docx import Document
from docx.opc.exceptions import PackageNotFoundError


def parse_docx(file_path: Path) -> dict:
    """
    Parse a Word document and extract text and metadata.
    """
    file_path = Path(file_path).resolve()

    # Calculate file hash
    with open(file_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    try:
        doc = Document(file_path)
    except PackageNotFoundError:
        raise ValueError(f"Could not open Word document: {file_path}")

    # Extract metadata
    metadata = _extract_metadata(doc)

    # Extract text and sections
    paragraphs, sections = _extract_text_and_sections(doc)

    # Full text for protocol info extraction
    full_text = "\n".join(paragraphs)
    protocol_info = _extract_protocol_info(full_text, metadata)

    return {
        "filename": file_path.name,
        "filepath": str(file_path),
        "file_hash": file_hash,
        "file_type": "docx",
        "title": protocol_info.get("title") or metadata.get("title"),
        "protocol_id": protocol_info.get("protocol_id"),
        "version": protocol_info.get("version"),
        "sponsor": protocol_info.get("sponsor"),
        "indication": protocol_info.get("indication"),
        "phase": protocol_info.get("phase"),
        "pages": [full_text],  # Word doesn't have page boundaries easily
        "sections": sections,
        "metadata": metadata,
    }


def _extract_metadata(doc: Document) -> dict:
    """Extract document metadata."""
    props = doc.core_properties
    return {
        "title": props.title,
        "author": props.author,
        "subject": props.subject,
        "created": str(props.created) if props.created else None,
        "modified": str(props.modified) if props.modified else None,
        "revision": props.revision,
    }


def _extract_text_and_sections(doc: Document) -> tuple[list[str], list[dict]]:
    """Extract paragraphs and detect sections from document structure."""
    paragraphs = []
    sections = []
    current_section = None
    section_index = 0

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        paragraphs.append(text)

        # Check if this is a heading (section header)
        if para.style and para.style.name.startswith("Heading"):
            level = _get_heading_level(para.style.name)
            section_type = _classify_section_from_title(text)

            if current_section:
                sections.append(current_section)
                section_index += 1

            current_section = {
                "index": section_index,
                "section_type": section_type,
                "section_number": _extract_section_number(text),
                "title": text,
                "level": level,
                "start_page": None,  # Word doesn't easily expose page numbers
                "raw_text": "",
            }
        elif current_section:
            current_section["raw_text"] += text + "\n"

    # Handle documents without headings - try pattern matching
    if not sections:
        sections = _detect_sections_from_text(paragraphs)

    # Close final section
    if current_section and current_section not in sections:
        sections.append(current_section)

    return paragraphs, sections


def _get_heading_level(style_name: str) -> int:
    """Extract heading level from style name."""
    if "Heading" in style_name:
        try:
            return int(style_name.replace("Heading", "").strip())
        except ValueError:
            return 1
    return 0


def _extract_section_number(title: str) -> str | None:
    """Extract section number from title."""
    match = re.match(r"^(\d+(?:\.\d+)*)\s*\.?\s*", title)
    return match.group(1) if match else None


def _classify_section_from_title(title: str) -> str:
    """Classify section type from its title."""
    title_lower = title.lower()

    if "inclusion" in title_lower:
        return "inclusion_criteria"
    elif "exclusion" in title_lower:
        return "exclusion_criteria"
    elif "eligibility" in title_lower:
        return "population"
    elif "objective" in title_lower:
        return "objectives"
    elif "background" in title_lower or "introduction" in title_lower:
        return "background"
    elif "design" in title_lower:
        return "study_design"
    elif "treatment" in title_lower or "intervention" in title_lower:
        return "treatment"
    elif "assessment" in title_lower or "procedure" in title_lower:
        return "assessments"
    elif "safety" in title_lower or "adverse" in title_lower:
        return "safety"
    elif "efficacy" in title_lower or "endpoint" in title_lower:
        return "efficacy"
    elif "statistic" in title_lower:
        return "statistics"
    elif "ethic" in title_lower:
        return "ethics"
    elif "admin" in title_lower:
        return "administration"
    elif "appendix" in title_lower:
        return "appendix"
    else:
        return "other"


def _detect_sections_from_text(paragraphs: list[str]) -> list[dict]:
    """Detect sections using text patterns when document has no headings."""
    sections = []
    section_index = 0

    # Patterns for section headers
    header_pattern = re.compile(
        r"^(\d+(?:\.\d+)*)\s*\.?\s*(INTRODUCTION|BACKGROUND|OBJECTIVES?|"
        r"STUDY DESIGN|POPULATION|ELIGIBILITY|INCLUSION|EXCLUSION|"
        r"TREATMENT|PROCEDURES?|ASSESSMENTS?|SAFETY|EFFICACY|"
        r"STATISTICAL|ETHICS|ADMINISTRATION|REFERENCES?|APPENDIX)",
        re.IGNORECASE,
    )

    current_section = None

    for para in paragraphs:
        match = header_pattern.match(para)
        if match:
            if current_section:
                sections.append(current_section)
                section_index += 1

            current_section = {
                "index": section_index,
                "section_type": _classify_section_from_title(para),
                "section_number": match.group(1),
                "title": para,
                "level": para.count("."),
                "start_page": None,
                "raw_text": "",
            }
        elif current_section:
            current_section["raw_text"] += para + "\n"

    if current_section:
        sections.append(current_section)

    return sections


def _extract_protocol_info(text: str, metadata: dict) -> dict:
    """Extract protocol-specific information from text."""
    info = {}

    # Protocol ID patterns
    nct_match = re.search(r"(NCT\d{8})", text)
    if nct_match:
        info["protocol_id"] = nct_match.group(1)
    else:
        proto_match = re.search(
            r"Protocol\s+(?:No\.?|Number|ID)?:?\s*([A-Z0-9]+-?[A-Z0-9]+)",
            text,
            re.IGNORECASE,
        )
        if proto_match:
            info["protocol_id"] = proto_match.group(1)

    # Version
    version_match = re.search(
        r"Version:?\s*(\d+(?:\.\d+)?)|Amendment\s*(\d+)",
        text,
        re.IGNORECASE,
    )
    if version_match:
        info["version"] = version_match.group(1) or version_match.group(2)

    # Sponsor
    sponsor_match = re.search(r"Sponsor:?\s*([A-Z][A-Za-z\s&,]+?)(?:\n|$)", text)
    if sponsor_match:
        info["sponsor"] = sponsor_match.group(1).strip()

    # Phase
    phase_match = re.search(
        r"Phase\s*(I{1,3}V?|[1-4])[/\s]*(I{1,3}V?|[1-4])?",
        text,
        re.IGNORECASE,
    )
    if phase_match:
        phase = phase_match.group(1)
        if phase_match.group(2):
            phase += "/" + phase_match.group(2)
        info["phase"] = f"Phase {phase}"

    # Title
    if metadata.get("title"):
        info["title"] = metadata["title"]
    else:
        title_patterns = [
            r"^(.+?Protocol.+?)$",
            r"^(A\s+.+?Study.+?)$",
        ]
        for pattern in title_patterns:
            match = re.search(pattern, text[:2000], re.MULTILINE | re.IGNORECASE)
            if match:
                info["title"] = match.group(1).strip()[:200]
                break

    return info

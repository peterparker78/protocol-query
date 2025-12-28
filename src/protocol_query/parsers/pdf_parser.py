"""PDF document parser."""

import hashlib
import re
from pathlib import Path
from typing import Optional

import pdfplumber
from pypdf import PdfReader


def parse_pdf(file_path: Path) -> dict:
    """
    Parse a PDF document and extract text and metadata.

    Uses pypdf for metadata and pdfplumber for text extraction
    (better handling of tables and layout).
    """
    file_path = Path(file_path).resolve()

    # Calculate file hash
    with open(file_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    # Extract metadata with pypdf
    metadata = _extract_metadata(file_path)

    # Extract text with pdfplumber (better for tables/layout)
    pages, sections = _extract_text_and_sections(file_path)

    # Try to extract protocol-specific info
    full_text = "\n".join(pages)
    protocol_info = _extract_protocol_info(full_text, metadata)

    return {
        "filename": file_path.name,
        "filepath": str(file_path),
        "file_hash": file_hash,
        "file_type": "pdf",
        "title": protocol_info.get("title") or metadata.get("title"),
        "protocol_id": protocol_info.get("protocol_id"),
        "version": protocol_info.get("version"),
        "sponsor": protocol_info.get("sponsor"),
        "indication": protocol_info.get("indication"),
        "phase": protocol_info.get("phase"),
        "pages": pages,
        "sections": sections,
        "metadata": metadata,
    }


def _extract_metadata(file_path: Path) -> dict:
    """Extract PDF metadata using pypdf."""
    try:
        reader = PdfReader(file_path)
        meta = reader.metadata or {}
        return {
            "title": meta.get("/Title"),
            "author": meta.get("/Author"),
            "subject": meta.get("/Subject"),
            "creator": meta.get("/Creator"),
            "producer": meta.get("/Producer"),
            "creation_date": str(meta.get("/CreationDate", "")),
            "page_count": len(reader.pages),
        }
    except Exception:
        return {}


def _extract_text_and_sections(file_path: Path) -> tuple[list[str], list[dict]]:
    """Extract text page by page and identify sections."""
    pages = []
    sections = []
    current_section = None
    section_index = 0

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            pages.append(text)

            # Detect section headers
            detected_sections = _detect_sections(text, page_num)
            for sec in detected_sections:
                if current_section:
                    current_section["end_page"] = page_num - 1
                    sections.append(current_section)
                    section_index += 1

                current_section = {
                    "index": section_index,
                    "section_type": sec["type"],
                    "section_number": sec.get("number"),
                    "title": sec["title"],
                    "level": sec.get("level", 0),
                    "start_page": page_num,
                    "raw_text": "",
                }

            # Accumulate text in current section
            if current_section:
                current_section["raw_text"] += text + "\n"

    # Close final section
    if current_section:
        current_section["end_page"] = len(pages)
        sections.append(current_section)

    return pages, sections


def _detect_sections(text: str, page_num: int) -> list[dict]:
    """Detect section headers in text."""
    sections = []

    # Common clinical protocol section patterns
    patterns = [
        # Numbered sections like "1. INTRODUCTION" or "1 INTRODUCTION"
        (
            r"^(\d+(?:\.\d+)*)\s*\.?\s*(INTRODUCTION|BACKGROUND|OBJECTIVES?|"
            r"STUDY DESIGN|POPULATION|ELIGIBILITY|INCLUSION|EXCLUSION|"
            r"TREATMENT|PROCEDURES?|ASSESSMENTS?|SAFETY|EFFICACY|"
            r"STATISTICAL|ETHICS|ADMINISTRATION|REFERENCES?|APPENDIX)",
            "numbered",
        ),
        # Standalone section headers
        (
            r"^(INCLUSION CRITERIA|EXCLUSION CRITERIA|ELIGIBILITY CRITERIA)",
            "eligibility",
        ),
        (r"^(STUDY OBJECTIVES?|PRIMARY OBJECTIVES?|SECONDARY OBJECTIVES?)", "objectives"),
        (r"^(STUDY DESIGN|TRIAL DESIGN)", "design"),
        (r"^(SAFETY ASSESSMENTS?|ADVERSE EVENTS?)", "safety"),
    ]

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        for pattern, section_category in patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                section_type = _classify_section(line, section_category)
                sections.append(
                    {
                        "type": section_type,
                        "number": match.group(1) if section_category == "numbered" else None,
                        "title": line,
                        "level": line.count(".") if section_category == "numbered" else 0,
                    }
                )
                break

    return sections


def _classify_section(title: str, category: str) -> str:
    """Classify section into standard types."""
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


def _extract_protocol_info(text: str, metadata: dict) -> dict:
    """Extract protocol-specific information from text."""
    info = {}

    # Protocol ID patterns (NCT number, sponsor protocol numbers)
    nct_match = re.search(r"(NCT\d{8})", text)
    if nct_match:
        info["protocol_id"] = nct_match.group(1)
    else:
        # Try other protocol ID patterns
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
    sponsor_match = re.search(
        r"Sponsor:?\s*([A-Z][A-Za-z\s&,]+?)(?:\n|$)",
        text,
    )
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

    # Title - try to extract from first page
    title_patterns = [
        r"^(.+?Protocol.+?)$",
        r"^(A\s+.+?Study.+?)$",
        r"^(Phase\s+.+?Study.+?)$",
    ]
    for pattern in title_patterns:
        match = re.search(pattern, text[:2000], re.MULTILINE | re.IGNORECASE)
        if match:
            info["title"] = match.group(1).strip()[:200]
            break

    return info

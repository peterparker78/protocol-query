"""Document parsing module."""

from pathlib import Path
from typing import Union

from protocol_query.parsers.pdf_parser import parse_pdf
from protocol_query.parsers.docx_parser import parse_docx


def parse_document(file_path: Union[str, Path]) -> dict:
    """Parse a document based on its file type."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return parse_pdf(path)
    elif suffix in (".docx", ".doc"):
        return parse_docx(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


__all__ = ["parse_document", "parse_pdf", "parse_docx"]

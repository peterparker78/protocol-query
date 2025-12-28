"""Protocol-aware document chunking."""

import re
from dataclasses import dataclass, field
from typing import Optional

from protocol_query.core.config import Config


@dataclass
class Chunk:
    """A text chunk with metadata."""

    chunk_text: str
    chunk_type: str = "text"  # 'text', 'criterion', 'table', 'list'
    section_type: Optional[str] = None
    section_index: Optional[int] = None
    page_number: Optional[int] = None
    criterion_type: Optional[str] = None  # 'inclusion' or 'exclusion'
    criterion_number: Optional[int] = None
    category: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class ProtocolChunker:
    """
    Specialized chunker for clinical protocols.

    Strategy:
    1. Eligibility criteria: One chunk per criterion
    2. Other sections: Semantic chunking with overlap
    3. Target chunk size: 512 tokens (~400 words)
    """

    def __init__(self, config: Config):
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap

    def chunk_document(self, doc_data: dict) -> list[dict]:
        """
        Chunk a parsed document.

        Returns list of chunk dictionaries.
        """
        chunks = []
        sections = doc_data.get("sections", [])

        if sections:
            for section in sections:
                section_chunks = self._chunk_section(section)
                chunks.extend(section_chunks)
        else:
            # No sections detected - chunk the full text
            pages = doc_data.get("pages", [])
            full_text = "\n".join(pages)
            text_chunks = self._chunk_text(full_text, "other", None)
            chunks.extend(text_chunks)

        return [self._chunk_to_dict(c) for c in chunks]

    def _chunk_section(self, section: dict) -> list[Chunk]:
        """Chunk a single section based on its type."""
        section_type = section.get("section_type", "other")
        section_index = section.get("index")
        raw_text = section.get("raw_text", "")

        if not raw_text.strip():
            return []

        if section_type in ("inclusion_criteria", "exclusion_criteria"):
            return self._chunk_criteria(
                raw_text,
                section_type,
                section_index,
            )
        else:
            return self._chunk_text(raw_text, section_type, section_index)

    def _chunk_criteria(
        self,
        text: str,
        section_type: str,
        section_index: Optional[int],
    ) -> list[Chunk]:
        """
        Extract individual eligibility criteria as separate chunks.
        """
        chunks = []
        criterion_type = "inclusion" if "inclusion" in section_type else "exclusion"

        # Pattern to match numbered criteria
        # Matches: "1.", "1)", "(1)", "a.", "a)", "(a)", bullets, dashes
        criterion_pattern = re.compile(
            r"(?:^|\n)\s*"
            r"(?:(\d+)[.\)]\s*|"  # 1. or 1)
            r"\((\d+)\)\s*|"  # (1)
            r"([a-z])[.\)]\s*|"  # a. or a)
            r"\(([a-z])\)\s*|"  # (a)
            r"[\u2022\u2023\u25E6\u2043\u2219•]\s*|"  # bullets
            r"[-–—]\s*)"  # dashes
            r"(.+?)(?=(?:\n\s*(?:\d+[.\)]|\(\d+\)|[a-z][.\)]|\([a-z]\)|[\u2022\u2023\u25E6\u2043\u2219•]|[-–—]))|$)",
            re.DOTALL,
        )

        matches = criterion_pattern.findall(text)

        if matches:
            for i, match in enumerate(matches):
                # Extract the number and text
                num1, num2, letter1, letter2, criterion_text = match
                criterion_num = None
                if num1:
                    criterion_num = int(num1)
                elif num2:
                    criterion_num = int(num2)
                elif letter1:
                    criterion_num = ord(letter1.lower()) - ord("a") + 1
                elif letter2:
                    criterion_num = ord(letter2.lower()) - ord("a") + 1
                else:
                    criterion_num = i + 1

                criterion_text = criterion_text.strip()
                if criterion_text and len(criterion_text) > 10:
                    chunk = Chunk(
                        chunk_text=criterion_text,
                        chunk_type="criterion",
                        section_type=section_type,
                        section_index=section_index,
                        criterion_type=criterion_type,
                        criterion_number=criterion_num,
                        category=self._categorize_criterion(criterion_text),
                    )
                    chunks.append(chunk)
        else:
            # Fallback: split by lines that look like criteria
            lines = text.split("\n")
            criterion_num = 0
            for line in lines:
                line = line.strip()
                if line and len(line) > 20:
                    criterion_num += 1
                    chunk = Chunk(
                        chunk_text=line,
                        chunk_type="criterion",
                        section_type=section_type,
                        section_index=section_index,
                        criterion_type=criterion_type,
                        criterion_number=criterion_num,
                        category=self._categorize_criterion(line),
                    )
                    chunks.append(chunk)

        return chunks

    def _chunk_text(
        self,
        text: str,
        section_type: str,
        section_index: Optional[int],
    ) -> list[Chunk]:
        """
        Chunk narrative text with sentence awareness and overlap.
        """
        chunks = []

        # Split into sentences (rough approximation)
        sentences = re.split(r"(?<=[.!?])\s+", text)

        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Rough token count (words * 1.3)
            sentence_tokens = len(sentence.split()) * 1.3

            if current_length + sentence_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(
                    Chunk(
                        chunk_text=chunk_text,
                        chunk_type="text",
                        section_type=section_type,
                        section_index=section_index,
                    )
                )

                # Start new chunk with overlap
                overlap_sentences = []
                overlap_tokens = 0
                for s in reversed(current_chunk):
                    s_tokens = len(s.split()) * 1.3
                    if overlap_tokens + s_tokens <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_tokens += s_tokens
                    else:
                        break

                current_chunk = overlap_sentences
                current_length = overlap_tokens

            current_chunk.append(sentence)
            current_length += sentence_tokens

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(
                Chunk(
                    chunk_text=chunk_text,
                    chunk_type="text",
                    section_type=section_type,
                    section_index=section_index,
                )
            )

        return chunks

    def _categorize_criterion(self, text: str) -> Optional[str]:
        """Categorize an eligibility criterion."""
        text_lower = text.lower()

        if any(
            term in text_lower
            for term in ["age", "year", "old", "adult", "pediatric", "elderly"]
        ):
            return "demographic"
        elif any(
            term in text_lower
            for term in ["male", "female", "gender", "sex", "pregnant", "nursing"]
        ):
            return "demographic"
        elif any(
            term in text_lower
            for term in [
                "diagnosis",
                "confirmed",
                "histolog",
                "patholog",
                "disease",
                "condition",
            ]
        ):
            return "clinical"
        elif any(
            term in text_lower
            for term in [
                "lab",
                "laboratory",
                "hemoglobin",
                "creatinine",
                "bilirubin",
                "ast",
                "alt",
                "wbc",
                "platelet",
            ]
        ):
            return "laboratory"
        elif any(
            term in text_lower
            for term in [
                "prior",
                "previous",
                "therapy",
                "treatment",
                "medication",
                "drug",
            ]
        ):
            return "prior_treatment"
        elif any(term in text_lower for term in ["consent", "willing", "able to"]):
            return "consent"
        elif any(
            term in text_lower for term in ["contraception", "birth control", "fertile"]
        ):
            return "reproductive"
        else:
            return None

    def _chunk_to_dict(self, chunk: Chunk) -> dict:
        """Convert Chunk dataclass to dictionary."""
        return {
            "chunk_text": chunk.chunk_text,
            "chunk_type": chunk.chunk_type,
            "section_type": chunk.section_type,
            "section_index": chunk.section_index,
            "page_number": chunk.page_number,
            "criterion_type": chunk.criterion_type,
            "criterion_number": chunk.criterion_number,
            "category": chunk.category,
            "metadata": chunk.metadata,
        }

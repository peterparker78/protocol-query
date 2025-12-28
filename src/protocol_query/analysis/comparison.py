"""Protocol comparison engine."""

from dataclasses import dataclass, field
from typing import Optional

from protocol_query.core.database import Database
from protocol_query.core.config import get_config
from protocol_query.embeddings.local import LocalEmbeddings
from protocol_query.analysis.llm import ClaudeLLM


@dataclass
class ComparisonResult:
    """Result of a protocol comparison."""

    protocols: list[str]
    aspect: str
    summary: str
    details: dict = field(default_factory=dict)
    similarities: list[dict] = field(default_factory=list)
    differences: list[dict] = field(default_factory=list)


@dataclass
class EligibilityComparisonResult:
    """Result of eligibility criteria comparison."""

    protocols: list[str]
    criteria_by_protocol: dict
    similar_criteria: list[dict] = field(default_factory=list)
    unique_criteria: dict = field(default_factory=dict)
    summary: str = ""


class ProtocolComparer:
    """
    Multi-protocol comparison engine.

    Compares:
    - Eligibility criteria (semantic matching)
    - Study design
    - Safety monitoring
    - Endpoints
    """

    SIMILARITY_THRESHOLD = 0.85

    def __init__(self, db: Database):
        self.db = db
        self.config = get_config()
        self.embedder = LocalEmbeddings()
        self._llm: Optional[ClaudeLLM] = None

    @property
    def llm(self) -> ClaudeLLM:
        """Lazy initialize LLM."""
        if self._llm is None:
            self._llm = ClaudeLLM(self.config)
        return self._llm

    def compare(
        self,
        protocol_ids: list[str],
        aspect: str = "all",
    ) -> ComparisonResult:
        """
        Compare multiple protocols.

        Args:
            protocol_ids: List of protocol IDs to compare
            aspect: Aspect to compare (all, eligibility, design, safety)
        """
        # Get protocol data
        protocols_data = {}
        for pid in protocol_ids:
            doc = self.db.get_document_by_protocol_id(pid)
            if doc:
                protocols_data[pid] = self._get_protocol_summary(doc["id"])

        if not protocols_data:
            return ComparisonResult(
                protocols=protocol_ids,
                aspect=aspect,
                summary="No protocols found",
            )

        # Build context for LLM comparison
        context_parts = {}
        for pid, data in protocols_data.items():
            context_parts[pid] = self._format_protocol_context(data, aspect)

        # Use LLM for detailed comparison if API key available
        if self.config.anthropic_api_key:
            question = self._build_comparison_question(aspect)
            summary = self.llm.compare(question, context_parts)
        else:
            summary = self._basic_comparison(protocols_data, aspect)

        return ComparisonResult(
            protocols=protocol_ids,
            aspect=aspect,
            summary=summary,
            details=protocols_data,
        )

    def compare_eligibility(
        self,
        protocol_ids: list[str],
        criterion_type: str = "all",
    ) -> EligibilityComparisonResult:
        """
        Compare eligibility criteria across protocols.

        Uses semantic similarity to find matching criteria.
        """
        # Fetch criteria for each protocol
        criteria_by_protocol = {}
        for pid in protocol_ids:
            doc = self.db.get_document_by_protocol_id(pid)
            if doc:
                criteria_by_protocol[pid] = self._get_criteria(
                    doc["id"], criterion_type
                )

        if not criteria_by_protocol:
            return EligibilityComparisonResult(
                protocols=protocol_ids,
                criteria_by_protocol={},
            )

        # Generate embeddings for all criteria
        all_criteria = []
        for pid, criteria in criteria_by_protocol.items():
            for c in criteria:
                c["protocol_id"] = pid
                all_criteria.append(c)

        if not all_criteria:
            return EligibilityComparisonResult(
                protocols=protocol_ids,
                criteria_by_protocol=criteria_by_protocol,
            )

        # Compute embeddings
        texts = [c["criterion_text"] for c in all_criteria]
        embeddings = self.embedder.embed_batch(texts)
        for c, emb in zip(all_criteria, embeddings):
            c["embedding"] = emb

        # Find similar criteria across protocols
        similar_pairs = self._find_similar_criteria(all_criteria)

        # Identify unique criteria
        matched_ids = set()
        for pair in similar_pairs:
            matched_ids.add((pair["criteria"][0]["id"], pair["criteria"][0]["protocol_id"]))
            matched_ids.add((pair["criteria"][1]["id"], pair["criteria"][1]["protocol_id"]))

        unique_criteria = {}
        for pid, criteria in criteria_by_protocol.items():
            unique = [
                c for c in criteria
                if (c["id"], pid) not in matched_ids
            ]
            if unique:
                unique_criteria[pid] = unique

        # Generate summary if LLM available
        summary = ""
        if self.config.anthropic_api_key:
            summary = self._generate_eligibility_summary(
                criteria_by_protocol, similar_pairs, unique_criteria
            )

        return EligibilityComparisonResult(
            protocols=protocol_ids,
            criteria_by_protocol=criteria_by_protocol,
            similar_criteria=similar_pairs,
            unique_criteria=unique_criteria,
            summary=summary,
        )

    def _get_protocol_summary(self, doc_id: int) -> dict:
        """Get summary data for a protocol."""
        with self.db.cursor() as cur:
            # Get sections
            cur.execute(
                """
                SELECT section_type, title, raw_text
                FROM sections WHERE document_id = ?
                """,
                (doc_id,),
            )
            sections = {row["section_type"]: dict(row) for row in cur.fetchall()}

            # Get criteria counts
            cur.execute(
                """
                SELECT criterion_type, COUNT(*) as count
                FROM eligibility_criteria WHERE document_id = ?
                GROUP BY criterion_type
                """,
                (doc_id,),
            )
            criteria_counts = {row["criterion_type"]: row["count"] for row in cur.fetchall()}

            return {
                "sections": sections,
                "criteria_counts": criteria_counts,
            }

    def _get_criteria(self, doc_id: int, criterion_type: str) -> list[dict]:
        """Get eligibility criteria for a document."""
        with self.db.cursor() as cur:
            if criterion_type == "all":
                cur.execute(
                    """
                    SELECT * FROM eligibility_criteria
                    WHERE document_id = ?
                    ORDER BY criterion_type, criterion_number
                    """,
                    (doc_id,),
                )
            else:
                cur.execute(
                    """
                    SELECT * FROM eligibility_criteria
                    WHERE document_id = ? AND criterion_type = ?
                    ORDER BY criterion_number
                    """,
                    (doc_id, criterion_type),
                )
            return [dict(row) for row in cur.fetchall()]

    def _find_similar_criteria(self, all_criteria: list[dict]) -> list[dict]:
        """Find semantically similar criteria across protocols."""
        import math

        similar_pairs = []

        for i, c1 in enumerate(all_criteria):
            for c2 in all_criteria[i + 1:]:
                # Only compare across different protocols
                if c1["protocol_id"] == c2["protocol_id"]:
                    continue

                # Compute cosine similarity
                similarity = self._cosine_similarity(c1["embedding"], c2["embedding"])

                if similarity >= self.SIMILARITY_THRESHOLD:
                    similar_pairs.append({
                        "criteria": [
                            {k: v for k, v in c1.items() if k != "embedding"},
                            {k: v for k, v in c2.items() if k != "embedding"},
                        ],
                        "similarity": similarity,
                        "protocols": [c1["protocol_id"], c2["protocol_id"]],
                    })

        return similar_pairs

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity."""
        import math

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def _format_protocol_context(self, data: dict, aspect: str) -> str:
        """Format protocol data as context string."""
        parts = []
        sections = data.get("sections", {})

        if aspect in ("all", "eligibility"):
            if "inclusion_criteria" in sections:
                parts.append(f"**Inclusion Criteria:**\n{sections['inclusion_criteria'].get('raw_text', '')[:1000]}")
            if "exclusion_criteria" in sections:
                parts.append(f"**Exclusion Criteria:**\n{sections['exclusion_criteria'].get('raw_text', '')[:1000]}")

        if aspect in ("all", "design"):
            if "study_design" in sections:
                parts.append(f"**Study Design:**\n{sections['study_design'].get('raw_text', '')[:1000]}")
            if "objectives" in sections:
                parts.append(f"**Objectives:**\n{sections['objectives'].get('raw_text', '')[:1000]}")

        if aspect in ("all", "safety"):
            if "safety" in sections:
                parts.append(f"**Safety:**\n{sections['safety'].get('raw_text', '')[:1000]}")

        return "\n\n".join(parts) if parts else "No relevant sections found."

    def _build_comparison_question(self, aspect: str) -> str:
        """Build comparison question based on aspect."""
        questions = {
            "all": "Compare these protocols across eligibility criteria, study design, and key differences.",
            "eligibility": "Compare the eligibility criteria (inclusion and exclusion) across these protocols.",
            "design": "Compare the study design and objectives across these protocols.",
            "safety": "Compare the safety monitoring and requirements across these protocols.",
        }
        return questions.get(aspect, questions["all"])

    def _basic_comparison(self, protocols_data: dict, aspect: str) -> str:
        """Generate basic comparison without LLM."""
        lines = ["## Protocol Comparison\n"]

        for pid, data in protocols_data.items():
            lines.append(f"### {pid}")
            counts = data.get("criteria_counts", {})
            lines.append(f"- Inclusion criteria: {counts.get('inclusion', 0)}")
            lines.append(f"- Exclusion criteria: {counts.get('exclusion', 0)}")
            lines.append(f"- Sections: {len(data.get('sections', {}))}")
            lines.append("")

        return "\n".join(lines)

    def _generate_eligibility_summary(
        self,
        criteria_by_protocol: dict,
        similar_pairs: list[dict],
        unique_criteria: dict,
    ) -> str:
        """Generate summary of eligibility comparison."""
        parts = [f"Found {len(similar_pairs)} similar criteria across protocols."]

        for pid, unique in unique_criteria.items():
            parts.append(f"{pid} has {len(unique)} unique criteria.")

        return " ".join(parts)

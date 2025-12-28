"""What-if analysis for clinical protocols."""

from dataclasses import dataclass, field
from typing import Optional

from protocol_query.core.config import Config
from protocol_query.core.database import Database
from protocol_query.search.hybrid import HybridSearch, SearchResult
from protocol_query.analysis.llm import ClaudeLLM


@dataclass
class WhatIfResult:
    """Result of a what-if analysis."""

    scenario: str
    protocol_id: str
    analysis: str
    relevant_chunks: list[dict] = field(default_factory=list)
    affected_criteria: list[dict] = field(default_factory=list)


class WhatIfAnalyzer:
    """
    Handles "what if" questions about protocols.

    Examples:
    - "What if a patient has diabetes?"
    - "What if we need to modify the dosing schedule?"
    - "What if a patient misses a visit?"
    """

    def __init__(
        self,
        search: HybridSearch,
        db: Database,
        config: Config,
    ):
        self.search = search
        self.db = db
        self.config = config
        self.llm = ClaudeLLM(config)

    def analyze(self, scenario: str, protocol_id: str) -> WhatIfResult:
        """
        Analyze a what-if scenario against a specific protocol.
        """
        # 1. Classify the scenario and identify key terms
        scenario_type = self._classify_scenario(scenario)
        relevant_sections = self._get_relevant_sections(scenario_type)

        # 2. Search for relevant content
        search_results = self.search.search(
            query=scenario,
            limit=15,
            protocol_ids=[protocol_id],
            section_types=relevant_sections,
        )

        # 3. Get any related eligibility criteria
        affected_criteria = self._find_affected_criteria(scenario, protocol_id)

        # 4. Build context for LLM
        context = self._build_context(search_results, affected_criteria)

        # 5. Generate analysis with LLM
        analysis = self.llm.analyze(
            question=f"What if {scenario}?",
            context=context,
            system_prompt=self._what_if_system_prompt(),
        )

        return WhatIfResult(
            scenario=scenario,
            protocol_id=protocol_id,
            analysis=analysis,
            relevant_chunks=[self._result_to_dict(r) for r in search_results[:5]],
            affected_criteria=affected_criteria,
        )

    def _classify_scenario(self, scenario: str) -> str:
        """Classify the scenario type."""
        scenario_lower = scenario.lower()

        if any(
            term in scenario_lower
            for term in [
                "eligible",
                "qualify",
                "criteria",
                "patient has",
                "history of",
                "diagnosis",
            ]
        ):
            return "eligibility"
        elif any(
            term in scenario_lower
            for term in ["adverse", "side effect", "safety", "toxicity", "reaction"]
        ):
            return "safety"
        elif any(
            term in scenario_lower
            for term in ["dose", "dosing", "schedule", "frequency", "mg", "modify"]
        ):
            return "dosing"
        elif any(
            term in scenario_lower
            for term in ["visit", "miss", "appointment", "procedure", "skip"]
        ):
            return "procedure"
        elif any(
            term in scenario_lower
            for term in ["withdraw", "discontinue", "stop", "terminate"]
        ):
            return "discontinuation"
        else:
            return "general"

    def _get_relevant_sections(self, scenario_type: str) -> Optional[list[str]]:
        """Map scenario type to relevant protocol sections."""
        mapping = {
            "eligibility": [
                "inclusion_criteria",
                "exclusion_criteria",
                "population",
            ],
            "safety": ["safety", "assessments", "treatment"],
            "dosing": ["treatment", "study_design", "safety"],
            "procedure": ["assessments", "study_design", "administration"],
            "discontinuation": ["treatment", "safety", "administration"],
            "general": None,  # Search all sections
        }
        return mapping.get(scenario_type)

    def _find_affected_criteria(
        self,
        scenario: str,
        protocol_id: str,
    ) -> list[dict]:
        """Find eligibility criteria that might be affected by the scenario."""
        doc = self.db.get_document_by_protocol_id(protocol_id)
        if not doc:
            return []

        # Search criteria using hybrid search
        criteria_results = self.search.search(
            query=scenario,
            limit=10,
            protocol_ids=[protocol_id],
            section_types=["inclusion_criteria", "exclusion_criteria"],
        )

        affected = []
        with self.db.cursor() as cur:
            for result in criteria_results:
                cur.execute(
                    """
                    SELECT * FROM eligibility_criteria
                    WHERE chunk_id = ?
                    """,
                    (result.chunk_id,),
                )
                row = cur.fetchone()
                if row:
                    affected.append(dict(row))

        return affected

    def _build_context(
        self,
        search_results: list[SearchResult],
        criteria: list[dict],
    ) -> str:
        """Build context string for LLM."""
        parts = []

        # Add search results
        if search_results:
            parts.append("## Relevant Protocol Sections\n")
            for i, result in enumerate(search_results, 1):
                section = result.section_type or "Unknown Section"
                parts.append(f"### {i}. {section}\n{result.chunk_text}\n")

        # Add affected criteria
        if criteria:
            parts.append("\n## Potentially Affected Eligibility Criteria\n")
            for c in criteria:
                ctype = "Inclusion" if c["criterion_type"] == "inclusion" else "Exclusion"
                parts.append(f"- **{ctype} #{c.get('criterion_number', '?')}**: {c['criterion_text']}")

        return "\n".join(parts)

    def _what_if_system_prompt(self) -> str:
        return """You are a clinical research expert analyzing "what if" scenarios for clinical trial protocols.

When analyzing a scenario:
1. Identify which protocol requirements are relevant to the scenario
2. Explain the implications based on the protocol text
3. If the scenario involves eligibility, clearly state whether it would affect eligibility
4. If the scenario involves safety, highlight any specific monitoring or actions required
5. Note any ambiguities or areas where the protocol doesn't provide clear guidance

Format your response with:
- A clear summary of the implications
- Specific references to protocol sections
- Any recommended actions or considerations
- Caveats about information not found in the provided context"""

    def _result_to_dict(self, result: SearchResult) -> dict:
        """Convert SearchResult to dictionary."""
        return {
            "chunk_id": result.chunk_id,
            "section_type": result.section_type,
            "chunk_text": result.chunk_text,
            "score": result.score,
        }

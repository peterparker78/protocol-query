"""Claude LLM integration for protocol analysis."""

from typing import Optional

from anthropic import Anthropic

from protocol_query.core.config import Config


class ClaudeLLM:
    """Claude API wrapper for protocol analysis."""

    def __init__(self, config: Config):
        self.config = config
        self._client: Optional[Anthropic] = None

    @property
    def client(self) -> Anthropic:
        """Lazy initialize the Anthropic client."""
        if self._client is None:
            if not self.config.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY not configured")
            self._client = Anthropic(api_key=self.config.anthropic_api_key)
        return self._client

    def analyze(
        self,
        question: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Analyze a question given protocol context.

        Args:
            question: The user's question
            context: Relevant protocol text chunks
            system_prompt: Optional system prompt override

        Returns:
            Analysis text from Claude
        """
        if system_prompt is None:
            system_prompt = self._default_system_prompt()

        user_message = f"""Based on the following protocol excerpts, please answer this question:

**Question:** {question}

**Protocol Context:**
{context}

Please provide a clear, specific answer based only on the information in the protocol context. If the context doesn't contain enough information to fully answer the question, say so. Include specific references to sections when possible."""

        response = self.client.messages.create(
            model=self.config.llm_model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        return response.content[0].text

    def compare(
        self,
        question: str,
        protocol_contexts: dict[str, str],
    ) -> str:
        """
        Compare multiple protocols.

        Args:
            question: Comparison question
            protocol_contexts: Dict mapping protocol_id to context text

        Returns:
            Comparison analysis from Claude
        """
        context_parts = []
        for protocol_id, context in protocol_contexts.items():
            context_parts.append(f"### Protocol: {protocol_id}\n{context}")

        full_context = "\n\n".join(context_parts)

        system_prompt = """You are a clinical research expert analyzing clinical trial protocols.
Your task is to compare protocols and identify similarities and differences.
Be specific and cite which protocol each point comes from.
Format your response in a clear, structured way."""

        user_message = f"""Please compare the following protocols:

**Question:** {question}

{full_context}

Provide a structured comparison highlighting key similarities and differences."""

        response = self.client.messages.create(
            model=self.config.llm_model,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        return response.content[0].text

    def _default_system_prompt(self) -> str:
        return """You are a clinical research expert helping to analyze clinical trial protocols.

Your role is to:
1. Answer questions about protocol requirements, procedures, and eligibility
2. Analyze "what-if" scenarios and their implications
3. Provide clear, actionable guidance based on the protocol text

Guidelines:
- Be precise and cite specific sections when possible
- If information is ambiguous or missing, clearly state that
- Consider safety implications in your analysis
- Use clinical research terminology appropriately
- Format responses in clear, readable markdown"""

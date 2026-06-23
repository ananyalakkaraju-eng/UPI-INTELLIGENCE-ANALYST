"""
Agent classes — each agent wraps the RAG pipeline with a domain-specific prompt.
"""

from rag.rag import RAGPipeline


class BaseAgent:
    """Shared RAG pipeline, subclasses override the query prefix."""

    _pipeline: RAGPipeline | None = None  # shared singleton

    @classmethod
    def _get_pipeline(cls) -> RAGPipeline:
        if cls._pipeline is None:
            cls._pipeline = RAGPipeline()
        return cls._pipeline

    def _ask(self, query: str) -> str:
        return self._get_pipeline().ask(query)


class AnalyticsAgent(BaseAgent):
    """Handles general UPI analytics — CB ratios, BHIM growth, rankings."""

    def run(self, query: str) -> str:
        return self._ask(query)


class MerchantAgent(BaseAgent):
    """Handles merchant-level insights (mapped to bank data in this dataset)."""

    def run(self, entity_name: str) -> str:
        query = f"Provide detailed analysis for entity: {entity_name}"
        return self._ask(query)


class CustomerAgent(BaseAgent):
    """Handles customer / user behaviour queries."""

    def run(self, customer_name: str) -> str:
        query = f"Provide customer behaviour insights related to: {customer_name}"
        return self._ask(query)


class FraudAgent(BaseAgent):
    """Detects fraud indicators and high-risk patterns."""

    def run(self, query: str) -> str:
        fraud_query = (
            f"Identify fraud indicators, suspicious patterns, and high-risk banks. "
            f"Context: {query}"
        )
        return self._ask(fraud_query)


class ReportingAgent(BaseAgent):
    """Generates executive summaries and reports."""

    def run(self, topic: str) -> str:
        report_query = (
            f"Generate a concise executive summary with key metrics and insights. "
            f"Topic: {topic}"
        )
        return self._ask(report_query)

"""
Router — Classifies incoming queries and dispatches to the right agent.

Uses keyword matching as the primary strategy (fast, free, no API call).
Falls back to Gemini classification only when keywords are ambiguous.
"""

import os
import re
from dotenv import load_dotenv
import google.generativeai as genai

from agents.agents import (
    AnalyticsAgent,
    MerchantAgent,
    CustomerAgent,
    FraudAgent,
    ReportingAgent,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Singleton agents
# ---------------------------------------------------------------------------

_analytics = AnalyticsAgent()
_merchant = MerchantAgent()
_customer = CustomerAgent()
_fraud = FraudAgent()
_reporting = ReportingAgent()

AGENT_MAP = {
    "analytics": _analytics,
    "merchant": _merchant,
    "customer": _customer,
    "fraud": _fraud,
    "report": _reporting,
}

# ---------------------------------------------------------------------------
# Keyword rules (checked in order — first match wins)
# ---------------------------------------------------------------------------

KEYWORD_RULES = [
    ("fraud",     ["fraud", "suspicious", "anomaly", "alert", "detect", "unusual", "risk"]),
    ("merchant",  ["merchant", "vendor", "seller", "shop", "mcc"]),
    ("customer",  ["customer", "user", "consumer", "buyer", "profile", "demographic"]),
    ("report",    ["report", "summary", "export", "generate", "compliance", "executive"]),
    ("analytics", ["ratio", "metric", "trend", "volume", "value", "growth", "rank",
                   "statistics", "analysis", "bhim", "bank", "chargeback", "cb"]),
]


def _keyword_classify(query: str) -> str | None:
    q = query.lower()
    for agent_type, keywords in KEYWORD_RULES:
        if any(kw in q for kw in keywords):
            return agent_type
    return None


def _llm_classify(query: str) -> str:
    """Use Gemini to classify when keywords don't match."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "analytics"  # safe default if no API key

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""\
Classify the following query into EXACTLY ONE of these categories:

analytics  — metrics, trends, ratios, BHIM, bank data
merchant   — merchant / vendor / business insights
customer   — customer / user / consumer behaviour
fraud      — fraud detection, suspicious activity, risk
report     — executive reports, summaries, exports

Query: {query}

Return ONLY the category name, nothing else.
"""
    response = model.generate_content(prompt)
    result = response.text.strip().lower()

    # Validate
    valid = {"analytics", "merchant", "customer", "fraud", "report"}
    return result if result in valid else "analytics"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def route_query(agent_type: str, query: str) -> str:
    """Run the specified agent on the query."""
    agent = AGENT_MAP.get(agent_type.lower(), _analytics)
    return agent.run(query)


def intelligent_router(query: str) -> dict:
    """
    Classify the query and route to the appropriate agent.
    Returns a dict with 'agent' and 'response' keys.
    """
    agent_type = _keyword_classify(query) or _llm_classify(query)
    print(f"  → Routing to: {agent_type}")
    response = route_query(agent_type, query)
    return {"agent": agent_type, "response": response}

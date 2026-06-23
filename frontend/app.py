"""
Streamlit frontend for UPI Intelligence Analyst.

Run with:
    streamlit run frontend/app.py
"""

import sys
from pathlib import Path

# Allow imports from the project root when running via `streamlit run`
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from agents.router import intelligent_router

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="UPI Intelligence Analyst",
    page_icon="🤖",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🔍 UPI Intelligence")

    st.markdown("""
    ### Available Agents

    - 📊 **Analytics** — Metrics, CB ratios, BHIM trends
    - 🏪 **Merchant** — Entity-level insights
    - 👤 **Customer** — User behaviour
    - 🚨 **Fraud** — Risk detection
    - 📋 **Report** — Executive summaries
    """)

    st.divider()

    st.markdown("### 💡 Try asking:")
    sample_queries = [
        "What is CB Ratio?",
        "Which bank has the highest risk score?",
        "How was BHIM growth in 2024?",
        "Find suspicious transaction patterns",
        "Generate an executive summary of bank risk",
    ]

    for sample in sample_queries:
        if st.button(sample, key=f"sample_{sample}", use_container_width=True):
            st.session_state["pending_query"] = sample

    st.divider()

    msg_count = len([m for m in st.session_state.get("messages", []) if m["role"] == "user"])
    st.metric("Queries sent", msg_count)

# ---------------------------------------------------------------------------
# Chat history initialisation
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "👋 Welcome to **UPI Intelligence Analyst**! "
                "Ask me anything about UPI transaction data, bank chargeback risk, "
                "BHIM growth, or fraud patterns."
            ),
            "agent": "System",
        }
    ]

# ---------------------------------------------------------------------------
# Render existing messages
# ---------------------------------------------------------------------------

st.title("🤖 UPI Intelligence Analyst")
st.caption("AI-powered UPI analytics — multi-agent RAG system")

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("agent") not in ("System", None):
            st.caption(f"🤖 Handled by: **{message['agent']} agent**")

# ---------------------------------------------------------------------------
# Handle input
# ---------------------------------------------------------------------------

query = st.chat_input("Ask anything about UPI transactions …")

# Allow sidebar sample buttons to inject a query
if "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")

if query:
    # Show user message
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Route and respond
    with st.chat_message("assistant"):
        with st.spinner("Routing to the right agent …"):
            try:
                result = intelligent_router(query)
                agent = result["agent"]
                response = result["response"]
            except EnvironmentError as e:
                agent = "system"
                response = f"⚠️ Configuration error: {e}\n\nPlease set your `GEMINI_API_KEY` in the `.env` file."
            except Exception as e:
                agent = "system"
                response = f"⚠️ Something went wrong: {e}"

        st.markdown(response)
        st.caption(f"🤖 Handled by: **{agent} agent**")

    st.session_state["messages"].append({
        "role": "assistant",
        "content": response,
        "agent": agent,
    })

    st.rerun()

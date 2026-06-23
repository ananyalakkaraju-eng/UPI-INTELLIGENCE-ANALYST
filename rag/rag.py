"""
RAG Pipeline — Retrieval-Augmented Generation using Gemini.
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

from rag.retriever import Retriever

load_dotenv()


def _get_model():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set. Add it to your .env file:\n"
            "  GEMINI_API_KEY=your_key_here"
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


SYSTEM_PROMPT = """\
You are an expert UPI Analytics Assistant for NPCI (National Payments Corporation of India).

Rules:
1. Answer ONLY using the provided context.
2. If the context does not contain the answer, say "I don't have that information in my knowledge base."
3. Quote specific numbers and values when available.
4. Be concise but complete.
5. Do not make up data.
"""


class RAGPipeline:
    def __init__(self):
        self._retriever = Retriever()
        self._model = _get_model()

    def ask(self, query: str, top_k: int = 5) -> str:
        """Answer a question using retrieved context + Gemini."""
        context = self._retriever.build_context(query, top_k)

        prompt = f"""{SYSTEM_PROMPT}

Context:
{context}

Question:
{query}

Answer:
"""
        response = self._model.generate_content(prompt)
        return response.text

    def ask_with_sources(self, query: str, top_k: int = 5) -> dict:
        """Answer a question and also return the source documents."""
        docs = self._retriever.retrieve(query, top_k)
        context = "\n\n".join(docs["document"].tolist())

        prompt = f"""{SYSTEM_PROMPT}

Context:
{context}

Question:
{query}

Answer:
"""
        response = self._model.generate_content(prompt)
        return {
            "answer": response.text,
            "sources": docs[["document_type", "entity", "similarity"]].to_dict(orient="records"),
        }

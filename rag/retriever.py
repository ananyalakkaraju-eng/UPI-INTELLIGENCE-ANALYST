"""
Retriever — Semantic search over the knowledge base embeddings.
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from rag.embeddings import load_embeddings

MODEL_NAME = "all-MiniLM-L6-v2"


class Retriever:
    def __init__(self):
        print("  Loading embedding model …")
        self._model = SentenceTransformer(MODEL_NAME)
        print("  Loading knowledge base embeddings …")
        self._kb = load_embeddings()
        self._doc_vectors = np.array(self._kb["embedding"].tolist())
        print(f"  ✓ Retriever ready — {len(self._kb)} documents indexed.")

    def retrieve(self, query: str, top_k: int = 5) -> pd.DataFrame:
        """Return the top_k most relevant documents for a query."""
        query_vec = self._model.encode([query])
        scores = cosine_similarity(query_vec, self._doc_vectors)[0]

        results = self._kb.copy()
        results["similarity"] = scores
        results = (
            results.sort_values("similarity", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )
        return results[["document_type", "entity", "similarity", "document"]]

    def build_context(self, query: str, top_k: int = 5) -> str:
        """Return retrieved documents joined as a single context string."""
        docs = self.retrieve(query, top_k)
        return "\n\n".join(docs["document"].tolist())

    def show(self, query: str, top_k: int = 5) -> None:
        """Pretty-print retrieval results."""
        results = self.retrieve(query, top_k)
        print(f"\nQUESTION: {query}\n")
        for _, row in results.iterrows():
            print("=" * 80)
            print(f"TYPE:       {row['document_type']}")
            print(f"ENTITY:     {row['entity']}")
            print(f"SIMILARITY: {row['similarity']:.4f}")
            print()
            print(row["document"])
            print()

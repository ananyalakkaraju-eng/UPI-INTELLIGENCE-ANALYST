"""
Embeddings — Generate and persist sentence embeddings for the knowledge base.
Run this once after the pipeline to create embeddings.pkl.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"


def generate_embeddings(force: bool = False) -> pd.DataFrame:
    """
    Load the knowledge base, generate embeddings, and save to disk.
    Skips generation if embeddings.pkl already exists (unless force=True).
    """
    if EMBEDDINGS_PATH.exists() and not force:
        print("  Embeddings already exist. Loading from disk …")
        return load_embeddings()

    kb_path = PROCESSED_DIR / "ai_knowledge_base.parquet"
    if not kb_path.exists():
        raise FileNotFoundError(
            "ai_knowledge_base.parquet not found. Run the pipeline first:\n"
            "  python scripts/run_pipeline.py"
        )

    print(f"  Loading knowledge base …")
    kb = pd.read_parquet(kb_path)
    print(f"  {len(kb)} documents found.")

    print(f"  Loading SentenceTransformer model: {MODEL_NAME} …")
    model = SentenceTransformer(MODEL_NAME)

    print("  Generating embeddings …")
    vectors = model.encode(kb["document"].tolist(), show_progress_bar=True)
    kb["embedding"] = vectors.tolist()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(kb, f)

    print(f"  ✓ Saved {len(kb)} embeddings → {EMBEDDINGS_PATH}")
    return kb


def load_embeddings() -> pd.DataFrame:
    """Load pre-computed embeddings from disk."""
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            "embeddings.pkl not found. Run:\n"
            "  python scripts/run_pipeline.py"
        )
    with open(EMBEDDINGS_PATH, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    generate_embeddings(force=True)

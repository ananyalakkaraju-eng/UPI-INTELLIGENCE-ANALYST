"""
Run the full data pipeline:
  Bronze → Silver → Gold → Embeddings

Run this once before starting the backend or frontend:
    python scripts/run_pipeline.py
"""

import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.bronze_layer import run as run_bronze
from pipeline.silver_layer import run as run_silver
from pipeline.gold_layer import run as run_gold
from rag.embeddings import generate_embeddings


def main():
    print("\n" + "=" * 60)
    print("  UPI Intelligence — Data Pipeline")
    print("=" * 60)

    run_bronze()
    run_silver()
    run_gold()

    print("\n=== Generating Embeddings ===")
    generate_embeddings(force=False)

    print("\n" + "=" * 60)
    print("  ✓ Pipeline complete! You can now run:")
    print()
    print("    Backend:   uvicorn backend.main:app --reload")
    print("    Frontend:  streamlit run frontend/app.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

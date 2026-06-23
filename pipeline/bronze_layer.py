"""
Bronze Layer — Raw data ingestion
Reads all chargeback and BHIM Excel files, cleans column names,
and saves combined Parquet files to data/processed/.
"""

import os
import glob
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names: strip spaces, replace special chars with _."""
    df.columns = [
        str(c)
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
        for c in df.columns
    ]
    return df


# ---------------------------------------------------------------------------
# Bronze Chargeback
# ---------------------------------------------------------------------------

def build_bronze_chargeback() -> pd.DataFrame:
    """Read all chargeback Excel files and combine into one DataFrame."""
    files = sorted(DATA_DIR.glob("chargeback_*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No chargeback files found in {DATA_DIR}")

    frames = []
    for f in files:
        print(f"  Reading {f.name} …")
        df = pd.read_excel(f)
        df = clean_columns(df)
        df["source_file"] = f.name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    # Bronze: preserve raw values as strings
    combined = combined.fillna("").astype(str)

    out = PROCESSED_DIR / "bronze_chargeback.parquet"
    combined.to_parquet(out, index=False)
    print(f"  ✓ Saved bronze_chargeback.parquet  ({len(combined):,} rows)")
    return combined


# ---------------------------------------------------------------------------
# Bronze BHIM
# ---------------------------------------------------------------------------

def build_bronze_bhim() -> pd.DataFrame:
    """Read all BHIM Excel files and combine into one DataFrame."""
    files = sorted(DATA_DIR.glob("bhim_*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No BHIM files found in {DATA_DIR}")

    frames = []
    for f in files:
        print(f"  Reading {f.name} …")
        df = pd.read_excel(f)
        df = clean_columns(df)
        df["source_file"] = f.name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.fillna("").astype(str)

    out = PROCESSED_DIR / "bronze_bhim.parquet"
    combined.to_parquet(out, index=False)
    print(f"  ✓ Saved bronze_bhim.parquet  ({len(combined):,} rows)")
    return combined


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def run():
    print("\n=== Bronze Layer ===")
    build_bronze_chargeback()
    build_bronze_bhim()
    print("Bronze layer complete.\n")


if __name__ == "__main__":
    run()

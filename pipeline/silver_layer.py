"""
Silver Layer — Type casting, cleaning, and feature engineering.
Reads bronze Parquet files and produces cleaned silver Parquet files.
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


# ---------------------------------------------------------------------------
# Silver Chargeback
# ---------------------------------------------------------------------------

def build_silver_chargeback() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "bronze_chargeback.parquet")

    # Unify the serial-number column (name differs across years)
    serial_candidates = [c for c in df.columns if "sr" in c.lower() or "serial" in c.lower()]
    if serial_candidates:
        df["serial_number"] = df[serial_candidates[0]].replace("", None)
        df.drop(columns=serial_candidates, inplace=True, errors="ignore")

    # Remove commas from numeric strings, then cast
    numeric_cols = [
        "Total_Txns_during_the_month",
        "Chargebacks_Received_during_the_month",
        "Re_presentment_Raised_during_the_month",
        "Chargebacks_Accepted_during_the_month",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].str.replace(",", "", regex=False).replace("", None)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "CB_Ratio" in df.columns:
        df["CB_Ratio"] = df["CB_Ratio"].replace("", None)
        df["CB_Ratio"] = pd.to_numeric(df["CB_Ratio"], errors="coerce")

    # Business logic: risk level
    df["risk_level"] = pd.cut(
        df["CB_Ratio"],
        bins=[-np.inf, 0.0005, 0.001, np.inf],
        labels=["Low", "Medium", "High"],
    )

    # Data quality flag
    df["is_valid"] = df["Beneficiary_Bank"].notna().astype(int)

    out = PROCESSED_DIR / "silver_chargeback.parquet"
    df.to_parquet(out, index=False)
    print(f"  ✓ Saved silver_chargeback.parquet  ({len(df):,} rows)")
    return df


# ---------------------------------------------------------------------------
# Silver BHIM
# ---------------------------------------------------------------------------

def build_silver_bhim() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "bronze_bhim.parquet")

    # Rename columns to consistent names
    rename_map = {
        "No_of_Banks_live_on_BHIM": "Banks_Live",
        "Volume_In_Mn": "Volume_Mn",
        "Value_In_Cr": "Value_Cr",
    }
    df.rename(columns=rename_map, inplace=True)

    # Parse month string → date
    df["Month_Date"] = pd.to_datetime(df["Month"], format="%B-%Y", errors="coerce")

    # Remove commas from Value_Cr
    if "Value_Cr" in df.columns:
        df["Value_Cr"] = df["Value_Cr"].str.replace(",", "", regex=False).replace("", None)

    # Cast numeric columns
    for col, dtype in [("Banks_Live", "Int64"), ("Volume_Mn", "float64"), ("Value_Cr", "float64")]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype if dtype != "Int64" else "float64")

    # Time features
    df["Year"] = df["Month_Date"].dt.year
    df["Month_Number"] = df["Month_Date"].dt.month
    df["Quarter"] = df["Month_Date"].dt.quarter

    # Data quality flag
    df["is_valid"] = df["Banks_Live"].notna().astype(int)

    # Drop duplicates (same month can appear in multiple source files)
    df.drop_duplicates(subset=["Month_Date"], keep="last", inplace=True)
    df.sort_values("Month_Date", inplace=True, ignore_index=True)

    out = PROCESSED_DIR / "silver_bhim.parquet"
    df.to_parquet(out, index=False)
    print(f"  ✓ Saved silver_bhim.parquet  ({len(df):,} rows)")
    return df


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def run():
    print("\n=== Silver Layer ===")
    build_silver_chargeback()
    build_silver_bhim()
    print("Silver layer complete.\n")


if __name__ == "__main__":
    run()

"""
Gold Layer — Business-ready aggregated tables.

Produces:
  gold_bank_risk.parquet      — Bank chargeback risk ranking
  gold_bhim_growth.parquet    — BHIM month-over-month growth metrics
  ai_knowledge_base.parquet   — Narrative documents for RAG
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


# ---------------------------------------------------------------------------
# Gold: Bank Risk
# ---------------------------------------------------------------------------

def build_gold_bank_risk() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "silver_chargeback.parquet")

    df = df[df["is_valid"] == 1].copy()
    df["risk_score"] = (df["CB_Ratio"] * 100_000).round(2)
    df["chargeback_percentage"] = (df["CB_Ratio"] * 100).round(4)
    df.sort_values("risk_score", ascending=False, inplace=True, ignore_index=True)
    df["bank_rank"] = df.index + 1

    keep = [
        "bank_rank", "Code", "Beneficiary_Bank",
        "Total_Txns_during_the_month", "Chargebacks_Received_during_the_month",
        "CB_Ratio", "chargeback_percentage", "risk_level", "risk_score",
    ]
    df = df[[c for c in keep if c in df.columns]]

    out = PROCESSED_DIR / "gold_bank_risk.parquet"
    df.to_parquet(out, index=False)
    print(f"  ✓ Saved gold_bank_risk.parquet  ({len(df):,} rows)")
    return df


# ---------------------------------------------------------------------------
# Gold: BHIM Growth
# ---------------------------------------------------------------------------

def build_gold_bhim_growth() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "silver_bhim.parquet")
    df = df[df["is_valid"] == 1].copy()
    df.sort_values("Month_Date", inplace=True, ignore_index=True)

    df["prev_volume"] = df["Volume_Mn"].shift(1)
    df["prev_value"] = df["Value_Cr"].shift(1)
    df["prev_banks"] = df["Banks_Live"].shift(1)

    df["volume_growth_pct"] = (
        (df["Volume_Mn"] - df["prev_volume"]) / df["prev_volume"] * 100
    ).round(2)
    df["value_growth_pct"] = (
        (df["Value_Cr"] - df["prev_value"]) / df["prev_value"] * 100
    ).round(2)
    df["banks_added"] = df["Banks_Live"] - df["prev_banks"]

    df["growth_status"] = np.where(
        df["volume_growth_pct"] > 0, "Growing",
        np.where(df["volume_growth_pct"] < 0, "Declining", "Stable"),
    )

    keep = [
        "Month_Date", "Year", "Quarter", "Banks_Live",
        "Volume_Mn", "Value_Cr", "banks_added",
        "volume_growth_pct", "value_growth_pct", "growth_status",
    ]
    df = df[[c for c in keep if c in df.columns]]

    out = PROCESSED_DIR / "gold_bhim_growth.parquet"
    df.to_parquet(out, index=False)
    print(f"  ✓ Saved gold_bhim_growth.parquet  ({len(df):,} rows)")
    return df


# ---------------------------------------------------------------------------
# Gold: AI Knowledge Base (narrative text documents)
# ---------------------------------------------------------------------------

def build_ai_knowledge_base() -> pd.DataFrame:
    """Convert gold tables into plain-English narrative documents for RAG."""
    records = []

    # --- Bank risk documents ---
    bank_df = pd.read_parquet(PROCESSED_DIR / "gold_bank_risk.parquet")
    for _, row in bank_df.iterrows():
        doc = (
            f"Bank {row['Beneficiary_Bank']} processed "
            f"{row.get('Total_Txns_during_the_month', 'N/A')} transactions. "
            f"It received {row.get('Chargebacks_Received_during_the_month', 'N/A')} chargebacks. "
            f"Its chargeback ratio is {round(float(row['CB_Ratio']) * 100, 4) if pd.notna(row.get('CB_Ratio')) else 'N/A'} percent. "
            f"The bank is categorised as {row.get('risk_level', 'N/A')} risk. "
            f"Its risk score is {row.get('risk_score', 'N/A')} and its risk rank is {row.get('bank_rank', 'N/A')}."
        )
        records.append({
            "document_type": "BANK",
            "entity": str(row["Beneficiary_Bank"]),
            "document": doc,
        })

    # --- BHIM growth documents ---
    bhim_df = pd.read_parquet(PROCESSED_DIR / "gold_bhim_growth.parquet")
    for _, row in bhim_df.iterrows():
        month_str = pd.Timestamp(row["Month_Date"]).strftime("%B %Y") if pd.notna(row["Month_Date"]) else "Unknown"
        doc = (
            f"For the month of {month_str}, BHIM had {row.get('Banks_Live', 'N/A')} live banks. "
            f"Transaction volume reached {round(float(row['Volume_Mn']), 2) if pd.notna(row.get('Volume_Mn')) else 'N/A'} million. "
            f"Transaction value reached {round(float(row['Value_Cr']), 2) if pd.notna(row.get('Value_Cr')) else 'N/A'} crore rupees. "
            f"Volume growth was {row.get('volume_growth_pct', 'N/A')} percent and value growth was {row.get('value_growth_pct', 'N/A')} percent. "
            f"The overall growth status was {row.get('growth_status', 'N/A')}."
        )
        records.append({
            "document_type": "BHIM",
            "entity": month_str,
            "document": doc,
        })

    # --- Business glossary ---
    glossary = [
        ("Chargeback", "A disputed transaction raised by a customer."),
        ("CB Ratio", "Chargebacks divided by total transactions. A key risk indicator."),
        ("Risk Score", "A score representing the chargeback risk of a bank. Calculated as CB_Ratio × 100,000."),
        ("Risk Level", "Categorisation of a bank's risk into Low, Medium, or High based on CB Ratio."),
        ("Volume Mn", "Transaction volume measured in millions."),
        ("Value Cr", "Transaction value measured in crores of rupees."),
        ("BHIM", "Bharat Interface for Money — NPCI's official UPI application."),
        ("UPI", "Unified Payments Interface — India's real-time payment system."),
        ("Re-presentment", "A chargeback that the merchant contests and re-presents to the bank."),
        ("Banks Live", "Number of banks actively participating in BHIM transactions."),
    ]
    for term, definition in glossary:
        records.append({
            "document_type": "GLOSSARY",
            "entity": term,
            "document": f"Term: {term}. Definition: {definition}",
        })

    kb = pd.DataFrame(records).reset_index(drop=True)
    kb.insert(0, "document_id", kb.index + 1)

    out = PROCESSED_DIR / "ai_knowledge_base.parquet"
    kb.to_parquet(out, index=False)
    print(f"  ✓ Saved ai_knowledge_base.parquet  ({len(kb):,} documents)")
    return kb


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def run():
    print("\n=== Gold Layer ===")
    build_gold_bank_risk()
    build_gold_bhim_growth()
    build_ai_knowledge_base()
    print("Gold layer complete.\n")


if __name__ == "__main__":
    run()

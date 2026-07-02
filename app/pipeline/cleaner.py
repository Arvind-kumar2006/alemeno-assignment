import pandas as pd
import numpy as np
from datetime import datetime
import re


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1: Data Cleaning Pipeline
    - Normalize date formats to ISO 8601
    - Strip currency symbols from amounts
    - Uppercase status and currency values
    - Fill missing categories with 'Uncategorised'
    - Remove exact duplicate rows
    """

    df = df.copy()

    # ── 1. Normalize column names ──────────────────────────────────────────
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # ── 2. Fix dates → ISO 8601 (YYYY-MM-DD) ──────────────────────────────
    df["date"] = df["date"].apply(normalize_date)

    # ── 3. Clean amount — strip $, commas, spaces ──────────────────────────
    df["amount"] = df["amount"].apply(clean_amount)

    # ── 4. Uppercase status and currency ──────────────────────────────────
    df["status"] = df["status"].str.strip().str.upper().fillna("UNKNOWN")
    df["currency"] = df["currency"].str.strip().str.upper().fillna("UNKNOWN")

    # ── 5. Fill missing categories ─────────────────────────────────────────
    df["category"] = df["category"].fillna("").str.strip()
    df["category"] = df["category"].replace("", "Uncategorised")

    # ── 6. Fill missing txn_id with generated value ────────────────────────
    df["txn_id"] = df["txn_id"].fillna("").str.strip()
    mask = df["txn_id"] == ""
    df.loc[mask, "txn_id"] = [
        f"GEN-{i:05d}" for i in range(mask.sum())
    ]

    # ── 7. Fill other nulls ────────────────────────────────────────────────
    df["merchant"] = df["merchant"].fillna("Unknown Merchant").str.strip()
    df["account_id"] = df["account_id"].fillna("").str.strip()
    df["notes"] = df["notes"].fillna("").str.strip()

    # ── 8. Remove exact duplicate rows ────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"[Cleaner] Removed {before - after} duplicate rows")

    # ── 9. Drop rows where amount is completely unparseable ────────────────
    df = df.dropna(subset=["amount"])

    df = df.reset_index(drop=True)
    print(f"[Cleaner] Clean rows: {len(df)}")
    return df


def normalize_date(date_val) -> str:
    """
    Handle mixed formats:
    - DD-MM-YYYY  → 2024-01-15
    - YYYY/MM/DD  → 2024-01-15
    - YYYY-MM-DD  → 2024-01-15 (already correct)
    - Any other → return as-is string
    """
    if pd.isna(date_val) or str(date_val).strip() == "":
        return None

    date_str = str(date_val).strip()

    formats = [
        "%d-%m-%Y",   # DD-MM-YYYY
        "%Y/%m/%d",   # YYYY/MM/DD
        "%Y-%m-%d",   # YYYY-MM-DD
        "%d/%m/%Y",   # DD/MM/YYYY
        "%m/%d/%Y",   # MM/DD/YYYY
        "%Y-%m-%dT%H:%M:%S",  # ISO with time
    ]

    for fmt in formats:
        try:
            parsed = datetime.strptime(date_str, fmt)
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue

    # Return as-is if nothing matched
    return date_str


def clean_amount(amount_val) -> float:
    """
    Strip $, commas, spaces from amount and convert to float
    """
    if pd.isna(amount_val):
        return None

    amount_str = str(amount_val).strip()

    # Remove $, commas, spaces, INR symbols
    amount_str = re.sub(r"[₹$,\s]", "", amount_str)

    try:
        return float(amount_str)
    except ValueError:
        return None

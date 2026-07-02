import pandas as pd
import numpy as np

# Domestic-only brands that should never have USD transactions
DOMESTIC_ONLY_MERCHANTS = [
    "swiggy", "zomato", "ola", "irctc", "bigbasket", "blinkit",
    "dunzo", "zepto", "jiomart", "flipkart", "nykaa", "meesho",
    "phonepe", "paytm", "gpay", "google pay", "amazon india"
]


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2: Anomaly Detection
    - Flag transactions where amount > 3x account median
    - Flag USD transactions on domestic-only merchants
    """
    df = df.copy()

    # Initialize anomaly columns
    df["is_anomaly"] = False
    df["anomaly_reason"] = ""

    # ── Rule 1: Amount > 3x account median 
    df = flag_statistical_outliers(df)

    # ── Rule 2: USD on domestic-only merchants 
    df = flag_currency_mismatch(df)

    anomaly_count = df["is_anomaly"].sum()
    print(f"[Anomaly] Flagged {anomaly_count} anomalies")

    return df


def flag_statistical_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each account_id, compute median amount.
    Flag any transaction where amount > 3x that median.
    """
    # Only consider successful transactions for median
    valid = df[df["amount"].notna() & (df["amount"] > 0)]

    if valid.empty:
        return df

    # Compute per-account median
    account_medians = (
        valid.groupby("account_id")["amount"]
        .median()
        .reset_index()
        .rename(columns={"amount": "median_amount"})
    )

    df = df.merge(account_medians, on="account_id", how="left")

    # Flag where amount > 3x median
    mask = (
        df["amount"].notna() &
        df["median_amount"].notna() &
        (df["amount"] > 3 * df["median_amount"])
    )

    df.loc[mask, "is_anomaly"] = True
    df.loc[mask, "anomaly_reason"] = df.loc[mask].apply(
        lambda row: (
            f"Amount {row['amount']:.2f} exceeds 3x account median "
            f"({row['median_amount']:.2f})"
        ),
        axis=1
    )

    df = df.drop(columns=["median_amount"])
    return df


def flag_currency_mismatch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag transactions where currency is USD but merchant is
    a domestic-only Indian brand.
    """
    def is_domestic_usd(row):
        if row.get("currency", "").upper() != "USD":
            return False
        merchant = str(row.get("merchant", "")).lower().strip()
        return any(d in merchant for d in DOMESTIC_ONLY_MERCHANTS)

    mismatch_mask = df.apply(is_domestic_usd, axis=1)

    df.loc[mismatch_mask, "is_anomaly"] = True
    df.loc[mismatch_mask, "anomaly_reason"] = df.loc[mismatch_mask].apply(
        lambda row: (
            f"Currency mismatch: USD used at domestic-only merchant "
            f"'{row['merchant']}'"
        ),
        axis=1
    )

    return df

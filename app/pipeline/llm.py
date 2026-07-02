import os
import json
import time
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import pandas as pd

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

VALID_CATEGORIES = [
    "Food", "Shopping", "Travel", "Transport",
    "Utilities", "Cash Withdrawal", "Entertainment", "Other"
]

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


# ── Retry decorator: 3 attempts, exponential backoff 1s→2s→4s ─────────────
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def call_groq(messages: list, response_format: str = "text") -> str:
    """
    Single Groq API call with retry logic built in via @retry decorator.
    """
    if not client:
        raise ValueError("GROQ_API_KEY not set")

    kwargs = {
        "model": "llama3-8b-8192",
        "messages": messages,
        "max_tokens": 2048,
        "temperature": 0.1,
    }

    if response_format == "json":
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def classify_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 3: LLM Classification
    For rows without a category (or Uncategorised), call LLM in batches.
    Never call LLM one row at a time.
    """
    df = df.copy()

    # Find uncategorised rows
    needs_category = df[
        df["category"].isin(["Uncategorised", "", None]) |
        df["category"].isna()
    ].copy()

    if needs_category.empty:
        print("[LLM] No transactions need categorisation")
        return df

    print(f"[LLM] Categorising {len(needs_category)} transactions in batches")

    # Batch into groups of 20 to avoid token limits
    batch_size = 20
    indices = needs_category.index.tolist()

    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]
        batch = df.loc[batch_indices]

        batch_data = batch[["merchant", "amount", "currency", "notes"]].to_dict("records")

        try:
            result = classify_batch(batch_data)

            # Map results back
            for j, idx in enumerate(batch_indices):
                if j < len(result):
                    category = result[j].get("category", "Other")
                    if category not in VALID_CATEGORIES:
                        category = "Other"
                    df.at[idx, "llm_category"] = category
                    df.at[idx, "category"] = category
                    df.at[idx, "llm_raw_response"] = json.dumps(result[j])

        except Exception as e:
            print(f"[LLM] Batch {i//batch_size + 1} failed after retries: {e}")
            for idx in batch_indices:
                df.at[idx, "llm_failed"] = True
                df.at[idx, "category"] = "Other"

    return df


def classify_batch(transactions: list) -> list:
    """
    Send one batch of transactions to LLM for categorisation.
    Returns list of {index, category} dicts.
    """
    transactions_str = json.dumps(transactions, indent=2)

    prompt = f"""You are a financial transaction classifier.

Classify each transaction into exactly one of these categories:
Food, Shopping, Travel, Transport, Utilities, Cash Withdrawal, Entertainment, Other

Transactions to classify:
{transactions_str}

Return a JSON array where each element has:
- "index": the position (0-based) of the transaction
- "category": one of the valid categories above

Return ONLY the JSON array, no explanation.
Example: [{{"index": 0, "category": "Food"}}, {{"index": 1, "category": "Travel"}}]"""

    messages = [{"role": "user", "content": prompt}]
    response = call_groq(messages, response_format="text")

    # Parse response
    response = response.strip()
    if response.startswith("```"):
        response = response.split("```")[1]
        if response.startswith("json"):
            response = response[4:]
    response = response.strip()

    result = json.loads(response)
    return result


def generate_narrative_summary(df: pd.DataFrame) -> dict:
    """
    Step 4: LLM Narrative Summary
    Single LLM call to produce structured JSON summary.
    """
    # Pre-compute stats to feed into LLM
    total_inr = df[df["currency"] == "INR"]["amount"].sum()
    total_usd = df[df["currency"] == "USD"]["amount"].sum()

    top_merchants = (
        df.groupby("merchant")["amount"]
        .sum()
        .sort_values(ascending=False)
        .head(3)
        .reset_index()
        .to_dict("records")
    )

    anomaly_count = int(df["is_anomaly"].sum())
    category_breakdown = df.groupby("category")["amount"].sum().to_dict()
    failed_count = int(df.get("llm_failed", pd.Series([False])).sum())

    context = {
        "total_transactions": len(df),
        "total_spend_inr": round(float(total_inr), 2),
        "total_spend_usd": round(float(total_usd), 2),
        "top_merchants": top_merchants,
        "anomaly_count": anomaly_count,
        "category_breakdown": {k: round(float(v), 2) for k, v in category_breakdown.items()},
        "failed_classifications": failed_count
    }

    prompt = f"""You are a financial analyst. Analyze this transaction data and return a JSON summary.

Transaction Statistics:
{json.dumps(context, indent=2)}

Return a JSON object with exactly these fields:
{{
  "total_spend_inr": <number>,
  "total_spend_usd": <number>,
  "top_merchants": [<list of top 3 merchant names>],
  "anomaly_count": <number>,
  "narrative": "<2-3 sentence summary of spending patterns and any concerns>",
  "risk_level": "<low|medium|high based on anomaly count and suspicious patterns>"
}}

Risk level guide:
- low: 0-1 anomalies, normal spending
- medium: 2-5 anomalies or unusual patterns
- high: 5+ anomalies or very suspicious activity

Return ONLY the JSON object."""

    try:
        messages = [{"role": "user", "content": prompt}]
        response = call_groq(messages, response_format="text")

        # Clean response
        response = response.strip()
        if response.startswith("```"):
            parts = response.split("```")
            response = parts[1]
            if response.startswith("json"):
                response = response[4:]
        response = response.strip()

        summary = json.loads(response)

        # Ensure all required fields exist
        summary.setdefault("total_spend_inr", context["total_spend_inr"])
        summary.setdefault("total_spend_usd", context["total_spend_usd"])
        summary.setdefault("top_merchants", [m["merchant"] for m in top_merchants])
        summary.setdefault("anomaly_count", anomaly_count)
        summary.setdefault("narrative", "Transaction analysis completed.")
        summary.setdefault("risk_level", "low")

        return summary

    except Exception as e:
        print(f"[LLM] Narrative summary failed: {e}")
        # Return computed fallback without LLM
        return {
            "total_spend_inr": context["total_spend_inr"],
            "total_spend_usd": context["total_spend_usd"],
            "top_merchants": [m["merchant"] for m in top_merchants],
            "anomaly_count": anomaly_count,
            "narrative": (
                f"Processed {len(df)} transactions totalling "
                f"₹{total_inr:.2f} INR and ${total_usd:.2f} USD. "
                f"Found {anomaly_count} anomalous transactions requiring review."
            ),
            "risk_level": "high" if anomaly_count > 5 else "medium" if anomaly_count > 1 else "low"
        }

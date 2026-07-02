import os
import io
import pandas as pd
from celery import Celery
from datetime import datetime
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import Job, Transaction, JobSummary
from app.pipeline.cleaner import clean_transactions
from app.pipeline.anomaly import detect_anomalies
from app.pipeline.llm import classify_transactions, generate_narrative_summary

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "alemeno",
    broker=REDIS_URL,
    backend=REDIS_URL
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
)


@celery_app.task(bind=True, name="process_csv_job")
def process_csv_job(self, job_id: str, csv_content: str):
    """
    Main Celery task — runs the full 5-step pipeline:
    a) Data Cleaning
    b) Anomaly Detection
    c) LLM Classification
    d) LLM Narrative Summary
    e) Retry Logic (handled via tenacity in llm.py)
    """
    db: Session = SessionLocal()

    try:
        # Mark job as processing 
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            print(f"[Task] Job {job_id} not found")
            return

        job.status = "processing"
        db.commit()

        print(f"[Task] Starting pipeline for job {job_id}")

        #  Parse CSV 
        df = pd.read_csv(io.StringIO(csv_content))
        job.row_count_raw = len(df)
        db.commit()

        # Step a: Data Cleaning 
        print("[Task] Step a: Cleaning data")
        df = clean_transactions(df)
        job.row_count_clean = len(df)
        db.commit()

        #  Step b: Anomaly Detection 
        print("[Task] Step b: Detecting anomalies")
        df = detect_anomalies(df)

        #  Step c: LLM Classification 
        print("[Task] Step c: LLM classification")
        df = classify_transactions(df)

        #  Save transactions to DB 
        print("[Task] Saving transactions to database")
        save_transactions(db, job_id, df)

        #  Step d: LLM Narrative Summary 
        print("[Task] Step d: Generating narrative summary")
        summary_data = generate_narrative_summary(df)
        save_summary(db, job_id, summary_data)

        #  Mark job completed 
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        db.commit()

        print(f"[Task] Job {job_id} completed successfully")

    except Exception as e:
        print(f"[Task] Job {job_id} failed: {e}")
        db.rollback()

        try:
            job = db.query(Job).filter(Job.id == job_id).first()
            if job:
                job.status = "failed"
                job.error_message = str(e)
                job.completed_at = datetime.utcnow()
                db.commit()
        except Exception as inner_e:
            print(f"[Task] Could not update job status: {inner_e}")

    finally:
        db.close()


def save_transactions(db: Session, job_id: str, df: pd.DataFrame):
    """Save cleaned + annotated transactions to PostgreSQL."""

    # Delete existing transactions for this job (in case of retry)
    db.query(Transaction).filter(Transaction.job_id == job_id).delete()
    db.commit()

    transactions = []
    for _, row in df.iterrows():
        txn = Transaction(
            job_id=job_id,
            txn_id=str(row.get("txn_id", "")) or None,
            date=str(row.get("date", "")) or None,
            merchant=str(row.get("merchant", "")) or None,
            amount=float(row["amount"]) if pd.notna(row.get("amount")) else None,
            currency=str(row.get("currency", "")) or None,
            status=str(row.get("status", "")) or None,
            category=str(row.get("category", "")) or None,
            account_id=str(row.get("account_id", "")) or None,
            notes=str(row.get("notes", "")) or None,
            is_anomaly=bool(row.get("is_anomaly", False)),
            anomaly_reason=str(row.get("anomaly_reason", "")) or None,
            llm_category=str(row.get("llm_category", "")) or None,
            llm_raw_response=str(row.get("llm_raw_response", "")) or None,
            llm_failed=bool(row.get("llm_failed", False)),
        )
        transactions.append(txn)

    db.bulk_save_objects(transactions)
    db.commit()
    print(f"[Task] Saved {len(transactions)} transactions")


def save_summary(db: Session, job_id: str, summary_data: dict):
    """Save LLM-generated summary to PostgreSQL."""

    # Delete existing summary
    db.query(JobSummary).filter(JobSummary.job_id == job_id).delete()
    db.commit()

    summary = JobSummary(
        job_id=job_id,
        total_spend_inr=float(summary_data.get("total_spend_inr", 0)),
        total_spend_usd=float(summary_data.get("total_spend_usd", 0)),
        top_merchants=summary_data.get("top_merchants", []),
        anomaly_count=int(summary_data.get("anomaly_count", 0)),
        narrative=summary_data.get("narrative", ""),
        risk_level=summary_data.get("risk_level", "low"),
    )

    db.add(summary)
    db.commit()
    print(f"[Task] Saved job summary — risk level: {summary.risk_level}")

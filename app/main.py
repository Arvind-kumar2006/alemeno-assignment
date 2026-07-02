import os
import uuid
import io
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import Optional, List
from collections import defaultdict

from app.database import get_db, create_tables
from app.models import Job, Transaction, JobSummary
from app.schemas import (
    JobStatusResponse,
    JobResultsResponse,
    JobListItem,
    TransactionSchema,
    JobSummarySchema,
)
from app.tasks import process_csv_job

app = FastAPI(
    title="Alemeno — AI-Powered Transaction Processing Pipeline",
    description="Upload dirty CSV transactions, process via job queue, classify with LLM",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    """Create DB tables on startup."""
    create_tables()
    print("[App] Database tables created")


# POST /jobs/upload
# Accept CSV, create Job record, enqueue Celery task, return job_id immediately
@app.post("/jobs/upload", status_code=202)
async def upload_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Validate file type
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are accepted"
        )

    # Read file content
    content_bytes = await file.read()
    if len(content_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        csv_content = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        csv_content = content_bytes.decode("latin-1")

    # Quick validation — check it has rows
    lines = [l for l in csv_content.strip().split("\n") if l.strip()]
    if len(lines) < 2:
        raise HTTPException(
            status_code=400,
            detail="CSV must have at least a header row and one data row"
        )

    # Create Job record
    job_id = str(uuid.uuid4())
    job = Job(
        id=job_id,
        filename=file.filename,
        status="pending",
        row_count_raw=len(lines) - 1,  # exclude header
    )
    db.add(job)
    db.commit()

    # Enqueue Celery task — returns immediately
    process_csv_job.delay(job_id, csv_content)

    return {
        "job_id": job_id,
        "status": "pending",
        "message": f"Job created. Poll /jobs/{job_id}/status for updates.",
        "filename": file.filename,
        "rows_detected": len(lines) - 1,
    }



# GET /jobs/{job_id}/status
@app.get("/jobs/{job_id}/status")
def get_job_status(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    response = {
        "job_id": job.id,
        "status": job.status,
        "filename": job.filename,
        "row_count_raw": job.row_count_raw,
        "row_count_clean": job.row_count_clean,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
        "error_message": job.error_message,
    }

    # Include summary if completed
    if job.status == "completed" and job.summary:
        s = job.summary
        response["summary"] = {
            "total_spend_inr": s.total_spend_inr,
            "total_spend_usd": s.total_spend_usd,
            "anomaly_count": s.anomaly_count,
            "risk_level": s.risk_level,
            "top_merchants": s.top_merchants,
        }

    return response


# GET /jobs/{job_id}/results
# Full structured output — cleaned transactions, anomalies, breakdown, summary
@app.get("/jobs/{job_id}/results")
def get_job_results(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status not in ("completed", "failed"):
        raise HTTPException(
            status_code=202,
            detail=f"Job is still {job.status}. Try again later."
        )

    transactions = db.query(Transaction).filter(
        Transaction.job_id == job_id
    ).all()

    anomalies = [t for t in transactions if t.is_anomaly]

    # Per-category spend breakdown
    category_breakdown = defaultdict(float)
    for t in transactions:
        if t.category and t.amount:
            category_breakdown[t.category] += t.amount
    category_breakdown = {k: round(v, 2) for k, v in category_breakdown.items()}

    summary = None
    if job.summary:
        s = job.summary
        summary = {
            "total_spend_inr": s.total_spend_inr,
            "total_spend_usd": s.total_spend_usd,
            "top_merchants": s.top_merchants,
            "anomaly_count": s.anomaly_count,
            "narrative": s.narrative,
            "risk_level": s.risk_level,
        }

    def serialize_txn(t):
        return {
            "id": t.id,
            "txn_id": t.txn_id,
            "date": t.date,
            "merchant": t.merchant,
            "amount": t.amount,
            "currency": t.currency,
            "status": t.status,
            "category": t.category,
            "account_id": t.account_id,
            "notes": t.notes,
            "is_anomaly": t.is_anomaly,
            "anomaly_reason": t.anomaly_reason,
            "llm_category": t.llm_category,
            "llm_failed": t.llm_failed,
        }

    return {
        "job_id": job_id,
        "status": job.status,
        "total_transactions": len(transactions),
        "transactions": [serialize_txn(t) for t in transactions],
        "anomalies": [serialize_txn(t) for t in anomalies],
        "category_breakdown": category_breakdown,
        "summary": summary,
    }



# GET /jobs
# List all jobs with optional ?status= filter
@app.get("/jobs")
def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    db: Session = Depends(get_db)
):
    query = db.query(Job)

    if status:
        valid_statuses = ["pending", "processing", "completed", "failed"]
        if status not in valid_statuses:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Must be one of: {valid_statuses}"
            )
        query = query.filter(Job.status == status)

    jobs = query.order_by(Job.created_at.desc()).all()

    return {
        "total": len(jobs),
        "jobs": [
            {
                "job_id": j.id,
                "filename": j.filename,
                "status": j.status,
                "row_count_raw": j.row_count_raw,
                "row_count_clean": j.row_count_clean,
                "created_at": j.created_at,
                "completed_at": j.completed_at,
            }
            for j in jobs
        ]
    }


# GET /health
# Health check endpoint
@app.get("/health")
def health():
    return {"status": "ok", "service": "alemeno-transaction-pipeline"}

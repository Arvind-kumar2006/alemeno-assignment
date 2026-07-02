from pydantic import BaseModel
from typing import Optional, List, Any
from datetime import datetime


class JobCreate(BaseModel):
    filename: str


class JobSummarySchema(BaseModel):
    total_spend_inr: float
    total_spend_usd: float
    top_merchants: Optional[Any]
    anomaly_count: int
    narrative: Optional[str]
    risk_level: Optional[str]

    class Config:
        from_attributes = True


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    filename: str
    row_count_raw: int
    row_count_clean: int
    created_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]
    summary: Optional[JobSummarySchema]

    class Config:
        from_attributes = True


class TransactionSchema(BaseModel):
    id: str
    txn_id: Optional[str]
    date: Optional[str]
    merchant: Optional[str]
    amount: Optional[float]
    currency: Optional[str]
    status: Optional[str]
    category: Optional[str]
    account_id: Optional[str]
    notes: Optional[str]
    is_anomaly: bool
    anomaly_reason: Optional[str]
    llm_category: Optional[str]
    llm_failed: bool

    class Config:
        from_attributes = True


class JobResultsResponse(BaseModel):
    job_id: str
    status: str
    transactions: List[TransactionSchema]
    anomalies: List[TransactionSchema]
    summary: Optional[JobSummarySchema]
    category_breakdown: Optional[dict]

    class Config:
        from_attributes = True


class JobListItem(BaseModel):
    job_id: str
    filename: str
    status: str
    row_count_raw: int
    created_at: Optional[datetime]

    class Config:
        from_attributes = True

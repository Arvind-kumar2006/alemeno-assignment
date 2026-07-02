# Alemeno — AI-Powered Transaction Processing Pipeline

A backend API that accepts dirty CSV transactions, processes them asynchronously via a job queue, classifies them using an LLM, detects anomalies, and generates structured reports.

## Architecture

```
Client
  │
  ▼
FastAPI (port 8000)
  │── POST /jobs/upload     → Creates Job, enqueues task, returns job_id
  │── GET  /jobs/{id}/status → Poll for completion
  │── GET  /jobs/{id}/results → Full results when done
  └── GET  /jobs             → List all jobs
  │
  ▼
Redis (message broker)
  │
  ▼
Celery Worker
  │── a) Data Cleaning      → Normalize dates, amounts, fix casing, dedup
  │── b) Anomaly Detection  → 3x median rule + domestic merchant USD check
  │── c) LLM Classification → Batch categorise uncategorised transactions
  │── d) LLM Summary        → Generate narrative + risk level
  └── e) Retry Logic        → Exponential backoff, 3 attempts per LLM call
  │
  ▼
PostgreSQL
  └── Jobs, Transactions, JobSummaries
```

## Tech Stack

- **API**: FastAPI + Uvicorn
- **Queue**: Celery + Redis
- **Database**: PostgreSQL + SQLAlchemy
- **LLM**: Groq API (llama3-8b-8192) — free tier
- **Containers**: Docker + Docker Compose

## Setup

### 1. Clone the repo
```bash
git clone <your-repo-url>
cd alemeno-assignment
```

### 2. Add your Groq API key
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
# Get free key at: https://console.groq.com
```

### 3. Start all services
```bash
docker compose up --build
```

That's it. All 4 services (API, Worker, Redis, PostgreSQL) start with one command.

### 4. Verify it's running
```bash
curl http://localhost:8000/health
# {"status": "ok", "service": "alemeno-transaction-pipeline"}
```

API docs available at: http://localhost:8000/docs

---

## API Usage

### Upload a CSV file
```bash
curl -X POST http://localhost:8000/jobs/upload \
  -F "file=@transactions.csv"

# Response:
# {
#   "job_id": "abc-123-...",
#   "status": "pending",
#   "message": "Job created. Poll /jobs/abc-123-.../status for updates."
# }
```

### Poll job status
```bash
curl http://localhost:8000/jobs/abc-123-.../status

# Response when processing:
# {"job_id": "abc-123", "status": "processing", ...}

# Response when completed:
# {"job_id": "abc-123", "status": "completed", "summary": {...}}
```

### Get full results
```bash
curl http://localhost:8000/jobs/abc-123-.../results

# Response includes:
# - transactions: all cleaned transactions
# - anomalies: flagged transactions
# - category_breakdown: spend per category
# - summary: LLM narrative + risk level
```

### List all jobs
```bash
# All jobs
curl http://localhost:8000/jobs

# Filter by status
curl "http://localhost:8000/jobs?status=completed"
curl "http://localhost:8000/jobs?status=failed"
```

---

## Processing Pipeline (5 Steps)

### a) Data Cleaning
- Normalizes mixed date formats (DD-MM-YYYY, YYYY/MM/DD → ISO 8601)
- Strips currency symbols ($, ₹) from amounts
- Uppercases status and currency values
- Fills missing categories with 'Uncategorised'
- Removes exact duplicate rows
- Generates txn_id for rows where it's missing

### b) Anomaly Detection
- Flags transactions where amount > 3× the account's median amount
- Flags USD transactions at domestic-only merchants (Swiggy, Ola, IRCTC, Zomato, etc.)

### c) LLM Classification
- Batches uncategorised transactions (20 per call) — never one call per row
- Categories: Food, Shopping, Travel, Transport, Utilities, Cash Withdrawal, Entertainment, Other
- Failed batches marked as `llm_failed=true`, assigned 'Other' — job continues

### d) LLM Narrative Summary
- Single LLM call generates: total spend by currency, top 3 merchants, anomaly count, 2-3 sentence narrative, risk level (low/medium/high)

### e) Retry Logic
- Every LLM call retries up to 3 times with exponential backoff (1s → 2s → 4s)
- If all retries fail, marks batch as `llm_failed` and continues — does not fail the entire job

---

## Data Model

```
Job
├── id, filename, status
├── row_count_raw, row_count_clean
├── created_at, completed_at, error_message
└── → Transactions, → JobSummary

Transaction
├── id, job_id (FK)
├── txn_id, date, merchant, amount, currency
├── status, category, account_id, notes
├── is_anomaly, anomaly_reason
└── llm_category, llm_raw_response, llm_failed

JobSummary
├── id, job_id (FK)
├── total_spend_inr, total_spend_usd
├── top_merchants (JSON), anomaly_count
├── narrative, risk_level
```

---

## Scaling Considerations

**Current bottlenecks at 100x traffic:**
1. Single Celery worker → add more workers: `--concurrency=8`
2. PostgreSQL connection pool → add PgBouncer
3. LLM API rate limits → implement request throttling + larger batch sizes
4. Single Redis instance → Redis Cluster for high availability

**Next iteration for enterprise scale:**
- Horizontal Celery workers behind a load balancer
- Read replicas for PostgreSQL reporting queries
- S3/GCS for CSV storage instead of in-memory passing
- Kafka instead of Redis for durability guarantees
- Separate LLM rate-limit service with token bucket algorithm

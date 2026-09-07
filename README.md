# Your Assistant — Autonomous AI Job Search Platform

**Your Assistant** is an enterprise-grade, autonomous backend service designed to streamline the modern job hunt. Built with FastAPI, SQLite/SQLAlchemy, ChromaDB vector RAG, and an intelligent multi-tier LLM engine, it continuously scrapes opportunities, ranks them with cosine similarity against your real resume, auto-generates 100% ATS-compliant PDFs without hallucinating skills, and drafts personalized cold outreach to recruiters.

---

## Key Pillars & Architecture

```mermaid
flowchart TD
    subgraph Daily Background Worker [Daily 24-Hour Autonomous Scheduler]
        A[SerpApi Google Jobs Scraper] -->|Filtered by 7-Day Deduplication| B[Candidate Job Descriptions]
        B --> C[ChromaDB + FastEmbed Vector RAG Engine]
        R[Static Base Resume data/resume.txt] --> C
        C -->|Match Score > 75% Cutoff| D[(SQLAlchemy SQLite JobHistory)]
        D -->|Auto-Retention Worker| E[Delete Records Older than 7 Days]
    end

    subgraph User & API Ingestion [FastAPI Endpoints with Bot Defense]
        Client[Web Client / API Consumer] -->|SlowAPI IP Throttling| F[FastAPI Gateway]
        F --> G[GET /jobs?job_keyword=...]
        F --> H[POST /generate-resume/{id}]
        F --> I[POST /generate-email/{id}]
    end

    subgraph Resilient LLM Engine [Multi-Tier Provider & Key Fallback]
        H --> L[LLM Manager]
        I --> L
        L --> M{Try Claude 3 Key 1}
        M -->|429 Rate Limit| N{Try Claude 3 Key 2}
        N -->|Claude Exhausted| O{Try Gemini 1.5 Key 1}
        O -->|429 Rate Limit| P{Try Gemini 1.5 Key 2}
        P -->|All Failed| Q[Raise AllProvidersExhaustedError]
    end

    H --> S[ReportLab ATS PDF Generator]
    S --> T[Publication-Ready PDF Download]
```

### 1. Robust Bot Protection & Validation
- **Strict Rate Limiting**: Powered by `slowapi` using remote IP addresses to mitigate brute force probing, scraping, and denial-of-service.
- **Strict Input Sanitization**: Pydantic v2 schemas enforce `extra='forbid'`, rejecting unexpected payload injections, and validate URLs and email structures.
- **Malicious Payload Defense**: Automated sanitizers strip HTML/XSS scripts from incoming inputs.

### 2. Multi-Tier LLM Key Rotation & Fallback Engine
- **Claude 3 First**: Attempts high-fidelity reasoning using Anthropic Claude 3 (`claude-3-5-sonnet` / `claude-3-haiku`) with Primary Key 1.
- **Intelligent Rotation**: If Key 1 triggers HTTP `429` (Rate Limited) or `529` (Overloaded), automatically rotates to Key 2, Key 3, etc.
- **Gemini 1.5 Fallback**: If all Claude keys are exhausted, cascades to Google Gemini 1.5 (`gemini-1.5-flash` / `gemini-1.5-pro`) with sequential Gemini key rotation.
- **Zero Downtime**: Guarantees generation reliability even during high-traffic provider outages.

### 3. "No-Fake-Skills" 100% ATS Resume Tailoring
- **Anti-Hallucination Constraint**: Enforces a strict system prompt:
  > *"Rewrite the resume for 100% ATS compatibility. You MUST ONLY use skills and experiences present in the original resume. DO NOT hallucinate or add any fake skills."*
- **Targeted Keyword Emphasis**: Re-aligns genuine past achievements and technical keywords to match target job descriptions.
- **Pure Python PDF Generation**: Utilizes `reportlab` to render clean, publication-ready PDFs with ATS-friendly typography (standard margins, structured headings, bullet points). No external binary dependencies (e.g. wkhtmltopdf) required.

### 4. Local Vector RAG Engine (ChromaDB + FastEmbed)
- **High-Performance Vector Store**: Local persistent ChromaDB instance with cosine distance indexing (`hnsw:space="cosine"`).
- **Lightweight Open-Source Model**: Generates 384-dimensional dense semantic vectors using `BAAI/bge-small-en-v1.5` via FastEmbed and ONNX Runtime locally on CPU.
- **Cost-Optimized (> 75% Score Filter)**: Evaluates semantic similarity between the job description and the user's base resume. **Prunes candidates with match score $\le 75\%$**, eliminating unnecessary downstream LLM API token costs.

### 5. Recruiter Discovery & Cold Email Generator
- **Contact Extraction**: Automatically parses recruiter contact emails from job descriptions via regex, safely falling back to `[Recruiter Email]` when missing.
- **Tailored Value Proposition**: Drafts concise (< 175 words) cold outreach emphasizing genuine skill overlap and a strong call-to-action.
- **Guaranteed JSON Structure**: Returns clean schema:
  ```json
  {
    "email": "recruiter@example.com",
    "subject": "Senior AI Systems Engineer - Alex Rivera | Python & RAG Specialist",
    "body": "Dear Recruiting Team,\n\nI noticed your opening for a Senior AI Systems Engineer at Anthropic..."
  }
  ```

### 6. Automated Background Scheduling & Retention
- **APScheduler Background Worker**: Runs daily every 24 hours to automatically scrape jobs, calculate vector similarity, and persist new high-matching opportunities.
- **7-Day Deduplication**: Checks incoming links against `JobHistory` to omit duplicate jobs recorded in the last 7 days.
- **Self-Cleaning Retention**: Automatically purges job history records older than 7 days to maintain a lean database.

---

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── database.py       # Engine, Base, session maker, 7-day retention cleanup & deduplication
│   ├── models.py         # SQLAlchemy JobHistory table + strict Pydantic schemas
│   ├── schemas.py        # Re-exported Pydantic schemas
│   ├── scraper.py        # SerpApi scraper, career portal predictor, 7-day duplicate filter
│   ├── rag_engine.py     # Local ChromaDB vector engine, FastEmbed model, >75% cosine filter
│   ├── llm_manager.py    # Multi-key rotation, Claude 3 -> Gemini 1.5 fallback engine
│   ├── resume_builder.py # ATS resume tailoring (anti-hallucination prompt) & ReportLab PDF
│   ├── email_generator.py# Recruiter email extraction (regex/fallback) & cold email generation
│   └── main.py           # Connected endpoints, SlowAPI limiter, APScheduler daily cron
├── data/
│   └── resume.txt        # User base resume profile for vector matching
├── tests/
│   ├── test_backend.py   # Test suite for validation, rate limiting, and cleanup
│   ├── test_scraper.py   # Test suite for queries, career prediction, and deduplication
│   ├── test_rag_engine.py# Test suite for ChromaDB, embeddings, and >75% match threshold
│   ├── test_llm_manager.py # Test suite for multi-key rotation and multi-tier LLM fallback
│   ├── test_resume_builder.py # Test suite for ATS prompt constraint and PDF generation
│   ├── test_email_generator.py# Test suite for recruiter email extraction and JSON schema
│   └── test_main_integration.py # Integration test suite for connected /jobs and generation routes
├── .env.example          # Environment variables template (Claude & Gemini multi-keys)
├── requirements.txt      # Project dependencies
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file based on `.env.example`:
```bash
cp .env.example .env
```

```ini
# Comma-separated API keys for rotation & fallback
CLAUDE_KEYS=sk-ant-api03-xxx1,sk-ant-api03-xxx2
GEMINI_KEYS=AIzaSyxxx1,AIzaSyxxx2

# SerpApi Key for automated job scraping
SERPAPI_API_KEY=your_serpapi_api_key_here

# Database URL (defaults to sqlite:///./jobs.db)
DATABASE_URL=sqlite:///./jobs.db
```

### 3. Run the Backend Server

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Access the interactive API documentation:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## API Reference

### Core Application Endpoints

| Method | Endpoint | Description | Rate Limit |
|---|---|---|---|
| `GET` | `/` | Health check, status, and easter egg | 30/min |
| `GET` | `/jobs` | Retrieve stored jobs with optional `job_keyword` search across title/company | 60/min |
| `POST` | `/generate-resume/{id}` | Generate tailored ATS resume Markdown & ReportLab PDF for a stored job | 10/min |
| `POST` | `/generate-email/{id}` | Extract recruiter contact and generate cold email JSON for a stored job | 15/min |

### Programmatic & Modular Sub-Routes

| Method | Endpoint | Description | Rate Limit |
|---|---|---|---|
| `POST` | `/api/jobs` | Record a discovered job opportunity | 20/min |
| `GET` | `/api/jobs` | Query & paginate stored job matches | 60/min |
| `POST` | `/api/jobs/scrape` | On-demand SerpApi scrape with career portal prediction & 7-day deduplication | 10/min |
| `POST` | `/api/jobs/match` | Vector RAG evaluation against resume, returning only jobs > 75% score | 10/min |
| `POST` | `/api/ai/generate` | Direct AI text generation via Claude 3 (Key 1 -> 2) -> Gemini 1.5 (Key 1 -> 2) | 15/min |
| `POST` | `/api/resume/tailor` | Tailor custom resume markdown with zero-hallucination constraint | 10/min |
| `POST` | `/api/resume/generate-pdf` | Convert custom markdown into an ATS-friendly PDF download | 10/min |
| `POST` | `/api/email/generate` | Generate cold email JSON for arbitrary job descriptions | 15/min |
| `POST` | `/api/jobs/cleanup` | Manually run retention pruning (default: 7 days) | 5/min |

---

## Running the Automated Test Suite

Execute the complete test suite across all platform modules:
```bash
python -m pytest -v
```

All **35 automated unit and integration tests** validate:
- Easter egg & security rate limiting
- Database 7-day automated retention cleanup
- Scraper query fallback, career portal prediction & duplicate check
- Vector RAG engine with ChromaDB, FastEmbed & >75% cutoff
- LLM multi-key rotation and Claude 3 -> Gemini 1.5 fallback
- ATS strict prompt constraint enforcement & ReportLab PDF generation
- Recruiter email regex extraction & cold email JSON generation
- Connected `/jobs`, `/generate-resume/{id}`, `/generate-email/{id}` routes & daily scheduler pipeline

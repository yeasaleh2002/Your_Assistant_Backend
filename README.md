# Your Assistant — Autonomous AI Job Search Platform

**Your Assistant** is an enterprise-grade, autonomous backend service designed to streamline the modern job search lifecycle. Built with FastAPI, SQLite/SQLAlchemy, ChromaDB vector RAG, ReportLab, and an intelligent multi-tier LLM engine, it continuously scrapes opportunities, ranks them with cosine similarity against your real resume, auto-generates 100% ATS-compliant PDFs without hallucinating fake skills, and drafts personalized cold outreach to recruiters.

---

## Table of Contents

1. [Architecture & Workflow](#architecture--workflow)
2. [Technology Stack](#technology-stack)
3. [Folder Structure](#folder-structure)
4. [Environment Setup & Installation](#environment-setup--installation)
5. [Complete API Reference](#complete-api-reference)
   - [Primary Application Endpoints](#primary-application-endpoints)
     - [`GET /`](#1-get-)
     - [`GET /jobs`](#2-get-jobs)
     - [`POST /generate-resume/{id}`](#3-post-generate-resumeid)
     - [`POST /generate-email/{id}`](#4-post-generate-emailid)
   - [Modular & Programmatic Endpoints](#modular--programmatic-endpoints)
     - [`POST /api/jobs`](#5-post-apijobs)
     - [`GET /api/jobs`](#6-get-apijobs)
     - [`POST /api/jobs/scrape`](#7-post-apijobsscrape)
     - [`POST /api/jobs/match`](#8-post-apijobsmatch)
     - [`POST /api/ai/generate`](#9-post-apiaigenerate)
     - [`POST /api/resume/tailor`](#10-post-apiresumetailor)
     - [`POST /api/resume/generate-pdf`](#11-post-apiresumegenerate-pdf)
     - [`POST /api/email/generate`](#12-post-apiemailgenerate)
     - [`POST /api/jobs/cleanup`](#13-post-apijobscleanup)
6. [Testing with Postman](#testing-with-postman)
7. [Automated Test Suite](#automated-test-suite)

---

## Architecture & Workflow

```mermaid
flowchart TD
    subgraph Daily Background Worker [Daily 24-Hour Autonomous Scheduler]
        A[SerpApi Google Jobs Scraper] -->|7-Day Deduplication Check| B[Candidate Job Postings]
        B --> C[ChromaDB + FastEmbed Vector RAG Engine]
        R[Static Base Resume data/resume.txt] --> C
        C -->|Match Score > 75% Cutoff| D[(SQLAlchemy SQLite JobHistory)]
        D -->|Auto-Retention Cleanup| E[Delete Records Older than 7 Days]
    end

    subgraph FastAPI Ingestion & Bot Defense [Secure Gateway]
        Client[Client / Postman / Frontend] -->|SlowAPI IP Throttling| F[FastAPI Application]
        F --> G[GET /jobs?job_keyword=...]
        F --> H[POST /generate-resume/{id}]
        F --> I[POST /generate-email/{id}]
    end

    subgraph Resilient Multi-LLM Engine [Cascading Provider & Key Rotation]
        H --> L[LLM Manager]
        I --> L
        L --> M{Claude 3 Key 1}
        M -->|429 / 529 Rate Limit| N{Claude 3 Key 2}
        N -->|All Claude Keys Fail| O{Gemini 1.5 Key 1}
        O -->|429 Rate Limit| P{Gemini 1.5 Key 2}
        P -->|All Providers Exhausted| Q[Raise AllProvidersExhaustedError 503]
    end

    H --> S[ReportLab ATS Engine]
    S --> T[Instant ATS-Friendly PDF Download]
```

### Core Innovations & Guarantees
1. **Zero-Hallucination ATS Resume Builder**: Strictly prompts the LLM:
   > *"Rewrite the resume for 100% ATS compatibility. You MUST ONLY use skills and experiences present in the original resume. DO NOT hallucinate or add any fake skills."*
2. **Local Vector RAG Engine**: Embeds job descriptions locally using `BAAI/bge-small-en-v1.5` on CPU with ChromaDB. Filters with a hard cutoff ($\text{Score} > 75\%$) to eliminate wasted LLM API tokens.
3. **Multi-Tier Fallback System**: Auto-rotates keys on Anthropic Claude 3 (`claude-3-5-sonnet` / `claude-3-haiku`), cascading to Google Gemini 1.5 (`gemini-1.5-flash` / `gemini-1.5-pro`) during rate limits.
4. **Automated Background Scheduler & Pruning**: Built-in `AsyncIOScheduler` runs every 24 hours to scrape, match, persist, and auto-delete jobs older than 7 days.
5. **Strict Rate Limiting & Malicious Input Defense**: `slowapi` IP-based limits protect all endpoints, while Pydantic `extra="forbid"` models reject unknown/injected parameters.

---

## Technology Stack

| Layer / Concern | Technology | Version | Purpose |
|---|---|---|---|
| **Web Framework** | [FastAPI](https://fastapi.tiangolo.com/) | `>= 0.115.0` | Asynchronous REST API framework, OpenAPI/Swagger generation, dependency injection. |
| **ASGI Server** | [Uvicorn](https://www.uvicorn.org/) | `>= 0.30.0` | High-speed production ASGI server with reload support. |
| **Database ORM** | [SQLAlchemy](https://www.sqlalchemy.org/) | `>= 2.0.30` | Modern Python SQL toolkit & ORM managing SQLite persistence with connection pooling. |
| **Database Engine** | SQLite 3 | Built-in | Zero-configuration relational database engine storing `JobHistory` records. |
| **Data Validation** | [Pydantic](https://docs.pydantic.dev/) | `>= 2.8.0` | Strict data parsing, URL/Email validation, and `extra="forbid"` payload protection. |
| **Rate Limiting** | [SlowAPI](https://slowapi.readthedocs.io/) | `>= 0.1.9` | Client IP-based rate limiting to prevent bot attacks and API abuse. |
| **Cron / Scheduler** | [APScheduler](https://apscheduler.readthedocs.io/) | `>= 3.10.4` | In-process background scheduler for autonomous 24-hour scraping and 7-day retention cleanup. |
| **Vector Database** | [ChromaDB](https://www.trychroma.com/) | `>= 0.5.0` | Local persistent vector database indexing jobs with HNSW cosine distance (`hnsw:space="cosine"`). |
| **Embeddings** | [FastEmbed](https://qdrant.github.io/fastembed/) | `>= 0.3.0` | Lightweight, CPU-optimized embedding generation using `BAAI/bge-small-en-v1.5` (384-dimensional dense vectors). |
| **PDF Generation** | [ReportLab](https://www.reportlab.com/) | `>= 4.0.0` | Pure Python programmatic ATS-optimized PDF generation (no external binaries like `wkhtmltopdf`). |
| **HTTP Client** | [Requests](https://requests.readthedocs.io/) | `>= 2.31.0` | Synchronous HTTP client for SerpApi scraping and LLM API endpoints. |
| **Type Stubs** | [types-requests](https://github.com/python/typeshed) | `>= 2.31.0` | PEP 561 static type stubs for Pyrefly / Pyright type checking. |
| **Primary LLM** | Anthropic Claude 3 | API | Primary reasoning provider (`claude-3-5-sonnet`, `claude-3-haiku`) with multi-key rotation. |
| **Fallback LLM** | Google Gemini 1.5 | API | High-speed secondary fallback provider (`gemini-1.5-flash`, `gemini-1.5-pro`). |
| **Job Search Engine**| [SerpApi](https://serpapi.com/) | API | Google Jobs scraping engine with structured employer metadata extraction. |
| **Testing** | [Pytest](https://docs.pytest.org/) | `>= 8.0.0` | Automated testing framework with 35 test suites verifying end-to-end functionality. |

---

## Folder Structure

```
Your_Assistant/
├── app/
│   ├── __init__.py               # Package initializer with dynamic runtime compatibility shim
│   ├── database.py               # SQLAlchemy engine, SessionLocal, get_db, and 7-day retention cleanup
│   ├── models.py                 # JobHistory SQLAlchemy model and strict Pydantic schemas
│   ├── schemas.py                # Schema aliases and re-exports
│   ├── scraper.py                # SerpApi Google Jobs scraper, career portal predictor, 7-day duplicate check
│   ├── rag_engine.py             # ChromaDB persistent vector engine, FastEmbed model, >75% cosine filter
│   ├── llm_manager.py            # Multi-key rotation, Claude 3 -> Gemini 1.5 cascading fallback engine
│   ├── resume_builder.py         # ATS resume tailoring (anti-hallucination prompt) & ReportLab PDF generator
│   ├── email_generator.py        # Recruiter contact email extraction (regex/fallback) & cold outreach JSON generator
│   └── main.py                   # FastAPI application, SlowAPI rate limiter, connected endpoints & APScheduler cron
├── data/
│   └── resume.txt                # Static user base resume profile text used for vector matching
├── output/                       # Output directory where generated tailored resume PDFs are stored
├── typings/
│   └── requests/                 # Local type stub definitions ensuring strict static analyzer compliance
├── tests/
│   ├── test_backend.py           # Rate limiting, strict validation, and automated 7-day retention cleanup tests
│   ├── test_scraper.py           # Query generation, SerpApi scraping, career URL predictor, and duplicate tests
│   ├── test_rag_engine.py        # FastEmbed dense vectors, ChromaDB storage, and >75% score cutoff tests
│   ├── test_llm_manager.py       # Claude 3 key rotation, Gemini 1.5 fallback, and error handling tests
│   ├── test_resume_builder.py    # Strict ATS prompt constraint, Markdown formatting, and PDF generation tests
│   ├── test_email_generator.py   # Recruiter regex extraction, fallback email, and cold email JSON tests
│   └── test_main_integration.py  # End-to-end tests for /jobs, /generate-resume/{id}, and daily background cron
├── .env.example                  # Template configuration for Claude keys, Gemini keys, and SerpApi
├── pyproject.toml                # Static type checking (Pyrefly) and project tool configuration
├── requirements.txt              # Production dependency specifications
├── Your_Assistant.postman_collection.json # Complete Postman Collection for testing all endpoints
└── README.md                     # Comprehensive architecture and API documentation
```

---

## Environment Setup & Installation

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 installed.
- (Optional) Git for version control.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure `.env` File
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Edit `.env` with your API keys:
```ini
# Comma-separated API keys for rotation & fallback
CLAUDE_KEYS=sk-ant-api03-key1,sk-ant-api03-key2
GEMINI_KEYS=AIzaSyKey1,AIzaSyKey2

# SerpApi Key for automated job scraping
SERPAPI_API_KEY=your_serpapi_api_key_here

# Database URL (defaults to sqlite:///./jobs.db)
DATABASE_URL=sqlite:///./jobs.db

# ChromaDB persistence directory
CHROMA_DB_PATH=./chroma_db
```

### 4. Provide Your Base Resume
Ensure your genuine, factual resume is placed in `data/resume.txt`. The RAG engine reads this file to compute cosine similarity against scraped job postings and enforces the anti-hallucination constraint.

### 5. Launch the Application
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
Interactive docs will be available at:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## Complete API Reference

---

### Primary Application Endpoints

#### 1. `GET /`
**Health Check & Easter Egg**
- **Description**: Verifies service liveness and confirms the `import antigravity` easter egg is active.
- **Rate Limit**: `30 requests/minute`
- **Request Headers**: None required
- **Query Parameters**: None
- **Request Body**: None

**Response `200 OK`**:
```json
{
  "service": "Your Assistant - Automated AI Job Search Backend",
  "status": "healthy",
  "version": "1.0.0",
  "easter_egg": "antigravity active"
}
```

---

#### 2. `GET /jobs`
**Search & Filter Discovered Jobs**
- **Description**: Retrieves stored job opportunities ranked by `match_score` descending. Supports optional keyword search across job titles and companies.
- **Rate Limit**: `60 requests/minute`
- **Query Parameters**:
  | Parameter | Type | Required | Default | Description |
  |---|---|---|---|---|
  | `job_keyword` | `string` | No | `null` | Keyword to search in job title or company name (e.g. `Python`, `AI`) |
  | `min_score` | `float` | No | `null` | Minimum match score threshold ($0.0 - 100.0$) |
  | `skip` | `integer` | No | `0` | Pagination offset |
  | `limit` | `integer` | No | `50` | Maximum results to return ($\le 100$) |
- **Request Body**: None

**Response `200 OK`**:
```json
[
  {
    "id": 1,
    "title": "Senior AI Systems Engineer",
    "company": "Anthropic",
    "job_link": "https://boards.greenhouse.io/anthropic/jobs/12345",
    "career_page_link": "https://anthropic.com/careers",
    "match_score": 89.42,
    "recruiter_email": "careers@anthropic.com",
    "created_at": "2026-09-07T08:00:00Z"
  }
]
```

---

#### 3. `POST /generate-resume/{id}`
**Generate Tailored ATS Resume & PDF for Job**
- **Description**: Loads the stored job matching `{id}`, applies the strict zero-hallucination constraint against the user's real resume, tailors the resume in Markdown, and renders a publication-ready PDF using ReportLab.
- **Rate Limit**: `10 requests/minute`
- **Path Parameters**:
  | Parameter | Type | Required | Description |
  |---|---|---|---|
  | `id` | `integer` | Yes | Target `JobHistory` record ID |
- **Request Body**: None

**Response `200 OK`**:
```json
{
  "status": "success",
  "job_id": 1,
  "job_title": "Senior AI Systems Engineer",
  "company": "Anthropic",
  "match_score": 89.42,
  "tailored_resume_markdown": "# Alex Rivera\n**AI Systems Engineer** | alex.rivera@example.com\n\n## PROFESSIONAL SUMMARY\n...",
  "pdf_filename": "resume_Anthropic_1.pdf",
  "pdf_path": "output/resume_Anthropic_1.pdf"
}
```

**Response `404 Not Found`**:
```json
{
  "detail": "Job record with ID 999 does not exist."
}
```

---

#### 4. `POST /generate-email/{id}`
**Generate Personalized Cold Outreach Email**
- **Description**: Extracts recruiter contact information from the stored job record, generates an engaging subject line, and drafts a concise, personalized cold email highlighting matched real skills.
- **Rate Limit**: `15 requests/minute`
- **Path Parameters**:
  | Parameter | Type | Required | Description |
  |---|---|---|---|
  | `id` | `integer` | Yes | Target `JobHistory` record ID |
- **Request Body**: None

**Response `200 OK`**:
```json
{
  "email": "careers@anthropic.com",
  "subject": "Senior AI Systems Engineer - Alex Rivera | Python & RAG Infrastructure",
  "body": "Hi Anthropic Recruiting Team,\n\nI noticed your opening for a Senior AI Systems Engineer..."
}
```

---

### Modular & Programmatic Endpoints

#### 5. `POST /api/jobs`
**Manually Record Job Opportunity**
- **Description**: Stores a discovered job opportunity with strict input sanitization. Rejects unknown properties (`extra="forbid"`) to prevent payload injections.
- **Rate Limit**: `20 requests/minute`
- **Request Body (`application/json`)**:
```json
{
  "title": "Machine Learning Engineer",
  "company": "Google DeepMind",
  "job_link": "https://deepmind.google/careers/ml-eng-456",
  "career_page_link": "https://deepmind.google/careers",
  "match_score": 92.5,
  "recruiter_email": "recruiting@deepmind.com"
}
```

**Response `201 Created`**:
```json
{
  "id": 2,
  "title": "Machine Learning Engineer",
  "company": "Google DeepMind",
  "job_link": "https://deepmind.google/careers/ml-eng-456",
  "career_page_link": "https://deepmind.google/careers",
  "match_score": 92.5,
  "recruiter_email": "recruiting@deepmind.com",
  "created_at": "2026-09-07T08:15:00Z"
}
```

**Response `422 Unprocessable Entity`**: If payload contains unknown fields, invalid URLs, or unverified email formatting.

---

#### 6. `GET /api/jobs`
**Paginated Query of Job Matches**
- **Description**: Paginated retrieval of job records with optional filtering by minimum match score and company name.
- **Rate Limit**: `60 requests/minute`
- **Query Parameters**:
  | Parameter | Type | Default | Description |
  |---|---|---|---|
  | `skip` | `integer` | `0` | Offset for pagination |
  | `limit` | `integer` | `50` | Maximum items to return ($\le 100$) |
  | `min_score` | `float` | `null` | Filter by minimum match score |
  | `company` | `string` | `null` | Filter by employer company name |

**Response `200 OK`**: List of `JobHistoryResponse` objects.

---

#### 7. `POST /api/jobs/scrape`
**On-Demand Job Scraping via SerpApi**
- **Description**: Scrapes Google Jobs via SerpApi. Prioritizes `keyword` if provided, otherwise falls back to `"Software Engineer remote"`. Predicts employer career portals and automatically omits any job URLs saved within the last 7 days.
- **Rate Limit**: `10 requests/minute`
- **Query Parameters**:
  | Parameter | Type | Default | Description |
  |---|---|---|---|
  | `keyword` | `string` | `null` | Target job title keyword |
  | `limit` | `integer` | `10` | Maximum jobs to fetch ($1 - 50$) |

**Response `200 OK`**:
```json
[
  {
    "title": "Backend Software Engineer",
    "company": "Stripe",
    "description": "We are looking for an experienced backend engineer to scale global payment APIs...",
    "job_link": "https://stripe.com/jobs/backend-engineer-789",
    "career_page_link": "https://stripe.com/careers"
  }
]
```

---

#### 8. `POST /api/jobs/match`
**Vector RAG Match Evaluation**
- **Description**: Takes a raw list of scraped jobs, computes dense embeddings using FastEmbed (`BAAI/bge-small-en-v1.5`), performs cosine similarity search against the user's resume in ChromaDB, and returns **only jobs strictly exceeding the cutoff score** (default: `> 75%`).
- **Rate Limit**: `10 requests/minute`
- **Query Parameters**:
  | Parameter | Type | Default | Description |
  |---|---|---|---|
  | `min_score` | `float` | `75.0` | Minimum match percentage threshold ($0.0 - 100.0$) |
- **Request Body (`application/json`)**:
```json
[
  {
    "title": "Python Distributed Systems Engineer",
    "company": "Snowflake",
    "description": "Build high-throughput distributed database query engines with Python and C++.",
    "job_link": "https://snowflake.com/jobs/dist-sys-101",
    "career_page_link": "https://snowflake.com/careers"
  }
]
```

**Response `200 OK`**:
```json
[
  {
    "title": "Python Distributed Systems Engineer",
    "company": "Snowflake",
    "description": "Build high-throughput distributed database query engines with Python and C++.",
    "job_link": "https://snowflake.com/jobs/dist-sys-101",
    "career_page_link": "https://snowflake.com/careers",
    "match_score": 83.15,
    "recruiter_email": null
  }
]
```

---

#### 9. `POST /api/ai/generate`
**Direct Multi-Tier AI Generation**
- **Description**: Unified text generation using multi-key rotation and multi-provider fallback: Claude 3 (Key 1 -> Key 2) cascading to Gemini 1.5 (Key 1 -> Key 2).
- **Rate Limit**: `15 requests/minute`
- **Request Body (`application/json`)**:
```json
{
  "prompt": "Draft a 2-sentence elevator pitch for a Senior Python Developer with FastAPI and RAG expertise.",
  "system_prompt": "You are an executive tech career coach."
}
```

**Response `200 OK`**:
```json
{
  "status": "success",
  "response": "With deep expertise in architecting asynchronous Python services using FastAPI and local vector RAG pipelines, I build production AI backends that operate with ultra-low latency and zero downtime. My work focuses on scalable system design, robust rate limiting, and high-precision retrieval systems."
}
```

**Response `503 Service Unavailable`**: If all Claude and Gemini keys are exhausted.

---

#### 10. `POST /api/resume/tailor`
**Tailor Resume Markdown (No Fake Skills)**
- **Description**: Tailors resume Markdown specifically aligned to a job description while strictly prohibiting skill hallucination.
- **Rate Limit**: `10 requests/minute`
- **Request Body (`application/json`)**:
```json
{
  "job_title": "AI Backend Architect",
  "job_description": "Seeking an engineer with strong FastAPI, ChromaDB, and LLM fallback architecture experience.",
  "company": "Cohere",
  "base_resume_text": null
}
```

**Response `200 OK`**:
```json
{
  "status": "success",
  "job_title": "AI Backend Architect",
  "company": "Cohere",
  "tailored_resume_markdown": "# Candidate Name\n**AI Backend Architect**\n\n## SUMMARY\n..."
}
```

---

#### 11. `POST /api/resume/generate-pdf`
**Convert Markdown Resume to ATS PDF**
- **Description**: Compiles customized Markdown resume text into a publication-quality, ATS-optimized PDF and returns it as a direct file download.
- **Rate Limit**: `10 requests/minute`
- **Request Body (`application/json`)**:
```json
{
  "markdown_text": "# Alex Rivera\n**AI Systems Engineer** | alex.rivera@example.com\n\n## PROFESSIONAL SUMMARY\nBackend engineer specialized in FastAPI, ChromaDB, and Python.",
  "filename": "alex_rivera_resume.pdf"
}
```

**Response `200 OK`**:
- **Content-Type**: `application/pdf`
- **Content-Disposition**: `attachment; filename="alex_rivera_resume.pdf"`
- **Body**: Binary PDF file.

---

#### 12. `POST /api/email/generate`
**Arbitrary Cold Email Generation**
- **Description**: Generates a high-impact recruiter cold email for arbitrary job descriptions without needing a pre-existing database record.
- **Rate Limit**: `15 requests/minute`
- **Request Body (`application/json`)**:
```json
{
  "job_title": "Senior Data Platform Engineer",
  "job_description": "We are seeking a senior engineer to scale streaming data pipelines. Contact hiring-lead@databricks.com for questions.",
  "company": "Databricks",
  "recruiter_email": null,
  "base_resume_text": null
}
```

**Response `200 OK`**:
```json
{
  "email": "hiring-lead@databricks.com",
  "subject": "Senior Data Platform Engineer - Alex Rivera | Distributed Systems Specialist",
  "body": "Dear Databricks Team,\n\nI was excited to see your opening for a Senior Data Platform Engineer..."
}
```

---

#### 13. `POST /api/jobs/cleanup`
**Trigger Retention Cleanup**
- **Description**: Manually runs the data retention worker to delete jobs older than `retention_days` (default: 7 days).
- **Rate Limit**: `5 requests/minute`
- **Query Parameters**:
  | Parameter | Type | Default | Description |
  |---|---|---|---|
  | `retention_days` | `integer` | `7` | Retention threshold in days ($1 - 365$) |
- **Request Body**: None

**Response `200 OK`**:
```json
{
  "status": "success",
  "retention_days": 7,
  "records_deleted": 14
}
```

---

## Testing with Postman

A complete, production-ready Postman collection is included in the root directory:
[`Your_Assistant.postman_collection.json`](file:///d:/Nurix_Hive/Your_Assistant/Your_Assistant.postman_collection.json)

### How to Import & Use:
1. Open **Postman**.
2. Click **Import** in the top-left corner.
3. Select `Your_Assistant.postman_collection.json`.
4. The collection defines a collection variable:
   - `baseUrl`: `http://127.0.0.1:8000` (modify if hosting on a remote server).
5. All 13 endpoints are organized into logical folders:
   - **01 - Health & Easter Egg**
   - **02 - Main Discovery & Generation**
   - **03 - Job Database Operations**
   - **04 - Scraper & RAG Engine**
   - **05 - AI & ATS Resume Builder**
   - **06 - Cold Outreach & Email**
6. Each request includes pre-configured headers (`Content-Type: application/json`), realistic sample payloads, and automated test scripts to verify `responseCode.code === 200` or `201`.

---

## Automated Test Suite

Run the full test suite using `pytest`:
```bash
python -m pytest -v
```

### Test Coverage Highlights (35 / 35 Passing Tests)
- `tests/test_backend.py`: Antigravity easter egg, `JobHistory` CRUD, SlowAPI rate limiting (`429`), strict Pydantic payload validation, and automated 7-day database cleanup.
- `tests/test_scraper.py`: Keyword query building, default fallback query, career portal URL prediction, 7-day duplicate filtering, and scraping endpoint integration.
- `tests/test_rag_engine.py`: FastEmbed BGE dense vector generation, ChromaDB cosine space persistence, resume loading, and strict $> 75\%$ score threshold cutoff.
- `tests/test_llm_manager.py`: Multi-key loading from `.env`, Claude 3 key rotation on 429 errors, Claude-to-Gemini cascading fallback, and unified `generate_ai_response()`.
- `tests/test_resume_builder.py`: Strict anti-hallucination prompt enforcement, Markdown sanitization, ReportLab PDF rendering, and `/api/resume/*` endpoints.
- `tests/test_email_generator.py`: Regex recruiter email extraction, `[Recruiter Email]` fallback handling, structured JSON formatting, and `/api/email/*` endpoints.
- `tests/test_main_integration.py`: End-to-end integration across `/jobs`, `/generate-resume/{id}`, `/generate-email/{id}`, and the 24-hour background scheduler pipeline.

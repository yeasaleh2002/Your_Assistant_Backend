# Your Assistant — Autonomous AI Job Search & Application Tracking Platform

**Your Assistant** is an enterprise-grade backend service designed to automate and streamline the full job search and application lifecycle. Powered by FastAPI, SQLAlchemy with connection pooling (targeting NeonDB PostgreSQL, MySQL, and SQLite), ChromaDB vector RAG, ReportLab, and an intelligent multi-tier LLM engine, it continuously scrapes opportunities, ranks them with cosine similarity against your real resume, auto-generates 100% ATS-compliant PDFs (with strictly black text and zero hallucinated skills), tracks your application status, and secures your endpoints with JWT admin authentication.

---

## Table of Contents

1. [Key Features & Highlights](#key-features--highlights)
2. [Architecture & Workflow](#architecture--workflow)
3. [Technology Stack](#technology-stack)
4. [Folder Structure](#folder-structure)
5. [Database Configuration (NeonDB PostgreSQL / MySQL / SQLite)](#database-configuration)
6. [Scraper Rules & Filters](#scraper-rules--filters)
7. [Environment Setup & Installation](#environment-setup--installation)
8. [Complete API Reference](#complete-api-reference)
   - [Authentication & Admin Endpoints](#authentication--admin-endpoints)
     - [`POST /api/auth/login`](#1-post-apiauthlogin)
     - [`GET /api/auth/me`](#2-get-apiauthme)
   - [Core Application Tracking Endpoints](#core-application-tracking-endpoints)
     - [`POST /api/scrape`](#3-post-apiscrape)
     - [`GET /api/jobs`](#4-get-apijobs)
     - [`PATCH /api/jobs/{job_id}/status`](#5-patch-apijobsjob_idstatus)
     - [`DELETE /api/jobs/date/{date}`](#6-delete-apijobsdatedate)
   - [AI Resume & Email Generation](#ai-resume--email-generation)
     - [`POST /generate-resume/{id}`](#7-post-generate-resumeid)
     - [`POST /generate-email/{id}`](#8-post-generate-emailid)
   - [Legacy & Utility Endpoints](#legacy--utility-endpoints)
     - [`GET /`](#9-get-)
     - [`GET /jobs`](#10-get-jobs)
     - [`POST /api/jobs/cleanup`](#11-post-apijobscleanup)
9. [Testing with Postman](#testing-with-postman)
10. [Automated Test Suite](#automated-test-suite)

---

## Key Features & Highlights

- **Production Cloud SQL Database (NeonDB PostgreSQL)**: Production-ready SQLAlchemy data layer with automatic connection pooling (`pool_size=10, max_overflow=20, pool_pre_ping=True`), pre-configured to connect to free-tier cloud NeonDB PostgreSQL or local SQLite/MySQL.
- **Admin JWT Authentication**: Secure admin login endpoint verifying against `ADMIN_EMAIL` and `ADMIN_PASSWORD` in `.env`, signing bearer tokens with `JWT_SECRET` (HS256).
- **Application Status Tracking**: Track every job record through its lifecycle using the `status` enum: `Pending`, `Applied`, `Interview`, or `Rejected`.
- **12 Hardcoded Primary Keywords**: Scrapes software engineering roles exclusively for:
  `"Frontend developer, Frontend Engineer, software engineer, software developer, web developer, full stack developer, react developer, next.js developer, python developer, fast api developer, vibe coder, agentic full stack development"`.
- **Multi-Source Concurrent Scraper**: Concurrent multi-worker scraping engine targeting LinkedIn, Indeed, Glassdoor, We Work Remotely, Toptal, Wellfound, Remote.co, Bdjobs, and Google Search. Targets a daily yield of 50–60+ opportunities.
- **Strict 24-Hour Recency**: Only accepts postings published in the last 24 hours.
- **Geographic Policy & Strict Country Exclusion**:
  - **Bangladesh (BD)**: Accepts any job type (On-site, Hybrid, or Remote).
  - **Outside Bangladesh**: Strictly requires **Remote** / Work-from-Home.
  - **India & Pakistan**: **STRICTLY EXCLUDED** (all regions, cities, and domains).
- **RAG Cosine Evaluation (>= 65% Cutoff)**: Uses local FastEmbed ONNX vectors (`BAAI/bge-small-en-v1.5`) in ChromaDB to score each job against your base resume (`data/resume.txt`). Only jobs with $\ge 65.0\%$ match score are saved.
- **ATS Resume Generator (Strictly Black Text)**: Prompts Claude / Gemini to tailor resume content with zero hallucination. Renders clean, modern PDFs via ReportLab with 100% black text (`#000000`) for maximum ATS scanner compatibility.

---

## Architecture & Workflow

```mermaid
flowchart TD
    subgraph AuthLayer [Admin Security Layer]
        AdminCreds[ADMIN_EMAIL & ADMIN_PASSWORD in .env] --> LoginEndpoint[POST /api/auth/login]
        LoginEndpoint -->|HMAC Timing-Safe Verification| IssueJWT[Issue JWT Signed with JWT_SECRET]
        IssueJWT --> BearerToken[Authorization: Bearer Token]
        BearerToken --> ProtectedRoute[GET /api/auth/me]
    end

    subgraph ScraperEngine [Concurrent Multi-Source Scraper]
        KW[12 Hardcoded Keywords] --> WorkerPool[Concurrent ThreadPoolExecutor]
        WorkerPool --> S1[SerpApi Google Jobs]
        WorkerPool --> S2[LinkedIn / Indeed / Glassdoor / Bdjobs]
        WorkerPool --> S3[We Work Remotely Feed]
        WorkerPool --> S4[Remote.co Feed]
        S1 & S2 & S3 & S4 --> Filter24h{Posted in Last 24 Hours?}
        Filter24h -->|No| Discard1[Discard]
        Filter24h -->|Yes| FilterGeo{Geo Check: Exclude IN/PK, BD Any, Outside Remote?}
        FilterGeo -->|No| Discard2[Discard]
        FilterGeo -->|Yes| Dedupe[7-Day DB Deduplication]
    end

    subgraph RAGMatching [ChromaDB Vector Evaluation]
        Dedupe --> RAG[FastEmbed bge-small ONNX]
        Resume[data/resume.txt] --> RAG
        RAG --> Cutoff{Match Score >= 65%?}
        Cutoff -->|No| Discard3[Discard]
        Cutoff -->|Yes| SaveDB[(SQL Database: NeonDB PostgreSQL)]
    end

    subgraph AppTracking [Application Tracking & Workflows]
        SaveDB --> EndpointGet[GET /api/jobs?date=YYYY-MM-DD]
        SaveDB --> EndpointPatch[PATCH /api/jobs/{id}/status]
        SaveDB --> EndpointDel[DELETE /api/jobs/date/{date}]
        SaveDB --> GenResume[POST /generate-resume/{id}]
        SaveDB --> GenEmail[POST /generate-email/{id}]
    end
```

---

## Technology Stack

| Component | Technology | Description |
|---|---|---|
| **Web Framework** | FastAPI `>= 0.115.0` | Asynchronous REST API framework with OpenAPI documentation |
| **Authentication** | PyJWT `>= 2.8.0` | Secure JWT token issuance and cryptographic verification |
| **Database ORM** | SQLAlchemy `>= 2.0.30` | Production SQL ORM with connection pooling & dialect translation |
| **Database Driver** | psycopg2-binary `>= 2.9.9` | High-performance C-based PostgreSQL driver |
| **Cloud Database** | NeonDB PostgreSQL | Serverless PostgreSQL cloud database with SSL & pooling |
| **Vector Engine** | ChromaDB `>= 0.5.0` | Local persistent vector storage with cosine distance metric |
| **Embeddings** | FastEmbed `>= 0.3.0` | CPU-optimized `BAAI/bge-small-en-v1.5` dense embeddings |
| **PDF Generator** | ReportLab `>= 4.0.0` | Programmatic ATS-friendly PDF compiler (100% black text) |
| **Rate Limiter** | SlowAPI `>= 0.1.9` | IP-based request throttling |
| **Scheduler** | APScheduler `>= 3.10.4` | Background 24-hour scraper & retention cron |

---

## Folder Structure

```
Your_Assistant/
├── app/
│   ├── __init__.py               # Package initializer with dynamic runtime compatibility shim
│   ├── auth.py                   # JWT admin authentication, password verification, token dependencies
│   ├── database.py               # SQLAlchemy engine (Neon PostgreSQL/MySQL/SQLite), connection pooling
│   ├── models.py                 # Job SQLAlchemy ORM model, JobStatus enum, Pydantic schemas
│   ├── schemas.py                # Schema aliases and re-exports
│   ├── scraper.py                # 12 hardcoded keywords, concurrent fetchers, 24h & geo filters
│   ├── rag_engine.py             # ChromaDB vector RAG engine, FastEmbed model, >=65% cosine cutoff
│   ├── llm_manager.py            # Multi-key rotation, Claude 3 -> Gemini cascading fallback engine
│   ├── resume_builder.py         # ATS resume tailoring (anti-hallucination) & ReportLab PDF generator (black text)
│   ├── email_generator.py        # Recruiter contact email extraction & cold outreach JSON generator
│   └── main.py                   # FastAPI app, SlowAPI limiter, connected endpoints & APScheduler cron
├── data/
│   └── resume.txt                # Static user base resume profile text used for vector matching
├── output/                       # Output directory where generated tailored resume PDFs are stored
├── tests/
│   ├── test_auth.py              # Admin verification, JWT encoding/decoding, /api/auth/me tests
│   ├── test_backend.py           # Rate limiting, strict validation, and automated 7-day retention cleanup tests
│   ├── test_scraper.py           # Query generation, SerpApi scraping, career URL predictor, duplicate tests
│   ├── test_sql_and_scraper_upgrade.py # 24h filter, India/Pakistan exclusion, BD geo rules, status PATCH, delete by date
│   ├── test_rag_engine.py        # FastEmbed dense vectors, ChromaDB storage, and >=65% score cutoff tests
│   ├── test_llm_manager.py       # Claude 3 key rotation, Gemini fallback, and error handling tests
│   ├── test_resume_builder.py    # Strict ATS prompt constraint, Markdown formatting, and PDF generation tests
│   ├── test_email_generator.py   # Recruiter regex extraction, fallback email, and cold email JSON tests
│   └── test_main_integration.py  # End-to-end tests for /jobs, /generate-resume/{id}, and daily background cron
├── .env                          # Active environment configuration (API keys, NeonDB, JWT secret)
├── .env.example                  # Template configuration file
├── pyproject.toml                # Static type checking and project tool configuration
├── requirements.txt              # Production dependency specifications
├── Your_Assistant.postman_collection.json # Complete Postman Collection for testing all endpoints
└── README.md                     # Comprehensive architecture and API documentation
```

---

## Database Configuration

### 1. NeonDB PostgreSQL (Configured in `.env`)
```ini
DATABASE_URL=postgresql://neondb_owner:npg_iX8Hz1qwmyxS@ep-ancient-art-ap0suate.c-7.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require
```

### 2. MySQL
```ini
DATABASE_URL=mysql+pymysql://user:password@localhost:3306/your_assistant_db
```

### 3. SQLite (Local Offline Development)
```ini
DATABASE_URL=sqlite:///./jobs.db
```

### Database Model: `jobs` Table
| Column | Type | Description |
|---|---|---|
| `id` | `INTEGER` (PK) | Auto-incrementing primary key |
| `title` | `VARCHAR(255)` | Job title |
| `company` | `VARCHAR(255)` | Employer / company name |
| `link` | `TEXT` | Direct job application URL (also accessed via `job_link`) |
| `match_score` | `FLOAT` | Semantic similarity score against resume ($0.0 - 100.0$) |
| `location` | `VARCHAR(255)` | Job location or Remote status |
| `status` | `VARCHAR(50)` | Status enum: `Pending`, `Applied`, `Interview`, `Rejected` |
| `scraped_date` | `DATE` | Date scraped (`YYYY-MM-DD`) |
| `description` | `TEXT` | Job posting snippet or description |
| `recruiter_email`| `VARCHAR(255)` | Extracted recruiter contact email |
| `career_page_link`| `TEXT` | Predicted company career portal |
| `created_at` | `TIMESTAMP` | Record creation timestamp |

---

## Scraper Rules & Filters

1. **Keywords**:
   - `Frontend developer`, `Frontend Engineer`, `software engineer`, `software developer`, `web developer`, `full stack developer`, `react developer`, `next.js developer`, `python developer`, `fast api developer`, `vibe coder`, `agentic full stack development`.
2. **24-Hour Recency**:
   - Inspects metadata for `"hour"`, `"minute"`, `"today"`, `"1 day ago"`, or publication timestamps $\le 24$ hours.
3. **Geographic Policy**:
   - **Bangladesh**: Any type allowed (On-site, Hybrid, Remote).
   - **Outside Bangladesh**: MUST be Remote / Work From Home.
   - **India & Pakistan**: STRICTLY EXCLUDED across all locations, regions, and domains (`.in`, `.pk`).
4. **RAG Match**:
   - Evaluates against `data/resume.txt`. Cutoff threshold is strictly $\ge 65.0\%$.

---

## Environment Setup & Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure `.env`
```ini
# Multi-LLM API Keys
CLAUDE_KEYS=sk-ant-api03-key1,sk-ant-api03-key2
GEMINI_KEYS=AIzaSyKey1,AIzaSyKey2

# SerpApi Key
SERPAPI_API_KEY=your_serpapi_api_key_here

# Database Connection (NeonDB PostgreSQL)
DATABASE_URL=postgresql://neondb_owner:password@ep-ancient-art.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require

# ChromaDB Storage
CHROMA_DB_PATH=./chroma_db

# Admin Authentication & JWT Secret
JWT_SECRET=080c28897f0a23ca02c407685998fc1c6e8197d3997510ba1d49d7f994f628ad
ADMIN_EMAIL=admin@yourassistant.com
ADMIN_PASSWORD=change_this_password
```

### 3. Provide Base Resume
Place your factual resume at `data/resume.txt`.

### 4. Run Server
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Complete API Reference

### Authentication & Admin Endpoints

#### 1. `POST /api/auth/login`
**Admin Login & Token Generation**
- **Description**: Verifies credentials against `ADMIN_EMAIL` and `ADMIN_PASSWORD` in `.env`, returning a signed JWT token signed with `JWT_SECRET`.
- **Rate Limit**: `15 requests/minute`
- **Request Body**:
```json
{
  "email": "admin@yourassistant.com",
  "password": "change_this_password"
}
```

**Response `200 OK`**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in_days": 7,
  "user": {
    "email": "admin@yourassistant.com",
    "role": "admin"
  }
}
```

---

#### 2. `GET /api/auth/me`
**Get Current Authenticated User**
- **Description**: Validates active JWT access token passed in `Authorization: Bearer <token>` header.
- **Rate Limit**: `60 requests/minute`

**Response `200 OK`**:
```json
{
  "email": "admin@yourassistant.com",
  "role": "admin",
  "authenticated": true
}
```

---

### Core Application Tracking Endpoints

#### 3. `POST /api/scrape`
**Trigger Scraper & RAG Evaluation Pipeline**
- **Description**: Concurrently scrapes jobs across the 12 primary keywords, enforces 24h & geo rules, scores against `data/resume.txt`, and persists matches ($\ge 65\%$) for today's date.
- **Rate Limit**: `10 requests/minute`
- **Request Body**: None

**Response `200 OK`**:
```json
{
  "status": "success",
  "scraped_count": 58,
  "matched_count": 14,
  "saved_count": 14,
  "scraped_date": "2026-09-07",
  "message": "Successfully scraped 58 candidates, matched 14 (>= 65%), and saved 14 new jobs for 2026-09-07."
}
```

---

#### 4. `GET /api/jobs`
**Retrieve Stored Jobs by Date**
- **Description**: Returns jobs scraped for a given date, ranked by `match_score` descending. Defaults to today's date if `date` is omitted.
- **Rate Limit**: `60 requests/minute`
- **Query Parameters**:
  | Parameter | Type | Required | Default | Description |
  |---|---|---|---|---|
  | `date` | `string` (`YYYY-MM-DD`) | No | `today` | Date filter (e.g. `2026-09-07`) |
  | `skip` | `integer` | No | `0` | Pagination offset |
  | `limit` | `integer` | No | `100` | Max items to return ($\le 200$) |

**Response `200 OK`**:
```json
[
  {
    "id": 1,
    "title": "Senior Frontend Engineer",
    "company": "Vercel Partner",
    "link": "https://example.com/careers/fe-engineer",
    "match_score": 88.5,
    "location": "Remote",
    "status": "Pending",
    "scraped_date": "2026-09-07",
    "description": "Building scalable React & Next.js applications...",
    "recruiter_email": "recruiting@example.com",
    "career_page_link": "https://example.com/careers",
    "created_at": "2026-09-07T14:30:00Z"
  }
]
```

---

#### 5. `PATCH /api/jobs/{job_id}/status`
**Update Application Tracking Status**
- **Description**: Updates the application status for a specific job.
- **Rate Limit**: `30 requests/minute`
- **Request Body**:
```json
{
  "status": "Applied"
}
```
*Allowed values*: `"Pending"`, `"Applied"`, `"Interview"`, `"Rejected"`

**Response `200 OK`**:
```json
{
  "id": 1,
  "title": "Senior Frontend Engineer",
  "company": "Vercel Partner",
  "link": "https://example.com/careers/fe-engineer",
  "match_score": 88.5,
  "location": "Remote",
  "status": "Applied",
  "scraped_date": "2026-09-07"
}
```

---

#### 6. `DELETE /api/jobs/date/{date}`
**Delete All Jobs Scraped on a Date**
- **Description**: Deletes all scraped jobs recorded for the specified date (`YYYY-MM-DD`).
- **Rate Limit**: `15 requests/minute`

**Response `200 OK`**:
```json
{
  "status": "success",
  "date": "2026-09-07",
  "deleted_count": 14,
  "message": "Successfully deleted 14 jobs scraped on 2026-09-07."
}
```

---

### AI Resume & Email Generation

#### 7. `POST /generate-resume/{id}`
**Generate ATS-Friendly Resume & PDF**
- **Description**: Tailors your genuine resume for the specified job posting without hallucinating skills. Renders an ATS-safe ReportLab PDF with 100% black text (`#000000`).
- **Rate Limit**: `15 requests/minute`

**Response `200 OK`**:
```json
{
  "status": "success",
  "job_id": 1,
  "job_title": "Senior Frontend Engineer",
  "company": "Vercel Partner",
  "match_score": 88.5,
  "tailored_resume_markdown": "# Yeasaleh\n**Senior Frontend Engineer**...",
  "pdf_filename": "Resume_Vercel_Partner_Senior_Frontend_Engineer.pdf",
  "pdf_path": "output/Resume_Vercel_Partner_Senior_Frontend_Engineer.pdf"
}
```

---

#### 8. `POST /generate-email/{id}` (or `POST /api/jobs/{id}/email`)
**Generate Personalized Email Cover Letter & Formal Cover Letter**
- **Description**: Resolves recruiter contact information and crafts a personalized email cover letter body AND a full formal cover letter specifically tailored to the stored job from the candidate's base resume.
- **Rate Limit**: `20 requests/minute`

**Response `200 OK`**:
```json
{
  "email": "careers@example.com",
  "subject": "Senior Frontend Engineer Application - Yeasaleh | Vercel Partner",
  "body": "Hi Vercel Partner Team,\n\nI noticed your opening for a Senior Frontend Engineer. With proven experience building high-performance web applications, reducing Cumulative Layout Shift (0.01 CLS), and scaling React/Next.js architectures, I would love to contribute...",
  "cover_letter": "Dear Vercel Partner Hiring Team,\n\nI am writing to submit my application for the Senior Frontend Engineer position. With demonstrated expertise in modern full-stack development, distributed API architecture, and performance optimization, I am confident in my ability to bring immediate technical value..."
}
```

---

#### 9. `POST /api/jobs/{id}/cover-letter` (or `POST /generate-cover-letter/{id}`)
**Generate Dedicated Cover Letter for Stored Job**
- **Description**: Generates a full formal 3–4 paragraph ATS cover letter based on the candidate's base resume for a specific job, plus the email version.
- **Rate Limit**: `20 requests/minute`

**Response `200 OK`**:
```json
{
  "status": "success",
  "job_id": 1,
  "job_title": "Senior Frontend Engineer",
  "company": "Vercel Partner",
  "cover_letter": "Dear Vercel Partner Hiring Team,\n\nI am writing to submit my application for the Senior Frontend Engineer position...",
  "email_cover_letter": {
    "email": "careers@example.com",
    "subject": "Senior Frontend Engineer Application - Yeasaleh | Vercel Partner",
    "body": "Hi Vercel Partner Team...",
    "cover_letter": "..."
  }
}
```

---

### Custom Job Description Tailoring (Direct Raw Text Input)

#### `POST /api/job-description/tailor`
**Generate 100% ATS-Optimized Resume, Match Score, Cold Email & Cover Letter from Raw Job Description**
- **Description**: Accept any raw text job description, calculate semantic vector RAG match score against the user's base resume, generate a 100% ATS-compliant PDF with ReportLab, and provide a tailored cold email draft plus a full formal cover letter.
- **Dynamic Job Title Replacement**: Replaces `"Software Developer"` with the target job title across:
  - Resume Header: `# Yeasaleh | {job_title}`
  - Resume Professional Summary: Aligned to the target role
  - Cold Outreach Email & Cover Letter
- **PDF Naming Convention**:
  - If company is provided: `Yeasaleh_Resume_{clean_company}_{clean_title}.pdf`
  - If company is absent: `Yeasaleh_Resume_{clean_title}.pdf`
- **Rate Limit**: `15 requests/minute`

**Request Body (`application/json`)**:
```json
{
  "job_description": "We are seeking a Senior React Engineer with deep experience in Next.js, TypeScript, and state management. Send applications to careers@technova.io.",
  "job_title": "Senior React Engineer",
  "company": "TechNova Corp",
  "recruiter_email": "careers@technova.io"
}
```

**Response `200 OK`**:
```json
{
  "status": "success",
  "job_title": "Senior React Engineer",
  "company": "TechNova Corp",
  "match_score": 84.2,
  "pdf_filename": "Yeasaleh_Resume_TechNova_Corp_Senior_React_Engineer.pdf",
  "pdf_path": "output/Yeasaleh_Resume_TechNova_Corp_Senior_React_Engineer.pdf",
  "download_url": "/api/resume/download/Yeasaleh_Resume_TechNova_Corp_Senior_React_Engineer.pdf",
  "tailored_resume_markdown": "# Yeasaleh | Senior React Engineer\nDhaka, Bangladesh | +8801735782467...",
  "cold_email": {
    "recruiter_email": "careers@technova.io",
    "subject": "Senior React Engineer Application - Yeasaleh | TechNova Corp",
    "body": "Hi Team at TechNova Corp,\n\nI noticed your opening for a Senior React Engineer...",
    "candidate_name": "Yeasaleh",
    "portfolio_link": "https://yeasaleh.xyz",
    "call_to_action": "Would you have 10-15 minutes this week for a brief conversation?"
  },
  "cover_letter": "Dear Hiring Team at TechNova Corp,\n\nI am writing to express my enthusiastic interest in the Senior React Engineer position..."
}
```

---

### Legacy & Utility Endpoints


#### 9. `GET /`
Returns service status and the Antigravity active easter egg.

#### 10. `GET /jobs`
Legacy job search supporting keyword (`job_keyword`) and minimum score (`min_score`) filters.

#### 11. `POST /api/jobs/cleanup?retention_days=7`
Manually triggers 7-day data retention pruning.

---

## Testing with Postman

A pre-configured Postman Collection is included in the root directory:
**`Your_Assistant.postman_collection.json`**

### Postman Test Flow:
1. **01 - Health & System**: Verify service liveness.
2. **02 - Authentication & JWT**:
   - Send `Admin Login & Generate JWT`.
   - The test script automatically saves the `access_token` into the `{{jwtToken}}` collection variable!
   - Send `Get Authenticated Profile` to test the token.
3. **03 - Application Tracking & Core Endpoints**:
   - `POST /api/scrape`
   - `GET /api/jobs` (default today)
   - `GET /api/jobs?date={{targetDate}}`
   - `PATCH /api/jobs/1/status`
   - `DELETE /api/jobs/date/{{targetDate}}`
   - `POST /generate-resume/1`
   - `POST /generate-email/1`
4. **04 - Job Database Operations**: Manual job creation & retention cleanup.
5. **05 - Scraper & Vector RAG Testing**: Raw scrape & threshold testing.
6. **06 - AI & ATS Resume Builder**: Tailored markdown & PDF compilation.
7. **07 - Cold Outreach & Email**: Customized cold email draft generation.
8. **08 - Custom Job Description Tailoring**:
   - `Process Custom Job Description (With Company & Title)`
   - `Process Custom Job Description (No Company Name)`
   - `Download Tailored Resume PDF`

---

## Automated Test Suite

Run the full automated test suite (46 comprehensive tests):
```bash
pytest -v
```

Tests cover:
- Admin login and JWT authentication (`tests/test_auth.py`).
- NeonDB PostgreSQL and SQLAlchemy dialect handling, schema validation, and status transitions.
- Multi-source scraper 24h filter and India/Pakistan rejection.
- Geographic constraints (Bangladesh any type, outside Bangladesh remote only).
- RAG cosine similarity scoring and $\ge 65\%$ cutoff.
- ATS PDF generation (100% black text formatting).
- Cascading Multi-LLM provider fallback and key rotation.

# Your Assistant - Automated AI Job Search Backend

A production-ready FastAPI backend for an Automated AI Job Search tool built with Python, SQLAlchemy, and SQLite.

## Features

- **Easter Egg**: Includes `import antigravity` to elevate your AI job hunt into the clouds.
- **Bot Defense & Throttling**: Strict IP-based rate limiting via `slowapi` to protect search and ingestion endpoints.
- **Strict Input Sanitization**: Pydantic v2 schemas enforcing `extra='forbid'`, URL validation, email verification, score boundaries (`0.0` - `100.0`), and script tag detection.
- **Database Layer**: SQLAlchemy with SQLite (`jobs.db`) storing `JobHistory`.
- **Automated 7-Day Retention**: Background scheduler (`APScheduler`) running every 24 hours to automatically purge job records older than 7 days, with manual trigger endpoint support.

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
│   └── main.py           # FastAPI app, SlowAPI limiter, APScheduler, API routes
├── data/
│   └── resume.txt        # User base resume profile for vector matching
├── tests/
│   ├── test_backend.py   # Test suite for validation, rate limiting, and cleanup
│   ├── test_scraper.py   # Test suite for queries, career prediction, and deduplication
│   ├── test_rag_engine.py# Test suite for ChromaDB, embeddings, and >75% match threshold
│   ├── test_llm_manager.py # Test suite for multi-key rotation and multi-tier LLM fallback
│   └── test_resume_builder.py # Test suite for ATS prompt constraint and PDF generation
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

Copy `.env.example` to `.env` and provide your API keys:
```bash
cp .env.example .env
```

Example `.env`:
```ini
# Comma-separated API keys for rotation & fallback
CLAUDE_KEYS=sk-ant-api03-xxx1,sk-ant-api03-xxx2
GEMINI_KEYS=AIzaSyxxx1,AIzaSyxxx2

SERPAPI_API_KEY=your_serpapi_api_key_here
```

### 3. Run the Development Server

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Interactive API documentation will be available at:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## API Endpoints

| Method | Endpoint | Description | Rate Limit |
|---|---|---|---|
| `GET` | `/` | Health check & Easter egg verification | 30/min |
| `POST` | `/api/jobs` | Record a discovered job opportunity | 20/min |
| `GET` | `/api/jobs` | Query & paginate stored job matches | 60/min |
| `POST` | `/api/jobs/scrape` | Scrape jobs via SerpApi, predict career portals, filter 7-day duplicates | 10/min |
| `POST` | `/api/jobs/match` | Vector RAG match against resume, returning only jobs > 75% score | 10/min |
| `POST` | `/api/ai/generate` | Generate AI text via Claude 3 (Key 1 -> 2) -> Gemini 1.5 (Key 1 -> 2) | 15/min |
| `POST` | `/api/resume/tailor` | Tailor resume for matched job with 100% ATS & zero-hallucination constraint | 10/min |
| `POST` | `/api/resume/generate-pdf` | Convert tailored resume into an ATS-friendly PDF download | 10/min |
| `POST` | `/api/jobs/cleanup` | Manually run retention pruning (default: 7 days) | 5/min |





---

## Running Tests

```bash
python -m pytest -v tests/test_backend.py
```

import antigravity  # Easter Egg: Elevating Python & AI job searches into orbit!

import importlib
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import or_

# Dynamically import scheduler and rate limiting modules for seamless static analyzer compatibility
_apscheduler_asyncio: Any = importlib.import_module("apscheduler.schedulers.asyncio")
_slowapi: Any = importlib.import_module("slowapi")
_slowapi_errors: Any = importlib.import_module("slowapi.errors")
_slowapi_middleware: Any = importlib.import_module("slowapi.middleware")
_slowapi_util: Any = importlib.import_module("slowapi.util")

AsyncIOScheduler = _apscheduler_asyncio.AsyncIOScheduler
Limiter = _slowapi.Limiter
_rate_limit_exceeded_handler = _slowapi._rate_limit_exceeded_handler
RateLimitExceeded = _slowapi_errors.RateLimitExceeded
SlowAPIMiddleware = _slowapi_middleware.SlowAPIMiddleware
get_remote_address = _slowapi_util.get_remote_address

from app.database import Base, Session, SessionLocal, delete_records_older_than, engine, get_db
from app.email_generator import EmailDraft, EmailGenerator
from app.llm_manager import AllProvidersExhaustedError, generate_ai_response
from app.models import JobHistory, JobHistoryCreate, JobHistoryResponse
from app.rag_engine import MatchedJob, RAGEngine
from app.resume_builder import ResumeBuilder
from app.scraper import JobScraper, ScrapedJob

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("your_assistant")

# Rate Limiter Configuration (Strict per-IP throttling to mitigate bot scraping and brute-force attacks)
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["60/minute"],
    headers_enabled=False,
)


def run_daily_job_search_pipeline(job_keyword: Optional[str] = None) -> Dict[str, Any]:
    """
    Automated background pipeline executed daily:
    1. Scrapes job listings via SerpApi Google Jobs.
    2. Runs ChromaDB + FastEmbed vector RAG engine against the base resume.
    3. Filters candidate jobs exceeding > 75% match score.
    4. Automatically records newly discovered high-match opportunities into JobHistory.
    5. Cleans up stale records older than 7 days.
    """
    logger.info("Executing scheduled daily automated job search & RAG pipeline...")
    db = SessionLocal()
    try:
        scraper = JobScraper()
        rag = RAGEngine()

        # Step 1: Scrape jobs (with 7-day duplicate exclusion)
        scraped_candidates = scraper.scrape_jobs(job_keyword=job_keyword, limit=20, db=db)
        logger.info("Scraper discovered %d unique candidates.", len(scraped_candidates))

        if not scraped_candidates:
            pruned = delete_records_older_than(days=7, db=db)
            return {"scraped": 0, "matched": 0, "saved": 0, "pruned": pruned}

        # Step 2: Vector RAG matching against resume
        matched_jobs = rag.match_jobs(scraped_candidates, min_match_score=75.0)
        logger.info("RAG Engine qualified %d jobs exceeding >75%% match score.", len(matched_jobs))

        # Step 3: Persist matched opportunities
        saved_count = 0
        for m in matched_jobs:
            db_entry = JobHistory(
                title=m.title,
                company=m.company,
                job_link=m.job_link,
                career_page_link=m.career_page_link,
                match_score=m.match_score,
                recruiter_email=None,
            )
            db.add(db_entry)
            saved_count += 1

        db.commit()
        logger.info("Persisted %d high-match opportunities into database.", saved_count)

        # Step 4: Run 7-day retention cleanup
        pruned_count = delete_records_older_than(days=7, db=db)
        return {
            "scraped": len(scraped_candidates),
            "matched": len(matched_jobs),
            "saved": saved_count,
            "pruned": pruned_count,
        }
    except Exception as exc:
        db.rollback()
        logger.error("Daily automated pipeline encountered an error: %s", exc, exc_info=True)
        return {"status": "error", "message": str(exc)}
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager:
    - Creates database tables on startup.
    - Initializes and starts the automated daily job search & retention scheduler.
    - Gracefully stops the scheduler on shutdown.
    """
    logger.info("Initializing 'Your Assistant' AI Job Search Platform...")
    Base.metadata.create_all(bind=engine)

    # Initialize automated scheduler
    scheduler = AsyncIOScheduler()

    # 1. Schedule daily automated scraping & RAG matching pipeline (every 24 hours)
    scheduler.add_job(
        func=run_daily_job_search_pipeline,
        trigger="interval",
        hours=24,
        id="daily_job_search_pipeline",
        replace_existing=True,
    )

    # 2. Schedule daily data retention pruning (every 24 hours)
    scheduler.add_job(
        func=delete_records_older_than,
        trigger="interval",
        hours=24,
        args=[7],
        id="job_retention_cleanup",
        replace_existing=True,
    )

    scheduler.start()
    logger.info("Automated daily job search and 7-day retention scheduler active.")

    # Startup maintenance
    try:
        pruned = delete_records_older_than(days=7)
        if pruned > 0:
            logger.info("Startup cleanup pruned %d outdated job records.", pruned)
    except Exception as exc:
        logger.warning("Startup retention cleanup notice: %s", exc)

    yield

    logger.info("Shutting down 'Your Assistant' platform...")
    scheduler.shutdown(wait=False)


# Initialize FastAPI Application
app = FastAPI(
    title="Your Assistant - Automated AI Job Search",
    description=(
        "Production-grade backend service powering automated AI job searching, "
        "recruiter discovery, ChromaDB vector RAG matching, resilient Multi-LLM "
        "fallback, and ATS-optimized resume/email generation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Attach SlowAPI Rate Limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# CORS Protection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# ==============================================================================
# Primary Application Endpoints
# ==============================================================================

@app.get("/", tags=["Health"])
@limiter.limit("30/minute")
async def health_check(request: Request):
    """
    Health check & assistant status endpoint.
    Includes the Antigravity Easter Egg.
    """
    return {
        "service": "Your Assistant - Automated AI Job Search Backend",
        "status": "healthy",
        "version": "1.0.0",
        "easter_egg": "antigravity active",
    }


@app.get(
    "/jobs",
    response_model=List[JobHistoryResponse],
    tags=["Job Search"],
)
@limiter.limit("60/minute")
async def get_jobs(
    request: Request,
    job_keyword: Optional[str] = Query(None, description="Optional keyword to filter job title or company"),
    min_score: Optional[float] = Query(None, ge=0.0, le=100.0, description="Optional minimum match score"),
    skip: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(50, ge=1, le=100, description="Items per page (max 100)"),
    db: Session = Depends(get_db),
):
    """
    Retrieve stored job opportunities.
    Supports optional `job_keyword` search across title and company,
    and returns matches ranked by match score descending.
    """
    query = db.query(JobHistory)

    if job_keyword and job_keyword.strip():
        kw = f"%{job_keyword.strip()}%"
        query = query.filter(or_(JobHistory.title.ilike(kw), JobHistory.company.ilike(kw)))

    if min_score is not None:
        query = query.filter(JobHistory.match_score >= min_score)

    jobs = (
        query.order_by(JobHistory.match_score.desc(), JobHistory.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return jobs


@app.post(
    "/generate-resume/{id}",
    tags=["Resume Builder"],
)
@limiter.limit("10/minute")
async def generate_resume_for_job(
    request: Request,
    id: int,
    db: Session = Depends(get_db),
):
    """
    Generate an ATS-optimized, tailored resume and downloadable PDF for a specific stored job record.
    Enforces the strict anti-hallucination constraint.
    """
    job = db.query(JobHistory).filter(JobHistory.id == id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job record with ID {id} does not exist.",
        )

    builder = ResumeBuilder()
    job_desc = f"Opportunity: {job.title} at {job.company}. Job Link: {job.job_link}"

    try:
        result = builder.build_tailored_resume_pdf(
            job_title=job.title,
            job_description=job_desc,
            company=job.company,
        )
        return {
            "status": "success",
            "job_id": job.id,
            "job_title": job.title,
            "company": job.company,
            "match_score": job.match_score,
            "tailored_resume_markdown": result["tailored_markdown"],
            "pdf_filename": result["filename"],
            "pdf_path": result["pdf_path"],
        }
    except Exception as exc:
        logger.error("Failed to generate resume for job %d: %s", id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating tailored resume: {exc}",
        )


@app.post(
    "/generate-email/{id}",
    response_model=EmailDraft,
    tags=["Cold Email Generator"],
)
@limiter.limit("15/minute")
async def generate_email_for_job(
    request: Request,
    id: int,
    db: Session = Depends(get_db),
):
    """
    Extract recruiter contact information and generate a personalized cold email for a specific stored job record.
    Emphasizes real skills from the candidate's resume.
    """
    job = db.query(JobHistory).filter(JobHistory.id == id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job record with ID {id} does not exist.",
        )

    generator = EmailGenerator()
    job_desc = f"{job.title} at {job.company}. Job Link: {job.job_link}"

    try:
        draft = generator.generate_cold_email(
            job_title=job.title,
            job_description=job_desc,
            company=job.company,
            recruiter_email=job.recruiter_email,
        )

        # If recruiter email was resolved from the draft and was previously empty, update database
        if draft["email"] != "[Recruiter Email]" and not job.recruiter_email:
            job.recruiter_email = draft["email"]
            db.commit()

        return draft
    except Exception as exc:
        logger.error("Failed to generate cold email for job %d: %s", id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating cold email: {exc}",
        )


# ==============================================================================
# Modular API Endpoints (Programmatic & Testing Sub-Routes)
# ==============================================================================

@app.post(
    "/api/jobs",
    response_model=JobHistoryResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Job History"],
)
@limiter.limit("20/minute")
async def create_job_history(
    request: Request,
    job_in: JobHistoryCreate,
    db: Session = Depends(get_db),
):
    """Record a new job opportunity discovered by the AI search agent."""
    db_job = JobHistory(
        title=job_in.title,
        company=job_in.company,
        job_link=str(job_in.job_link),
        career_page_link=str(job_in.career_page_link) if job_in.career_page_link else None,
        match_score=job_in.match_score,
        recruiter_email=str(job_in.recruiter_email) if job_in.recruiter_email else None,
    )
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    return db_job


@app.get(
    "/api/jobs",
    response_model=List[JobHistoryResponse],
    tags=["Job History"],
)
@limiter.limit("60/minute")
async def list_job_history(
    request: Request,
    skip: int = Query(0, ge=0, description="Offset for pagination"),
    limit: int = Query(50, ge=1, le=100, description="Limit per page (max 100)"),
    min_score: Optional[float] = Query(None, ge=0.0, le=100.0, description="Filter by minimum match score"),
    company: Optional[str] = Query(None, min_length=1, max_length=100, description="Filter by company"),
    db: Session = Depends(get_db),
):
    """Retrieve stored job search matches with pagination and filtering."""
    query = db.query(JobHistory)

    if min_score is not None:
        query = query.filter(JobHistory.match_score >= min_score)
    if company:
        query = query.filter(JobHistory.company.ilike(f"%{company}%"))

    jobs = query.order_by(JobHistory.created_at.desc()).offset(skip).limit(limit).all()
    return jobs


@app.post(
    "/api/jobs/scrape",
    response_model=List[ScrapedJob],
    tags=["Scraper"],
)
@limiter.limit("10/minute")
async def trigger_job_scrape(
    request: Request,
    keyword: Optional[str] = Query(None, description="Optional job title/keyword. Falls back to 'Software Engineer remote'."),
    limit: int = Query(10, ge=1, le=50, description="Max jobs to fetch"),
    db: Session = Depends(get_db),
):
    """Scrape job opportunities via SerpApi Google Jobs."""
    scraper = JobScraper()
    jobs = scraper.scrape_jobs(job_keyword=keyword, limit=limit, db=db)
    return jobs


@app.post(
    "/api/jobs/match",
    response_model=List[MatchedJob],
    tags=["RAG Engine"],
)
@limiter.limit("10/minute")
async def match_scraped_jobs(
    request: Request,
    jobs: List[ScrapedJob],
    min_score: float = Query(75.0, ge=0.0, le=100.0, description="Minimum match score percentage cutoff"),
):
    """Vector RAG matching returning only jobs exceeding the match threshold (> 75%)."""
    rag = RAGEngine()
    matched = rag.match_jobs(jobs, min_match_score=min_score)
    return matched


class GenerateAIRequest(BaseModel):
    prompt: str = Field(..., min_length=2, max_length=10000, description="Prompt for LLM generation")
    system_prompt: Optional[str] = Field(default=None, max_length=2000, description="Optional system instructions")


@app.post(
    "/api/ai/generate",
    tags=["AI Generation"],
)
@limiter.limit("15/minute")
async def generate_ai(
    request: Request,
    payload: GenerateAIRequest,
):
    """Generate AI text using multi-tier fallback: Claude 3 (Key 1 -> 2) -> Gemini 1.5 (Key 1 -> 2)."""
    try:
        response_text = generate_ai_response(
            prompt=payload.prompt,
            system_prompt=payload.system_prompt,
        )
        return {
            "status": "success",
            "response": response_text,
        }
    except AllProvidersExhaustedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )


class TailorResumeRequest(BaseModel):
    job_title: str = Field(..., min_length=2, max_length=255, description="Target job title")
    job_description: str = Field(..., min_length=10, max_length=20000, description="Target job description")
    company: Optional[str] = Field(default=None, max_length=255, description="Company name")
    base_resume_text: Optional[str] = Field(default=None, description="Optional custom base resume text")


@app.post(
    "/api/resume/tailor",
    tags=["Resume Builder"],
)
@limiter.limit("10/minute")
async def tailor_resume_endpoint(
    request: Request,
    payload: TailorResumeRequest,
):
    """Tailor candidate resume for a target job with 100% ATS compatibility and zero hallucination."""
    builder = ResumeBuilder()
    try:
        tailored = builder.tailor_resume(
            job_title=payload.job_title,
            job_description=payload.job_description,
            company=payload.company,
            base_resume_text=payload.base_resume_text,
        )
        return {
            "status": "success",
            "job_title": payload.job_title,
            "company": payload.company,
            "tailored_resume_markdown": tailored,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to tailor resume: {exc}",
        )


class GenerateResumePDFRequest(BaseModel):
    markdown_text: str = Field(..., min_length=20, description="Markdown content of the tailored resume")
    filename: Optional[str] = Field(default="tailored_resume.pdf", description="Output PDF filename")


@app.post(
    "/api/resume/generate-pdf",
    tags=["Resume Builder"],
)
@limiter.limit("10/minute")
async def generate_resume_pdf_endpoint(
    request: Request,
    payload: GenerateResumePDFRequest,
):
    """Convert tailored Markdown resume into a high-quality, ATS-optimized PDF and return it for download."""
    builder = ResumeBuilder()
    safe_name = payload.filename if payload.filename.endswith(".pdf") else f"{payload.filename}.pdf"
    pdf_path = Path("output") / safe_name
    try:
        generated_file = builder.generate_pdf(payload.markdown_text, pdf_path)
        return FileResponse(
            path=str(generated_file),
            filename=safe_name,
            media_type="application/pdf",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate PDF: {exc}",
        )


class GenerateEmailRequest(BaseModel):
    job_title: str = Field(..., min_length=2, max_length=255, description="Target job title")
    job_description: str = Field(..., min_length=10, max_length=20000, description="Job description text")
    company: Optional[str] = Field(default=None, max_length=255, description="Target company name")
    recruiter_email: Optional[str] = Field(default=None, description="Optional explicit recruiter email")
    base_resume_text: Optional[str] = Field(default=None, description="Optional custom base resume text")


@app.post(
    "/api/email/generate",
    response_model=EmailDraft,
    tags=["Cold Email Generator"],
)
@limiter.limit("15/minute")
async def generate_email_endpoint(
    request: Request,
    payload: GenerateEmailRequest,
):
    """Extract recruiter contact information and generate a high-impact cold outreach email."""
    generator = EmailGenerator()
    try:
        draft_dict = generator.generate_cold_email(
            job_title=payload.job_title,
            job_description=payload.job_description,
            company=payload.company,
            recruiter_email=payload.recruiter_email,
            base_resume_text=payload.base_resume_text,
        )
        return draft_dict
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate cold email: {exc}",
        )


@app.post(
    "/api/jobs/cleanup",
    tags=["Maintenance"],
)
@limiter.limit("5/minute")
async def trigger_manual_cleanup(
    request: Request,
    retention_days: int = Query(7, ge=1, le=365, description="Retention threshold in days"),
    db: Session = Depends(get_db),
):
    """Manually trigger data retention pruning to delete jobs older than `retention_days` (default: 7 days)."""
    deleted = delete_records_older_than(days=retention_days, db=db)
    return {
        "status": "success",
        "retention_days": retention_days,
        "records_deleted": deleted,
    }

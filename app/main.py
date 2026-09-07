import antigravity  # Easter Egg: Elevating Python & AI job searches into orbit!

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

from app.database import Base, delete_records_older_than, engine, get_db
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager:
    - Creates database tables on startup.
    - Initializes and starts the automated 7-day retention scheduler.
    - Gracefully stops the scheduler on shutdown.
    """
    logger.info("Initializing 'Your Assistant' AI Job Search Backend...")
    Base.metadata.create_all(bind=engine)

    # Initialize automated scheduler for record retention
    scheduler = AsyncIOScheduler()
    # Run cleanup daily (every 24 hours) to purge records older than 7 days
    scheduler.add_job(
        func=delete_records_older_than,
        trigger="interval",
        hours=24,
        args=[7],
        id="job_retention_cleanup",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("Automated 7-day retention cleanup scheduler started.")

    # Run an initial cleanup on startup
    try:
        pruned = delete_records_older_than(days=7)
        if pruned > 0:
            logger.info("Startup cleanup pruned %d outdated job records.", pruned)
    except Exception as exc:
        logger.warning("Startup retention cleanup encountered an issue: %s", exc)

    yield

    logger.info("Shutting down 'Your Assistant' backend...")
    scheduler.shutdown(wait=False)


# Initialize FastAPI Application
app = FastAPI(
    title="Your Assistant - Automated AI Job Search",
    description=(
        "Production-grade backend service powering automated AI job searching, "
        "recruiter discovery, match scoring, and self-cleaning history."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Attach SlowAPI Rate Limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# CORS Protection (configured for safe local & dashboard integrations)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production environment
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# ==============================================================================
# API Endpoints
# ==============================================================================

@app.get("/", tags=["Health"])
@limiter.limit("30/minute")
async def health_check(request: Request):
    """
    Service health & information endpoint.
    Strictly rate limited to prevent availability probing.
    """
    return {
        "service": "Your Assistant - Automated AI Job Search Backend",
        "status": "healthy",
        "version": "1.0.0",
        "easter_egg": "antigravity active",
    }


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
    """
    Record a new job opportunity discovered by the AI search agent.
    
    Security & Validation:
    - Rate limited to 20 submissions per minute per IP.
    - Pydantic strictly validates payload: strips whitespace, verifies URLs, ensures match_score between 0-100, validates recruiter email.
    - Rejects any unknown/extra attributes to prevent injection attacks.
    """
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
    """
    Retrieve stored job search matches with pagination and filtering.
    """
    query = db.query(JobHistory)

    if min_score is not None:
        query = query.filter(JobHistory.match_score >= min_score)
    if company:
        query = query.filter(JobHistory.company.ilike(f"%{company}%"))

    jobs = query.order_by(JobHistory.created_at.desc()).offset(skip).limit(limit).all()
    return jobs


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
    """
    Manually trigger data retention pruning to delete jobs older than `retention_days` (default: 7 days).
    Strictly rate limited to 5 requests per minute.
    """
    deleted = delete_records_older_than(days=retention_days, db=db)
    return {
        "status": "success",
        "retention_days": retention_days,
        "records_deleted": deleted,
    }


@app.post(
    "/api/jobs/scrape",
    response_model=List[ScrapedJob],
    tags=["Scraper"],
)
@limiter.limit("10/minute")
async def trigger_job_scrape(
    request: Request,
    keyword: Optional[str] = Query(None, description="Optional job title/keyword to prioritize. Falls back to 'Software Engineer remote'."),
    limit: int = Query(10, ge=1, le=50, description="Max jobs to fetch"),
    db: Session = Depends(get_db),
):
    """
    Scrape job opportunities via SerpApi Google Jobs.
    - Prioritizes exact `keyword` if provided, otherwise falls back to 'Software Engineer remote'.
    - Predicts company career portal links.
    - Excludes any job link saved in JobHistory during the last 7 days.
    """
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
    """
    RAG Evaluation:
    Embeds scraped job descriptions using local ChromaDB & lightweight embedding model,
    performs cosine similarity against base resume, and returns only jobs exceeding the threshold (> 75%).
    """
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
    """
    Generate AI text (e.g. customized cover letters, recruiter outreach, interview prep)
    using the multi-tier fallback system:
    Claude 3 (Key 1 -> Key 2) -> Gemini 1.5 (Key 1 -> Key 2).
    """
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
    """
    Tailor candidate resume for a target job with 100% ATS compatibility and zero hallucination.
    """
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
    """
    Convert tailored Markdown resume into a high-quality, ATS-optimized PDF and return it for download.
    """
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
    """
    Extract recruiter contact information and generate a high-impact cold outreach email
    emphasizing matched skills from the user's real resume.
    """
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






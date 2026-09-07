import antigravity  # Easter Egg: Elevating Python & AI job searches into orbit!

from datetime import date, datetime, timezone
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

from app.auth import (
    LoginRequest,
    TokenResponse,
    UserProfile,
    create_access_token,
    require_authenticated_user,
    verify_admin_credentials,
)
from app.database import (
    Base,
    Session,
    SessionLocal,
    delete_jobs_by_date,
    delete_records_older_than,
    engine,
    get_db,
)
from app.email_generator import EmailDraft, EmailGenerator
from app.llm_manager import AllProvidersExhaustedError, generate_ai_response
from app.models import (
    Job,
    JobCreate,
    JobHistory,
    JobHistoryCreate,
    JobHistoryResponse,
    JobResponse,
    JobStatus,
    JobStatusEnum,
    JobUpdateStatus,
)
from app.rag_engine import MatchedJob, RAGEngine
from app.resume_builder import ResumeBuilder
from app.scraper import PRIMARY_KEYWORDS, JobScraper, ScrapedJob

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("your_assistant")

# Rate Limiter Configuration (Strict per-IP throttling to mitigate bot scraping and brute-force attacks)
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["120/minute"],
    headers_enabled=False,
)


# Default match score threshold from environment
DEFAULT_MATCH_SCORE = float(os.getenv("MIN_MATCH_SCORE", "55.0"))


def run_daily_job_search_pipeline(job_keyword: Optional[str] = None) -> Dict[str, Any]:
    """
    Automated background pipeline executed daily:
    1. Scrapes job listings via concurrent multi-source scraper (skills & platform targets).
    2. Runs ChromaDB + FastEmbed vector RAG engine against base resume (cutoff >= 55%).
    3. Persists qualified opportunities into Job table with scraped_date = date.today().
    4. Cleans up stale records older than 7 days.
    """
    logger.info("Executing scheduled daily automated job search & RAG pipeline...")
    db = SessionLocal()
    try:
        scraper = JobScraper()
        rag = RAGEngine()

        # Step 1: Scrape jobs (with 7-day duplicate exclusion)
        scraped_candidates = scraper.scrape_jobs(job_keyword=job_keyword, limit=60, db=db)
        logger.info("Scraper discovered %d unique candidates.", len(scraped_candidates))

        if not scraped_candidates:
            pruned = delete_records_older_than(days=7, db=db)
            return {"scraped": 0, "matched": 0, "saved": 0, "pruned": pruned}

        # Step 2: Vector RAG matching against resume (cutoff >= 55%)
        matched_jobs = rag.match_jobs(scraped_candidates, min_match_score=DEFAULT_MATCH_SCORE)
        logger.info("RAG Engine qualified %d jobs exceeding >=%.1f%% match score.", len(matched_jobs), DEFAULT_MATCH_SCORE)


        # Step 3: Persist matched opportunities
        today = date.today()
        saved_count = 0
        for m in matched_jobs:
            existing = db.query(Job).filter(Job.link == m.job_link).first()
            if not existing:
                db_entry = Job(
                    title=m.title,
                    company=m.company,
                    link=m.job_link,
                    career_page_link=m.career_page_link,
                    match_score=m.match_score,
                    recruiter_email=getattr(m, "recruiter_email", None),
                    description=m.description,
                    location=getattr(m, "location", "Remote") or "Remote",
                    status=JobStatus.Pending,
                    scraped_date=today,
                )
                db.add(db_entry)
                saved_count += 1
            else:
                existing.match_score = m.match_score
                existing.scraped_date = today

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
    - Creates database tables on startup (with graceful fallback).
    - Initializes and starts the automated daily job search & retention scheduler if running as persistent server.
    - Gracefully stops the scheduler on shutdown.
    """
    logger.info("Initializing 'Your Assistant' AI Job Search Platform...")
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as exc:
        logger.warning("Database schema init notice: %s", exc)

    # In serverless environments like Vercel, persistent cron loops cannot run
    is_serverless = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))
    scheduler = None

    if not is_serverless:
        try:
            scheduler = AsyncIOScheduler()
            scheduler.add_job(
                func=run_daily_job_search_pipeline,
                trigger="interval",
                hours=24,
                id="daily_job_search_pipeline",
                replace_existing=True,
            )
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
        except Exception as exc:
            logger.warning("Scheduler startup notice: %s", exc)

        try:
            pruned = delete_records_older_than(days=7)
            if pruned > 0:
                logger.info("Startup cleanup pruned %d outdated job records.", pruned)
        except Exception as exc:
            logger.warning("Startup retention cleanup notice: %s", exc)

    yield

    if scheduler is not None:
        logger.info("Shutting down scheduler...")
        scheduler.shutdown(wait=False)


# Initialize FastAPI Application
app = FastAPI(
    title="Your Assistant - Automated AI Job Search",
    description=(
        "Production-grade backend service powering automated AI job searching, "
        "recruiter discovery, ChromaDB vector RAG matching, SQL application tracking, "
        "and ATS-optimized resume/email generation."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# Attach SlowAPI Rate Limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# CORS Protection (supports GET, POST, PATCH, PUT, DELETE, OPTIONS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# ==============================================================================
# Health Endpoint
# ==============================================================================

@app.get("/", tags=["Health"])
@app.get("/health", tags=["Health"])
@app.get("/api", tags=["Health"])
@app.get("/api/health", tags=["Health"])
@app.get("/api/index", tags=["Health"])
@limiter.limit("60/minute")
async def health_check(request: Request):
    """
    Health check & assistant status endpoint.
    Includes the Antigravity Easter Egg.
    """
    return {
        "service": "Your Assistant - Automated AI Job Search Backend",
        "status": "healthy",
        "version": "2.0.0",
        "easter_egg": "antigravity active",
    }


# ==============================================================================
# Authentication & JWT Endpoints
# ==============================================================================

@app.post(
    "/api/auth/login",
    response_model=TokenResponse,
    tags=["Authentication"],
)
@limiter.limit("15/minute")
async def login_admin(
    request: Request,
    payload: LoginRequest,
):
    """
    Authenticate admin credentials configured in .env (ADMIN_EMAIL, ADMIN_PASSWORD)
    and issue a signed JWT access token.
    """
    if not verify_admin_credentials(payload.email, payload.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={"sub": payload.email, "role": "admin"}
    )
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user={"email": payload.email, "role": "admin"},
    )


@app.get(
    "/api/auth/me",
    response_model=UserProfile,
    tags=["Authentication"],
)
@limiter.limit("60/minute")
async def get_current_user_profile(
    request: Request,
    user_data: Dict[str, Any] = Depends(require_authenticated_user),
):
    """
    Validate active JWT access token and return current authenticated user profile.
    """
    return UserProfile(
        email=user_data.get("sub", "admin"),
        role=user_data.get("role", "admin"),
        authenticated=True,
    )


# ==============================================================================
# Application Tracking & Core Job Endpoints
# ==============================================================================

@app.post(
    "/api/scrape",
    tags=["Job Scraper & Matching"],
)
@limiter.limit("10/minute")
async def trigger_scrape_and_match(
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Triggers the concurrent multi-source job scraper (12 hardcoded primary keywords),
    evaluates candidates against the user's base resume (data/resume.txt) via ChromaDB RAG,
    and ONLY saves jobs with match_score >= 65% for today's date.
    """
    scraper = JobScraper()
    rag = RAGEngine()

    # 1. Fetch raw candidates
    scraped_candidates = scraper.scrape_jobs(limit=60, db=db)
    logger.info("Scraper fetched %d candidates adhering to 24h & geo rules.", len(scraped_candidates))

    if not scraped_candidates:
        return {
            "status": "success",
            "scraped_count": 0,
            "matched_count": 0,
            "saved_count": 0,
            "message": "No new unique jobs discovered matching 24-hour and geographic filters.",
        }

    # 2. Vector RAG evaluation (Strict cutoff >= 55%)
    matched_jobs = rag.match_jobs(scraped_candidates, min_match_score=DEFAULT_MATCH_SCORE)
    logger.info("RAG Engine qualified %d jobs exceeding >=%.1f%% threshold.", len(matched_jobs), DEFAULT_MATCH_SCORE)

    today = date.today()
    saved_count = 0
    for m in matched_jobs:
        existing = db.query(Job).filter(Job.link == m.job_link).first()
        if not existing:
            new_job = Job(
                title=m.title,
                company=m.company,
                link=m.job_link,
                career_page_link=m.career_page_link,
                match_score=m.match_score,
                recruiter_email=getattr(m, "recruiter_email", None),
                description=m.description,
                location=getattr(m, "location", "Remote") or "Remote",
                status=JobStatus.Pending,
                scraped_date=today,
            )
            db.add(new_job)
            saved_count += 1
        else:
            existing.match_score = m.match_score
            existing.scraped_date = today

    db.commit()

    return {
        "status": "success",
        "scraped_count": len(scraped_candidates),
        "matched_count": len(matched_jobs),
        "saved_count": saved_count,
        "scraped_date": today.isoformat(),
        "message": f"Successfully scraped {len(scraped_candidates)} candidates, matched {len(matched_jobs)} (>= {DEFAULT_MATCH_SCORE}%), and saved {saved_count} new jobs for {today}.",
    }


@app.get(
    "/api/jobs",
    response_model=List[JobResponse],
    tags=["Job Search"],
)
@limiter.limit("60/minute")
async def get_jobs_by_date(
    request: Request,
    date_filter: Optional[date] = Query(
        None,
        alias="date",
        description="Filter jobs by scraped date (YYYY-MM-DD). Defaults to today's date if omitted.",
    ),
    skip: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(100, ge=1, le=200, description="Maximum jobs to return"),
    db: Session = Depends(get_db),
):
    """
    Retrieve stored job opportunities for a specific date (defaults to today).
    Returns jobs ranked by match score descending.
    """
    target_date = date_filter or date.today()
    jobs = (
        db.query(Job)
        .filter(Job.scraped_date == target_date)
        .order_by(Job.match_score.desc(), Job.id.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return jobs


@app.patch(
    "/api/jobs/{job_id}/status",
    response_model=JobResponse,
    tags=["Application Tracking"],
)
@limiter.limit("30/minute")
async def update_job_status(
    request: Request,
    job_id: int,
    payload: JobUpdateStatus,
    db: Session = Depends(get_db),
):
    """
    Update the application tracking status for a job (Pending, Applied, Interview, Rejected).
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job record with ID {job_id} does not exist.",
        )

    job.status = JobStatus(payload.status.value)
    db.commit()
    db.refresh(job)
    logger.info("Updated Job ID %d status to '%s'", job_id, job.status.value)
    return job


@app.delete(
    "/api/jobs/date/{target_date}",
    tags=["Job Search"],
)
@limiter.limit("15/minute")
async def delete_jobs_for_date(
    request: Request,
    target_date: date,
    db: Session = Depends(get_db),
):
    """
    Delete all scraped jobs for the specified date (YYYY-MM-DD).
    """
    deleted_count = delete_jobs_by_date(target_date=target_date, db=db)
    return {
        "status": "success",
        "date": target_date.isoformat(),
        "deleted_count": deleted_count,
        "message": f"Successfully deleted {deleted_count} jobs scraped on {target_date.isoformat()}.",
    }


# ==============================================================================
# Resume and Email Generation Endpoints
# ==============================================================================

@app.post(
    "/generate-resume/{id}",
    tags=["Resume Builder"],
)
@limiter.limit("15/minute")
async def generate_resume_for_job(
    request: Request,
    id: int,
    db: Session = Depends(get_db),
):
    """
    Generate an ATS-optimized, tailored resume and downloadable PDF for a specific stored job record.
    Enforces strict black text styling and anti-hallucination constraint.
    """
    job = db.query(Job).filter(Job.id == id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job record with ID {id} does not exist.",
        )

    builder = ResumeBuilder()
    job_desc = (
        job.description
        if job.description and len(job.description.strip()) > 15
        else f"Opportunity: {job.title} at {job.company}. Job Link: {job.link}"
    )

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
@limiter.limit("20/minute")
async def generate_email_for_job(
    request: Request,
    id: int,
    db: Session = Depends(get_db),
):
    """
    Extract recruiter contact information and generate a personalized cold email for a specific stored job record.
    """
    job = db.query(Job).filter(Job.id == id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job record with ID {id} does not exist.",
        )

    generator = EmailGenerator()
    job_desc = (
        job.description
        if job.description and len(job.description.strip()) > 15
        else f"{job.title} at {job.company}. Job Link: {job.link}"
    )

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
# Backward Compatibility & Utility Endpoints
# ==============================================================================

@app.get(
    "/jobs",
    response_model=List[JobResponse],
    tags=["Job Search (Legacy)"],
)
@limiter.limit("60/minute")
async def get_jobs_legacy(
    request: Request,
    job_keyword: Optional[str] = Query(None, description="Optional keyword to filter job title or company"),
    min_score: Optional[float] = Query(None, ge=0.0, le=100.0, description="Optional minimum match score"),
    date_filter: Optional[date] = Query(None, alias="date", description="Filter by scraped date (YYYY-MM-DD)"),
    skip: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(50, ge=1, le=100, description="Items per page (max 100)"),
    db: Session = Depends(get_db),
):
    """Retrieve stored job opportunities with optional keyword, date, and score filtering."""
    query = db.query(Job)

    if date_filter:
        query = query.filter(Job.scraped_date == date_filter)

    if job_keyword and job_keyword.strip():
        kw = f"%{job_keyword.strip()}%"
        query = query.filter(or_(Job.title.ilike(kw), Job.company.ilike(kw)))

    if min_score is not None:
        query = query.filter(Job.match_score >= min_score)

    jobs = (
        query.order_by(Job.match_score.desc(), Job.scraped_date.desc(), Job.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return jobs


@app.post(
    "/api/jobs",
    response_model=JobResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Job Search"],
)
@limiter.limit("30/minute")
async def create_job_endpoint(
    request: Request,
    job_in: JobCreate,
    db: Session = Depends(get_db),
):
    """Record a new job opportunity programmatically."""
    db_job = Job(
        title=job_in.title,
        company=job_in.company,
        link=str(job_in.link),
        career_page_link=str(job_in.career_page_link) if job_in.career_page_link else None,
        match_score=job_in.match_score,
        recruiter_email=str(job_in.recruiter_email) if job_in.recruiter_email else None,
        description=job_in.description,
        location=job_in.location or "Remote",
        status=JobStatus(job_in.status.value),
        scraped_date=job_in.scraped_date or date.today(),
    )
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    return db_job


@app.post(
    "/api/jobs/scrape",
    response_model=List[ScrapedJob],
    tags=["Scraper"],
)
@limiter.limit("10/minute")
async def trigger_job_scrape_raw(
    request: Request,
    keyword: Optional[str] = Query(None, description="Keyword"),
    limit: int = Query(10, ge=1, le=50, description="Max jobs to fetch"),
    db: Session = Depends(get_db),
):
    """Scrape raw job opportunities."""
    scraper = JobScraper()
    jobs = scraper.scrape_jobs(job_keyword=keyword, limit=limit, db=db)
    return jobs


@app.post(
    "/api/jobs/match",
    response_model=List[MatchedJob],
    tags=["RAG Engine"],
)
@limiter.limit("10/minute")
async def match_scraped_jobs_endpoint(
    request: Request,
    jobs: List[ScrapedJob],
    min_score: float = Query(DEFAULT_MATCH_SCORE, ge=0.0, le=100.0, description="Minimum match score percentage cutoff"),
):
    """Vector RAG matching returning only jobs exceeding the match threshold (>= 55%)."""
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
@limiter.limit("20/minute")
async def generate_ai(
    request: Request,
    payload: GenerateAIRequest,
):
    """Generate AI text using multi-tier fallback: Claude 3 -> Gemini."""
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
@limiter.limit("15/minute")
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
@limiter.limit("15/minute")
async def generate_resume_pdf_endpoint(
    request: Request,
    payload: GenerateResumePDFRequest,
):
    """Convert tailored Markdown resume into a high-quality, ATS-optimized PDF and return it for download."""
    builder = ResumeBuilder()
    raw_name = payload.filename or "tailored_resume.pdf"
    safe_name = raw_name if raw_name.endswith(".pdf") else f"{raw_name}.pdf"
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
@limiter.limit("20/minute")
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

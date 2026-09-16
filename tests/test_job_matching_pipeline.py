import os
import sys
import logging
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database import Base, SessionLocal, engine
from app.models import Job, JobStatus
from app.rag_engine import (
    JobMatchAnalysis,
    MatchedJob,
    RAGEngine,
    evaluate_job_match_with_llm,
    get_min_match_score,
)
from app.scraper import DEFAULT_TARGET_ROLES, JobScraper, ScrapedJob, get_target_roles
from app.main import run_daily_job_search_pipeline, save_or_update_matched_job


def setup_function():
    """Ensure clean database before each test run."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


# ==============================================================================
# 1. Environment & Config Validation Tests
# ==============================================================================

def test_get_target_roles_parsing():
    """Verify target roles cleanly parse comma-separated lists, quotes, and whitespace."""
    with patch.dict(os.environ, {"TARGET_ROLES": '"Python Developer" , Senior AI Engineer ,  Backend Lead '}):
        roles = get_target_roles()
        assert roles == ["Python Developer", "Senior AI Engineer", "Backend Lead"]

    with patch.dict(os.environ, {"TARGET_ROLES": ""}):
        roles = get_target_roles()
        assert roles == DEFAULT_TARGET_ROLES

    with patch.dict(os.environ, {}, clear=True):
        if "TARGET_ROLES" in os.environ:
            del os.environ["TARGET_ROLES"]
        roles = get_target_roles()
        assert roles == DEFAULT_TARGET_ROLES


def test_get_min_match_score_parsing():
    """Verify min match score parsing handles floats, ints, invalid strings, and boundary values."""
    with patch.dict(os.environ, {"MIN_MATCH_SCORE": "75.5"}):
        assert get_min_match_score() == 75.5

    with patch.dict(os.environ, {"MIN_MATCH_SCORE": "80"}):
        assert get_min_match_score() == 80.0

    with patch.dict(os.environ, {"MIN_MATCH_SCORE": "invalid_score"}):
        assert get_min_match_score() == 55.0

    with patch.dict(os.environ, {"MIN_MATCH_SCORE": "150"}):
        assert get_min_match_score() == 100.0

    with patch.dict(os.environ, {"MIN_MATCH_SCORE": "-10"}):
        assert get_min_match_score() == 0.0


# ==============================================================================
# 2. ChromaDB Context Retrieval & Structured LLM Matching Tests
# ==============================================================================

def test_evaluate_job_match_with_llm_json():
    """Verify LLM match evaluation produces structured JobMatchAnalysis from valid JSON."""
    llm_payload = """
    ```json
    {
      "match_score": 85,
      "matching_skills": ["Python", "FastAPI", "PostgreSQL"],
      "missing_skills": ["Kubernetes"],
      "reasoning": "Strong match with Python backend stack."
    }
    ```
    """
    with patch("app.llm_manager.generate_ai_response", return_value=llm_payload):
        result = evaluate_job_match_with_llm(
            job_title="Senior Python Backend Engineer",
            job_description="Seeking Python and FastAPI developer with PostgreSQL experience and Kubernetes knowledge.",
            candidate_resume_context="Expert in Python, FastAPI, Docker, and PostgreSQL databases.",
            semantic_vector_score=82.0,
        )
        assert isinstance(result, JobMatchAnalysis)
        assert result.match_score == 85
        assert "Python" in result.matching_skills
        assert "Kubernetes" in result.missing_skills
        assert "Strong match" in result.reasoning


def test_evaluate_job_match_with_llm_fallback_resilience():
    """Verify corrupted LLM responses or API failures gracefully fallback without raising."""
    with patch("app.llm_manager.generate_ai_response", side_effect=RuntimeError("Provider API timeout")):
        result = evaluate_job_match_with_llm(
            job_title="Senior Python Backend Engineer",
            job_description="Requires Python, FastAPI, and PostgreSQL experience.",
            candidate_resume_context="Experienced Python engineer with FastAPI expertise.",
            semantic_vector_score=78.0,
        )
        assert isinstance(result, JobMatchAnalysis)
        assert result.match_score == 78
        assert len(result.matching_skills) > 0
        assert "Python" in result.matching_skills
        assert "Automated evaluation" in result.reasoning


# ==============================================================================
# 3. Database Saving & Threshold Isolation Tests
# ==============================================================================

def test_save_or_update_matched_job_persistence():
    """Verify matched job persistence, unique constraints, and db.commit."""
    db = SessionLocal()
    try:
        matched = MatchedJob(
            title="Full Stack Engineer",
            company="Acme Corp",
            description="Building Next.js and Python apps",
            job_link="https://acme.example/jobs/101",
            career_page_link="https://acme.example/careers",
            location="Remote",
            match_score=88.0,
            matching_skills=["Python", "Next.js"],
            missing_skills=["GraphQL"],
            reasoning="High skill alignment.",
        )

        # 1. Insert new job
        record = save_or_update_matched_job(db=db, job=matched)
        assert record is not None
        assert record.id is not None
        assert record.title == "Full Stack Engineer"
        assert record.company == "Acme Corp"
        assert record.match_score == 88.0
        assert record.status == JobStatus.Pending

        # 2. Update existing job link (Upsert)
        matched.match_score = 92.0
        updated = save_or_update_matched_job(db=db, job=matched)
        assert updated is not None
        assert updated.id == record.id
        assert updated.match_score == 92.0

        # Verify in DB
        db_job = db.query(Job).filter(Job.link == "https://acme.example/jobs/101").first()
        assert db_job is not None
        assert db_job.match_score == 92.0
    finally:
        db.close()


def test_save_or_update_matched_job_rollback_on_error():
    """Verify session rollback occurs when an error is raised during save."""
    db = SessionLocal()
    try:
        matched = MatchedJob(
            title="Corrupted Job",
            company="Fail Inc",
            description="Test",
            job_link="https://fail.example/job",
            career_page_link="https://fail.example/careers",
            match_score=80.0,
        )

        with patch.object(db, "commit", side_effect=Exception("Database lock error")):
            result = save_or_update_matched_job(db=db, job=matched)
            assert result is None
    finally:
        db.close()


# ==============================================================================
# 4. Comprehensive Logging Verification
# ==============================================================================

def test_logging_formats(caplog):
    """
    Verify all required logging formats:
    - [INFO] Fetched X jobs for role: Y
    - [DEBUG] Job: "Role Title" | Calculated Score: 80% | Required Score: 75%
    - [SUCCESS] Saved job ID X to Database.
    - [ERROR] Database save failed for job ID X: <Error Message>
    """
    caplog.set_level(logging.DEBUG)

    # 1. Test [INFO] Fetched X jobs for role: Y in scraper
    scraper = JobScraper()
    with patch.object(scraper, "fetch_serpapi_jobs", return_value=[]), \
         patch.object(scraper, "fetch_weworkremotely_feed", return_value=[]), \
         patch.object(scraper, "fetch_remoteco_feed", return_value=[]):
        with patch.dict(os.environ, {"TARGET_ROLES": "AI Engineer"}):
            scraper.scrape_jobs(limit=10)
            assert any("[INFO] Fetched" in record.message and "for role: AI Engineer" in record.message for record in caplog.records)

    # 2. Test [DEBUG] Job: "Role Title" | Calculated Score: 80% | Required Score: 75% in rag_engine
    rag = RAGEngine()
    dummy_candidate = ScrapedJob(
        title="Software Engineer",
        company="TechCorp",
        description="Python backend microservices",
        job_link="https://techcorp.example/jobs/201",
        career_page_link="https://techcorp.example/careers",
    )
    with patch.object(rag, "get_resume_text", return_value="Python, Docker, SQL"), \
         patch.object(rag, "query_resume", return_value=["Python, Docker, SQL"]), \
         patch("app.rag_engine.evaluate_job_match_with_llm", return_value=JobMatchAnalysis(
             match_score=80,
             matching_skills=["Python"],
             missing_skills=[],
             reasoning="Good match"
         )):
        rag.match_jobs([dummy_candidate], min_match_score=75.0)
        assert any('[DEBUG] Job: "Software Engineer" | Calculated Score: 80% | Required Score: 75%' in record.message for record in caplog.records)

    # 3. Test [SUCCESS] Saved job ID X to Database.
    db = SessionLocal()
    try:
        matched_job = MatchedJob(
            title="Software Engineer",
            company="TechCorp",
            description="Python backend",
            job_link="https://techcorp.example/jobs/201",
            career_page_link="https://techcorp.example/careers",
            match_score=80.0,
        )
        saved = save_or_update_matched_job(db=db, job=matched_job)
        assert saved is not None
        assert any(f"[SUCCESS] Saved job ID {saved.id} to Database." in record.message for record in caplog.records)

        # 4. Test [ERROR] Database save failed for job ID X: <Error Message>
        with patch.object(db, "commit", side_effect=Exception("Disk full")):
            save_or_update_matched_job(db=db, job=matched_job)
            assert any("[ERROR] Database save failed for job ID" in record.message and "Disk full" in record.message for record in caplog.records)
    finally:
        db.close()


# ==============================================================================
# 5. Full Pipeline End-to-End Test
# ==============================================================================

def test_daily_pipeline_saves_only_above_threshold():
    """Verify that run_daily_job_search_pipeline saves only jobs exceeding MIN_MATCH_SCORE."""
    dummy_candidates = [
        ScrapedJob(
            title="High Match Role",
            company="Good Co",
            description="Python, FastAPI, RAG systems",
            job_link="https://good.example/1",
            career_page_link="https://good.example/careers",
        ),
        ScrapedJob(
            title="Low Match Role",
            company="Bad Co",
            description="Cobol legacy maintenance",
            job_link="https://bad.example/2",
            career_page_link="https://bad.example/careers",
        ),
    ]

    with patch.dict(os.environ, {"TARGET_ROLES": "Python Developer", "MIN_MATCH_SCORE": "75.0"}), \
         patch("app.scraper.JobScraper.scrape_jobs", return_value=dummy_candidates), \
         patch("app.rag_engine.RAGEngine.get_resume_text", return_value="Python, FastAPI, RAG, SQL"):

        # Mock evaluate_job_match_with_llm to return 85% for High Match and 40% for Low Match
        def mock_eval(job_title, job_description, candidate_resume_context, semantic_vector_score=None):
            if "High Match" in job_title:
                return JobMatchAnalysis(match_score=85, matching_skills=["Python"], missing_skills=[], reasoning="Match")
            else:
                return JobMatchAnalysis(match_score=40, matching_skills=[], missing_skills=["Cobol"], reasoning="Low")

        with patch("app.rag_engine.evaluate_job_match_with_llm", side_effect=mock_eval):
            results = run_daily_job_search_pipeline()
            assert results["saved"] == 1
            assert results["matched"] == 1

            db = SessionLocal()
            try:
                saved_jobs = db.query(Job).all()
                assert len(saved_jobs) == 1
                assert saved_jobs[0].title == "High Match Role"
                assert saved_jobs[0].match_score == 85.0
            finally:
                db.close()

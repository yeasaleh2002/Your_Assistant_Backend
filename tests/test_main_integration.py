import os
import sys
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database import Base, SessionLocal, engine
from app.main import app, run_daily_job_search_pipeline
from app.models import JobHistory
from app.scraper import ScrapedJob

client = TestClient(app)


def setup_function():
    """Ensure clean database before each integration test."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def test_get_jobs_with_and_without_keyword():
    """Verify GET /jobs with optional job_keyword filtering."""
    db = SessionLocal()
    try:
        j1 = JobHistory(
            title="Senior Python Architect",
            company="Stripe",
            job_link="https://stripe.example/job1",
            match_score=94.0,
        )
        j2 = JobHistory(
            title="Frontend React Specialist",
            company="Vercel",
            job_link="https://vercel.example/job2",
            match_score=82.0,
        )
        db.add_all([j1, j2])
        db.commit()

        # 1. Without keyword (returns all, ordered by score descending)
        res_all = client.get("/jobs")
        assert res_all.status_code == 200
        items_all = res_all.json()
        assert len(items_all) == 2
        assert items_all[0]["title"] == "Senior Python Architect"

        # 2. With keyword filter
        res_kw = client.get("/jobs?job_keyword=Python")
        assert res_kw.status_code == 200
        items_kw = res_kw.json()
        assert len(items_kw) == 1
        assert items_kw[0]["title"] == "Senior Python Architect"

        # 3. With company filter keyword
        res_co = client.get("/jobs?job_keyword=Vercel")
        assert res_co.status_code == 200
        items_co = res_co.json()
        assert len(items_co) == 1
        assert items_co[0]["company"] == "Vercel"
    finally:
        db.close()


def test_generate_resume_by_job_id():
    """Verify POST /generate-resume/{id} for valid and invalid job IDs."""
    db = SessionLocal()
    try:
        job = JobHistory(
            title="AI Systems Engineer",
            company="OpenAI",
            job_link="https://openai.example/jobs/ai",
            match_score=92.5,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        job_id = job.id
    finally:
        db.close()

    mock_resume_result = {
        "tailored_markdown": "# ALEX RIVERA\nAI Systems Engineer",
        "filename": "Resume_OpenAI_AI_Systems_Engineer.pdf",
        "pdf_path": "output/Resume_OpenAI_AI_Systems_Engineer.pdf",
    }

    with patch("app.resume_builder.ResumeBuilder.build_tailored_resume_pdf", return_value=mock_resume_result):
        # Existing job ID
        response = client.post(f"/generate-resume/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["job_id"] == job_id
        assert data["company"] == "OpenAI"
        assert data["pdf_filename"] == "Resume_OpenAI_AI_Systems_Engineer.pdf"

        # Nonexistent job ID (404)
        not_found_res = client.post("/generate-resume/999999")
        assert not_found_res.status_code == 404
        assert "does not exist" in not_found_res.json()["detail"]


def test_generate_email_by_job_id():
    """Verify POST /generate-email/{id} for valid and invalid job IDs."""
    db = SessionLocal()
    try:
        job = JobHistory(
            title="Staff ML Platform Engineer",
            company="Anthropic",
            job_link="https://anthropic.example/jobs/ml",
            match_score=95.0,
            recruiter_email=None,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        job_id = job.id
    finally:
        db.close()

    mock_email = {
        "email": "recruiting@anthropic.com",
        "subject": "Staff ML Platform Engineer - Alex Rivera",
        "body": "Dear Anthropic Team, excited to apply.",
    }

    with patch("app.email_generator.EmailGenerator.generate_cold_email", return_value=mock_email):
        # Existing job ID
        response = client.post(f"/generate-email/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "recruiting@anthropic.com"
        assert data["subject"] == "Staff ML Platform Engineer - Alex Rivera"

        # Check that DB was updated with the discovered email
        db_check = SessionLocal()
        try:
            updated_job = db_check.query(JobHistory).filter(JobHistory.id == job_id).first()
            assert updated_job.recruiter_email == "recruiting@anthropic.com"
        finally:
            db_check.close()

        # Nonexistent job ID (404)
        nf_res = client.post("/generate-email/999999")
        assert nf_res.status_code == 404


def test_run_daily_job_search_pipeline():
    """Verify run_daily_job_search_pipeline scrapes, matches via RAG, persists, and prunes."""
    mock_scraped = [
        ScrapedJob(
            title="Senior Python Backend Engineer",
            company="DeepMind",
            description="High throughput FastAPI and ChromaDB vector systems.",
            job_link="https://deepmind.example/jobs/1",
            career_page_link="https://deepmind.example/careers",
        )
    ]

    with patch("app.scraper.JobScraper.scrape_jobs", return_value=mock_scraped):
        stats = run_daily_job_search_pipeline(job_keyword="Python")
        assert stats["scraped"] == 1
        assert stats["saved"] == 1
        assert stats["matched"] == 1

        # Check record in DB
        db = SessionLocal()
        try:
            stored = db.query(JobHistory).filter(JobHistory.company == "DeepMind").all()
            assert len(stored) == 1
            assert stored[0].match_score > 75.0
        finally:
            db.close()

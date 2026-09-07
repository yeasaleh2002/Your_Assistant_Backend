from datetime import date, datetime, timedelta, timezone
import os
import sys
from unittest.mock import patch

from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database import Base, SessionLocal, delete_jobs_by_date, engine
from app.main import app
from app.models import Job, JobStatus
from app.rag_engine import MatchedJob
from app.scraper import (
    PRIMARY_KEYWORDS,
    ScrapedJob,
    is_india_or_pakistan,
    is_last_24_hours,
    is_location_eligible,
)

client = TestClient(app)


def setup_function():
    """Reset database state for clean test isolation."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


# ==============================================================================
# Scraper Filter Tests (24h, Geography, India/Pakistan Exclusion)
# ==============================================================================

def test_primary_keywords_exact_12():
    """Verify that the 12 primary keywords are strictly configured."""
    assert len(PRIMARY_KEYWORDS) == 12
    expected_keywords = [
        "Frontend developer",
        "Frontend Engineer",
        "software engineer",
        "software developer",
        "web developer",
        "full stack developer",
        "react developer",
        "next.js developer",
        "python developer",
        "fast api developer",
        "vibe coder",
        "agentic full stack development",
    ]
    for kw in expected_keywords:
        assert kw in PRIMARY_KEYWORDS


def test_is_india_or_pakistan_filter():
    """Verify strict exclusion of Indian and Pakistani locations, companies, and domains."""
    # 1. Indian locations
    assert is_india_or_pakistan({"location": "Bengaluru, Karnataka, India"}) is True
    assert is_india_or_pakistan({"location": "Gurgaon, Haryana"}) is True
    assert is_india_or_pakistan({"location": "Mumbai, India"}) is True
    assert is_india_or_pakistan({"location": "Noida, Uttar Pradesh"}) is True
    assert is_india_or_pakistan({"location": "Hyderabad, Telangana"}) is True

    # 2. Pakistani locations
    assert is_india_or_pakistan({"location": "Lahore, Pakistan"}) is True
    assert is_india_or_pakistan({"location": "Karachi, Sindh"}) is True
    assert is_india_or_pakistan({"location": "Islamabad, Pakistan"}) is True

    # 3. Excluded domains
    assert is_india_or_pakistan({"link": "https://in.linkedin.com/jobs/view/12345"}) is True
    assert is_india_or_pakistan({"link": "https://pk.linkedin.com/jobs/view/67890"}) is True
    assert is_india_or_pakistan({"link": "https://www.naukri.com/job-listings"}) is True

    # 4. Valid locations (Bangladesh, USA, Germany, Remote)
    assert is_india_or_pakistan({"location": "Dhaka, Bangladesh"}) is False
    assert is_india_or_pakistan({"location": "Remote - Worldwide", "link": "https://example.com/job"}) is False
    assert is_india_or_pakistan({"location": "Berlin, Germany", "link": "https://example.de/job"}) is False


def test_is_location_eligible_logic():
    """
    Verify:
    1. India / Pakistan are strictly rejected.
    2. Bangladesh allows any type (onsite, hybrid, remote).
    3. Outside Bangladesh requires Remote.
    """
    # India/Pakistan rejected even if labeled Remote
    assert is_location_eligible({"location": "Remote, India"}) is False
    assert is_location_eligible({"location": "Remote, Lahore, Pakistan"}) is False

    # Bangladesh: All types allowed
    assert is_location_eligible({"location": "Dhaka, Bangladesh"}) is True
    assert is_location_eligible({"location": "Chittagong, Bangladesh", "description": "Onsite office"}) is True
    assert is_location_eligible({"location": "Banani, Dhaka", "detected_extensions": {"work_from_home": False}}) is True

    # Outside Bangladesh: Only Remote
    assert is_location_eligible({"location": "San Francisco, CA"}) is False
    assert is_location_eligible({"location": "Remote", "description": "100% remote opportunity"}) is True
    assert is_location_eligible({"location": "Anywhere", "detected_extensions": {"work_from_home": True}}) is True
    assert is_location_eligible({"location": "London, UK", "title": "Remote React Engineer"}) is True


def test_is_last_24_hours_filter():
    """Verify 24-hour recency check."""
    now = datetime.now(timezone.utc)

    # 1. Within 24h via detected_extensions
    assert is_last_24_hours({"detected_extensions": {"posted_at": "3 hours ago"}}) is True
    assert is_last_24_hours({"detected_extensions": {"posted_at": "35 minutes ago"}}) is True
    assert is_last_24_hours({"detected_extensions": {"posted_at": "1 day ago"}}) is True

    # Older than 24h
    assert is_last_24_hours({"detected_extensions": {"posted_at": "3 days ago"}}) is False
    assert is_last_24_hours({"detected_extensions": {"posted_at": "2 weeks ago"}}) is False

    # 2. DateTime object
    assert is_last_24_hours({"published_datetime": now - timedelta(hours=6)}) is True
    assert is_last_24_hours({"published_datetime": now - timedelta(hours=36)}) is False


# ==============================================================================
# Endpoint Integration Tests (Scrape, Jobs, Status Update, Delete by Date)
# ==============================================================================

def test_api_scrape_and_rag_match():
    """Verify POST /api/scrape scrapes, evaluates with RAG (>=65%), and saves for today."""
    mock_candidates = [
        ScrapedJob(
            title="Senior Frontend Engineer",
            company="Stripe",
            description="React, TypeScript, Next.js",
            job_link="https://stripe.example/jobs/fe-1",
            career_page_link="https://stripe.example/careers",
            location="Remote",
        ),
        ScrapedJob(
            title="Junior PHP Developer",
            company="OldTech",
            description="Legacy PHP",
            job_link="https://oldtech.example/jobs/php",
            career_page_link="https://oldtech.example/careers",
            location="Remote",
        ),
    ]

    # Only Stripe passes >= 65%
    mock_matched = [
        MatchedJob(
            title="Senior Frontend Engineer",
            company="Stripe",
            description="React, TypeScript, Next.js",
            job_link="https://stripe.example/jobs/fe-1",
            career_page_link="https://stripe.example/careers",
            location="Remote",
            match_score=88.0,
        )
    ]

    with patch("app.scraper.JobScraper.scrape_jobs", return_value=mock_candidates), \
         patch("app.rag_engine.RAGEngine.match_jobs", return_value=mock_matched):
        res = client.post("/api/scrape")
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["status"] == "success"
        assert data["scraped_count"] == 2
        assert data["matched_count"] == 1
        assert data["saved_count"] == 1
        assert data["scraped_date"] == date.today().isoformat()

        # Check DB
        db = SessionLocal()
        try:
            stored = db.query(Job).all()
            assert len(stored) == 1
            assert stored[0].title == "Senior Frontend Engineer"
            assert stored[0].company == "Stripe"
            assert stored[0].status == JobStatus.Pending
            assert stored[0].scraped_date == date.today()
            assert stored[0].match_score == 88.0
        finally:
            db.close()


def test_get_jobs_by_date_endpoint():
    """Verify GET /api/jobs defaults to today's date and allows explicit date filtering."""
    db = SessionLocal()
    try:
        today = date.today()
        yesterday = today - timedelta(days=1)

        job_today = Job(
            title="Full Stack Developer",
            company="Vercel",
            link="https://vercel.example/jobs/1",
            match_score=92.0,
            location="Remote",
            status=JobStatus.Pending,
            scraped_date=today,
        )
        job_yesterday = Job(
            title="FastAPI Developer",
            company="Supabase",
            link="https://supabase.example/jobs/2",
            match_score=85.0,
            location="Remote",
            status=JobStatus.Applied,
            scraped_date=yesterday,
        )
        db.add_all([job_today, job_yesterday])
        db.commit()

        # 1. Default (no date query param) -> returns today's jobs only
        res_today = client.get("/api/jobs")
        assert res_today.status_code == 200
        jobs_today = res_today.json()
        assert len(jobs_today) == 1
        assert jobs_today[0]["company"] == "Vercel"
        assert jobs_today[0]["status"] == "Pending"

        # 2. Explicit date query param
        res_yesterday = client.get(f"/api/jobs?date={yesterday.isoformat()}")
        assert res_yesterday.status_code == 200
        jobs_y = res_yesterday.json()
        assert len(jobs_y) == 1
        assert jobs_y[0]["company"] == "Supabase"
        assert jobs_y[0]["status"] == "Applied"
    finally:
        db.close()


def test_patch_job_status_endpoint():
    """Verify PATCH /api/jobs/{id}/status updates the status column and validates inputs."""
    db = SessionLocal()
    try:
        job = Job(
            title="AI Agent Engineer",
            company="LangChain",
            link="https://langchain.example/jobs/1",
            match_score=89.0,
            location="Remote",
            status=JobStatus.Pending,
            scraped_date=date.today(),
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        job_id = job.id
    finally:
        db.close()

    # 1. Valid status updates
    for new_status in ["Applied", "Interview", "Rejected", "Pending"]:
        res = client.patch(f"/api/jobs/{job_id}/status", json={"status": new_status})
        assert res.status_code == 200, res.text
        assert res.json()["status"] == new_status

        # Check DB
        db_check = SessionLocal()
        try:
            updated = db_check.query(Job).filter(Job.id == job_id).first()
            assert updated.status.value == new_status
        finally:
            db_check.close()

    # 2. Invalid status (422 Unprocessable Entity)
    bad_res = client.patch(f"/api/jobs/{job_id}/status", json={"status": "Ghosted"})
    assert bad_res.status_code == 422

    # 3. Nonexistent job ID (404 Not Found)
    nf_res = client.patch("/api/jobs/999999/status", json={"status": "Applied"})
    assert nf_res.status_code == 404


def test_delete_jobs_by_date_endpoint():
    """Verify DELETE /api/jobs/date/{date} deletes all records for that date."""
    db = SessionLocal()
    try:
        target_date = date(2026, 9, 1)
        other_date = date(2026, 9, 2)

        j1 = Job(
            title="Dev 1",
            company="Co 1",
            link="https://co1.example/job",
            match_score=80.0,
            location="Remote",
            status=JobStatus.Pending,
            scraped_date=target_date,
        )
        j2 = Job(
            title="Dev 2",
            company="Co 2",
            link="https://co2.example/job",
            match_score=75.0,
            location="Remote",
            status=JobStatus.Pending,
            scraped_date=target_date,
        )
        j3 = Job(
            title="Dev 3",
            company="Co 3",
            link="https://co3.example/job",
            match_score=90.0,
            location="Remote",
            status=JobStatus.Pending,
            scraped_date=other_date,
        )
        db.add_all([j1, j2, j3])
        db.commit()

        # Delete for target_date
        res = client.delete(f"/api/jobs/date/{target_date.isoformat()}")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["deleted_count"] == 2
        assert data["date"] == target_date.isoformat()

        # Verify DB: only j3 remains
        db_check = SessionLocal()
        try:
            remaining = db_check.query(Job).all()
            assert len(remaining) == 1
            assert remaining[0].company == "Co 3"
        finally:
            db_check.close()
    finally:
        db.close()

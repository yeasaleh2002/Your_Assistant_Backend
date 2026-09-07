import os
import sys
from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient
import pytest

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database import Base, SessionLocal, delete_records_older_than, engine
from app.main import app
from app.models import JobHistory

client = TestClient(app)


def setup_function():
    """Ensure a clean database state for each test run."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def test_antigravity_easter_egg_and_health():
    """Verify health endpoint and easter egg presence."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["easter_egg"] == "antigravity active"
    assert data["status"] == "healthy"


def test_create_and_query_job_history():
    """Verify standard job creation and retrieval."""
    payload = {
        "title": "Staff AI Engineer",
        "company": "Anthropic",
        "job_link": "https://anthropic.com/careers/staff-ai-engineer",
        "career_page_link": "https://anthropic.com/careers",
        "match_score": 96.5,
        "recruiter_email": "recruiting@anthropic.com",
    }
    response = client.post("/api/jobs", json=payload)
    assert response.status_code == 201, response.text
    created = response.json()
    assert created["id"] is not None
    assert created["title"] == payload["title"]
    assert created["match_score"] == 96.5

    # List endpoint
    list_res = client.get("/api/jobs")
    assert list_res.status_code == 200
    items = list_res.json()
    assert len(items) == 1
    assert items[0]["company"] == "Anthropic"


def test_strict_validation_prevent_malicious_payloads():
    """Verify Pydantic validation rejects bad scores, invalid URLs, script injection, and extra fields."""
    # 1. Invalid match_score (> 100)
    bad_score = {
        "title": "Machine Learning Engineer",
        "company": "OpenAI",
        "job_link": "https://openai.com/careers",
        "match_score": 150.0,
    }
    res = client.post("/api/jobs", json=bad_score)
    assert res.status_code == 422

    # 2. Invalid URL
    bad_url = {
        "title": "Machine Learning Engineer",
        "company": "OpenAI",
        "job_link": "not-a-valid-url",
        "match_score": 85.0,
    }
    res = client.post("/api/jobs", json=bad_url)
    assert res.status_code == 422

    # 3. Malicious script tag in title
    script_injection = {
        "title": "<script>alert('xss')</script> Engineer",
        "company": "Tech Corp",
        "job_link": "https://techcorp.example/jobs/1",
        "match_score": 75.0,
    }
    res = client.post("/api/jobs", json=script_injection)
    assert res.status_code == 422

    # 4. Extra/unexpected fields (forbidden by extra='forbid')
    extra_fields = {
        "title": "Data Scientist",
        "company": "DeepMind",
        "job_link": "https://deepmind.google/careers",
        "match_score": 90.0,
        "isAdmin": True,
        "role": "admin",
    }
    res = client.post("/api/jobs", json=extra_fields)
    assert res.status_code == 422


def test_automated_cleanup_function_older_than_7_days():
    """Verify that records older than 7 days are deleted while newer records remain."""
    db = SessionLocal()
    try:
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(days=9)
        recent_time = now - timedelta(days=2)

        # Insert expired record
        expired_job = JobHistory(
            title="Old Job 10 days ago",
            company="Legacy Co",
            job_link="https://legacy.example.com/job",
            match_score=70.0,
            created_at=old_time,
        )
        # Insert fresh record
        fresh_job = JobHistory(
            title="Fresh Job 2 days ago",
            company="Modern Co",
            job_link="https://modern.example.com/job",
            match_score=95.0,
            created_at=recent_time,
        )

        db.add_all([expired_job, fresh_job])
        db.commit()

        # Check that both exist
        count_before = db.query(JobHistory).count()
        assert count_before == 2

        # Run automated retention cleanup (threshold: 7 days)
        deleted = delete_records_older_than(days=7, db=db)
        assert deleted == 1

        # Check remaining records
        remaining = db.query(JobHistory).all()
        assert len(remaining) == 1
        assert remaining[0].title == "Fresh Job 2 days ago"
    finally:
        db.close()


def test_rate_limiting_triggers_429():
    """Verify that hammering the cleanup endpoint triggers HTTP 429 Too Many Requests."""
    # /api/jobs/cleanup has a strict 5/minute limit
    statuses = []
    for _ in range(8):
        res = client.post("/api/jobs/cleanup?retention_days=7")
        statuses.append(res.status_code)

    assert 429 in statuses


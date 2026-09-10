import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database import Base, SessionLocal, engine
from app.main import app
from app.models import JobHistory
from app.scraper import (
    DEFAULT_FALLBACK_QUERY,
    JobScraper,
    predict_career_page_url,
)

client = TestClient(app)


def setup_function():
    """Reset database state for clean test isolation without dropping companies table."""
    db = SessionLocal()
    try:
        db.query(JobHistory).delete()
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def test_scraper_query_builder_keyword_and_fallback():
    """Verify exact keyword prioritization and default query fallback."""
    scraper = JobScraper(api_key="test_key")

    # Provided keyword should be exact-matched in quotes
    assert scraper.build_query("AI Platform Engineer") == '"AI Platform Engineer"'
    assert scraper.build_query("  MLOps Specialist   ") == '"MLOps Specialist"'

    # Blank / None keywords should fallback to default query
    assert scraper.build_query(None) == DEFAULT_FALLBACK_QUERY
    assert scraper.build_query("") == DEFAULT_FALLBACK_QUERY
    assert scraper.build_query("   ") == DEFAULT_FALLBACK_QUERY


def test_predict_career_page_url():
    """Verify career page URL prediction for standard companies and known ATS portals."""
    # 1. Standard company with corporate suffixes
    assert predict_career_page_url("Acme Corp.") == "https://acme.com/careers"
    assert predict_career_page_url("DeepMind Technologies LLC") == "https://deepmind.com/careers"

    # 2. Greenhouse ATS detection
    gh_link = "https://boards.greenhouse.io/figma/jobs/56789"
    assert predict_career_page_url("Figma", gh_link) == "https://boards.greenhouse.io/figma"

    # 3. Lever ATS detection
    lever_link = "https://jobs.lever.co/spotify/12345"
    assert predict_career_page_url("Spotify", lever_link) == "https://jobs.lever.co/spotify"

    # 4. Non-aggregator domain
    custom_domain = "https://jobs.datadoghq.com/detail/1234"
    assert predict_career_page_url("Datadog", custom_domain) == "https://jobs.datadoghq.com/careers"


def test_duplicate_check_filters_jobs_saved_within_7_days():
    """Verify that jobs saved in JobHistory within the last 7 days are filtered out."""
    db = SessionLocal()
    try:
        now = datetime.now(timezone.utc)
        recent_link = "https://company.example/jobs/recent"
        old_link = "https://company.example/jobs/old-8-days"
        fresh_link = "https://company.example/jobs/brand-new"

        # 1. Job saved 2 days ago (within 7-day retention window)
        db.add(
            JobHistory(
                title="Recent Job",
                company="Company A",
                job_link=recent_link,
                match_score=90.0,
                created_at=now - timedelta(days=2),
            )
        )
        # 2. Job saved 9 days ago (outside 7-day retention window, eligible to re-show)
        db.add(
            JobHistory(
                title="Old Job",
                company="Company B",
                job_link=old_link,
                match_score=85.0,
                created_at=now - timedelta(days=9),
            )
        )
        db.commit()

        # Mock SerpApi response returning all 3 jobs
        mock_raw_jobs = [
            {
                "title": "Recent Job",
                "company_name": "Company A",
                "description": "Should be filtered as duplicate",
                "apply_options": [{"link": recent_link}],
            },
            {
                "title": "Old Job",
                "company_name": "Company B",
                "description": "Allowed because older than 7 days",
                "apply_options": [{"link": old_link}],
            },
            {
                "title": "Brand New Job",
                "company_name": "Company C",
                "description": "Brand new job posting",
                "apply_options": [{"link": fresh_link}],
            },
        ]

        scraper = JobScraper(api_key="mock_key")
        with patch.object(scraper, "fetch_serpapi_jobs", return_value=mock_raw_jobs):
            scraped = scraper.scrape_jobs(job_keyword="Engineer", db=db)

        # Only old_link and fresh_link should remain (recent_link omitted)
        returned_links = [j.job_link for j in scraped]
        assert recent_link not in returned_links
        assert old_link in returned_links
        assert fresh_link in returned_links
        assert len(scraped) == 2
        assert scraped[0].title == "Old Job"
        assert scraped[1].title == "Brand New Job"
        assert scraped[1].company == "Company C"
    finally:
        db.close()


def test_scrape_api_endpoint_integration():
    """Verify the /api/jobs/scrape endpoint runs successfully and applies duplicate checks."""
    mock_jobs = [
        {
            "title": "Lead Software Architect",
            "company_name": "Global Tech",
            "description": "Leading distributed cloud systems architecture.",
            "apply_options": [{"link": "https://globaltech.example/jobs/arch-1"}],
        }
    ]

    with patch("app.scraper.JobScraper.fetch_serpapi_jobs", return_value=mock_jobs):
        response = client.post("/api/jobs/scrape?keyword=Software+Architect&limit=5")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["title"] == "Lead Software Architect"
        assert data[0]["company"] == "Global Tech"
        assert data[0]["career_page_link"] == "https://globaltech.example/careers"

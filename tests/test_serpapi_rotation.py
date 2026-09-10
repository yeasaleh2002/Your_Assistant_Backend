import os
from unittest.mock import MagicMock, patch
import pytest

from app.scraper import JobScraper


def test_serpapi_multi_key_parsing():
    """Verify that multiple comma-separated keys are parsed and trimmed correctly."""
    scraper = JobScraper(api_key="key_alpha,  key_beta , key_gamma")
    assert scraper.api_keys == ["key_alpha", "key_beta", "key_gamma"]
    assert scraper.api_key == "key_alpha"


def test_serpapi_multi_key_from_env(monkeypatch):
    """Verify loading comma-separated keys directly from SERPAPI_API_KEY environment variable."""
    monkeypatch.setenv("SERPAPI_API_KEY", "token1234567, token8901234")
    scraper = JobScraper()
    assert scraper.api_keys == ["token1234567", "token8901234"]


def test_serpapi_automatic_rotation_on_rate_limit_or_quota():
    """Verify scraper automatically rotates to next key when key 1 fails with 429 or quota limit."""
    scraper = JobScraper(api_key="exhausted_key, working_key")

    mock_resp_exhausted = MagicMock()
    mock_resp_exhausted.status_code = 429
    mock_resp_exhausted.text = "Rate limit exceeded"

    mock_resp_working = MagicMock()
    mock_resp_working.status_code = 200
    mock_resp_working.json.return_value = {
        "jobs_results": [
            {
                "title": "Senior Frontend Developer",
                "company_name": "Tech Corp",
                "share_link": "https://example.com/job/1",
            }
        ]
    }

    with patch("requests.get", side_effect=[mock_resp_exhausted, mock_resp_working]) as mock_get:
        results = scraper.fetch_serpapi_jobs(query="Frontend developer", limit=5)
        assert len(results) == 1
        assert results[0]["title"] == "Senior Frontend Developer"
        assert scraper.api_key == "working_key"
        assert mock_get.call_count == 2


def test_serpapi_automatic_rotation_on_json_quota_error():
    """Verify scraper rotates to next key when key 1 returns 200 with an error in the JSON payload."""
    scraper = JobScraper(api_key="depleted_key, active_key")

    mock_resp_depleted = MagicMock()
    mock_resp_depleted.status_code = 200
    mock_resp_depleted.json.return_value = {
        "error": "Your account has run out of searches. Please upgrade your plan."
    }

    mock_resp_active = MagicMock()
    mock_resp_active.status_code = 200
    mock_resp_active.json.return_value = {
        "jobs_results": [
            {
                "title": "Full Stack Engineer",
                "company_name": "SaaS Platform",
                "share_link": "https://example.com/job/2",
            }
        ]
    }

    with patch("requests.get", side_effect=[mock_resp_depleted, mock_resp_active]) as mock_get:
        results = scraper.fetch_serpapi_jobs(query="Full stack", limit=5)
        assert len(results) == 1
        assert results[0]["title"] == "Full Stack Engineer"
        assert scraper.api_key == "active_key"
        assert mock_get.call_count == 2

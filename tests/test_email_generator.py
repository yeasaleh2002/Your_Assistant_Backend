import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.email_generator import (
    DEFAULT_MISSING_EMAIL,
    EmailGenerator,
    extract_recruiter_email,
)


def test_extract_recruiter_email_success():
    """Verify regex correctly extracts a recruiter email from job description text."""
    text = (
        "We are looking for a Senior Python Engineer. For immediate consideration, "
        "reach out to sarah.recruiter@techinnovations.com with your portfolio."
    )
    email = extract_recruiter_email(text)
    assert email == "sarah.recruiter@techinnovations.com"


def test_extract_recruiter_email_missing_fallback():
    """Verify missing email correctly returns '[Recruiter Email]'."""
    text = "We are looking for an AI Engineer. Apply directly via our career portal."
    email = extract_recruiter_email(text)
    assert email == DEFAULT_MISSING_EMAIL


def test_generate_cold_email_json_output(tmp_path):
    """Verify generate_cold_email produces the exact required JSON structure."""
    resume_file = tmp_path / "resume.txt"
    resume_file.write_text(
        "ALEX RIVERA\nSenior Python & AI Engineer\nSkills: Python, FastAPI, ChromaDB, RAG",
        encoding="utf-8",
    )

    generator = EmailGenerator(base_resume_path=resume_file)

    mock_llm_json = json.dumps({
        "email": "hiring@anthropic.com",
        "subject": "Senior AI Systems Engineer - Alex Rivera | Python & RAG Specialist",
        "body": "Hi Team,\n\nI noticed your opening for a Senior AI Engineer. With 7+ years scaling FastAPI and ChromaDB pipelines, I would love to contribute to Anthropic.\n\nBest,\nAlex Rivera",
    })

    with patch("app.email_generator.generate_ai_response", return_value=mock_llm_json) as mock_llm:
        result = generator.generate_cold_email(
            job_title="Senior AI Engineer",
            job_description="Looking for Python and ChromaDB experience. Contact: hiring@anthropic.com",
            company="Anthropic",
        )

        assert isinstance(result, dict)
        assert result["email"] == "hiring@anthropic.com"
        assert "Senior AI Systems Engineer" in result["subject"]
        assert "ChromaDB" in result["body"]

        # Verify anti-hallucination prompt was used
        args, kwargs = mock_llm.call_args
        assert "DO NOT fabricate or hallucinate" in kwargs["system_prompt"]
        assert "ALEX RIVERA" in kwargs["prompt"]


def test_generate_cold_email_with_markdown_fences(tmp_path):
    """Verify json parser cleans ```json ... ``` markdown code fences."""
    resume_file = tmp_path / "resume.txt"
    resume_file.write_text("ALEX RIVERA\nPython Developer", encoding="utf-8")
    generator = EmailGenerator(base_resume_path=resume_file)

    fenced_output = """```json
{
  "email": "[Recruiter Email]",
  "subject": "Application for Python Engineer",
  "body": "Dear Hiring Manager, I am excited to apply."
}
```"""

    with patch("app.email_generator.generate_ai_response", return_value=fenced_output):
        result = generator.generate_cold_email(
            job_title="Python Engineer",
            job_description="No contact email here.",
            company="Acme Corp",
        )

        assert result["email"] == "[Recruiter Email]"
        assert result["subject"] == "Application for Python Engineer"
        assert result["body"] == "Dear Hiring Manager, I am excited to apply."


def test_api_generate_email_endpoint():
    """Verify POST /api/email/generate endpoint."""
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    mock_email_result = {
        "email": "recruiting@deepmind.com",
        "subject": "Senior AI Platform Engineer - Candidate Introduction",
        "body": "Dear DeepMind Recruiting Team, excited to connect regarding the AI Engineer role.",
    }


    with patch("app.email_generator.EmailGenerator.generate_cold_email", return_value=mock_email_result):
        response = client.post(
            "/api/email/generate",
            json={
                "job_title": "Senior AI Platform Engineer",
                "job_description": "Join our AI Platform team. Reach out to recruiting@deepmind.com",
                "company": "DeepMind",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "recruiting@deepmind.com"
        assert "Senior AI Platform Engineer" in data["subject"]
        assert "DeepMind" in data["body"]

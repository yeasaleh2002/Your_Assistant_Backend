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
        "cover_letter": "Dear Anthropic Hiring Team,\n\nI am applying for the Senior AI Engineer role. With extensive experience in Python and FastAPI, I look forward to contributing.",
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


def test_generate_cover_letter_for_job_api():
    """Verify POST /api/jobs/{id}/cover-letter and /api/jobs/{id}/email return tailored cover letter."""
    from fastapi.testclient import TestClient
    from app.main import app
    from app.database import SessionLocal
    from app.models import Job
    import datetime

    client = TestClient(app)
    db = SessionLocal()
    test_job = Job(
        title="Full Stack Cloud Architect",
        company="NexTech Solutions",
        link="https://nextech.io/jobs/123",
        match_score=87.5,
        location="Remote",
        status="Pending",
        scraped_date=datetime.date.today(),
        description="Looking for an engineer skilled in Next.js, Node.js, and FastAPI.",
        recruiter_email="hiring@nextech.io",
    )
    db.add(test_job)
    db.commit()
    db.refresh(test_job)
    job_id = test_job.id

    try:
        fake_email = {
            "email": "hiring@nextech.io",
            "subject": "Full Stack Cloud Architect Application - Yeasaleh | NexTech Solutions",
            "body": "Dear NexTech Team, I am thrilled to apply for the Full Stack Cloud Architect opening.",
            "cover_letter": "Dear NexTech Solutions Team,\n\nI am writing to submit my application for the Full Stack Cloud Architect position.",
        }
        fake_cl = "Dear NexTech Solutions Team,\n\nI am writing to submit my application for the Full Stack Cloud Architect position with proven expertise in Next.js and FastAPI."

        with patch("app.email_generator.EmailGenerator.generate_cover_letter", return_value=fake_cl), \
             patch("app.email_generator.EmailGenerator.generate_cold_email", return_value=fake_email):

            # 1. Test POST /api/jobs/{id}/cover-letter
            res_cl = client.post(f"/api/jobs/{job_id}/cover-letter")
            assert res_cl.status_code == 200
            cl_data = res_cl.json()
            assert cl_data["status"] == "success"
            assert cl_data["job_id"] == job_id
            assert cl_data["job_title"] == "Full Stack Cloud Architect"
            assert cl_data["company"] == "NexTech Solutions"
            assert len(cl_data["cover_letter"]) > 50
            assert "Full Stack Cloud Architect" in cl_data["cover_letter"]
            assert "email_cover_letter" in cl_data
            assert "Full Stack Cloud Architect" in cl_data["email_cover_letter"]["subject"]

            # 2. Test POST /api/jobs/{id}/email
            res_email = client.post(f"/api/jobs/{job_id}/email")
            assert res_email.status_code == 200
            email_data = res_email.json()
            assert email_data["email"] == "hiring@nextech.io"
            assert "Full Stack Cloud Architect" in email_data["subject"]
            assert len(email_data["body"]) > 30
            assert "cover_letter" in email_data
            assert len(email_data["cover_letter"]) > 50
    finally:
        db.delete(test_job)
        db.commit()
        db.close()

import json
import os
from pathlib import Path
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_ai_generator():
    """Mock generate_ai_response to provide instantaneous, deterministic responses for tests."""
    def _fake_generate(prompt, system_prompt=None, response_schema=None, temperature=0.2):
        combined = (str(prompt) + " " + str(system_prompt)).lower()
        if "cover letter" in combined:
            # Check target role in prompt
            role = "Target Role"
            if "senior react engineer" in combined:
                role = "Senior React Engineer"
            elif "fastapi backend specialist" in combined:
                role = "FastAPI Backend Specialist"
            elif "staff frontend architect" in combined:
                role = "Staff Frontend Architect"

            return (
                f"Dear Hiring Team,\n\n"
                f"I am writing to express my enthusiastic interest in the {role} position. "
                f"With hands-on experience architecting full-stack web applications, reducing latency, "
                f"and building scalable systems, I am confident in my ability to deliver immediate value.\n\n"
                f"Sincerely,\nYeasaleh\nSoftware Engineer"
            )
        elif response_schema is not None or "cold outreach email" in combined:
            role = "Software Engineer"
            company = "the team"
            if "senior react engineer" in combined:
                role = "Senior React Engineer"
            if "technova corp" in combined:
                company = "TechNova Corp"
            return json.dumps({
                "email": "careers@technova.io",
                "subject": f"{role} Application - Yeasaleh | {company}",
                "body": f"Hi Team at {company},\n\nI noticed your opening for a {role}. With experience building scalable web applications and optimizing latency, I would love to contribute.\n\nBest regards,\nYeasaleh\n{role}",
            })
        else:
            # Resume markdown generator
            role = "Software Engineer"
            if "senior react engineer" in combined:
                role = "Senior React Engineer"
            elif "fastapi backend specialist" in combined:
                role = "FastAPI Backend Specialist"
            elif "staff frontend architect" in combined:
                role = "Staff Frontend Architect"

            return (
                f"# Yeasaleh | {role}\n"
                f"Dhaka, Bangladesh | +8801735782467 | yeasaleh.contact@gmail.com\n"
                f"https://www.linkedin.com/in/yea-saleh | https://github.com/yeasaleh2002 | https://yeasaleh.xyz\n\n"
                f"## Professional Summary\n"
                f"Dedicated {role} with extensive experience architecting high-performance web applications and backend microservices.\n\n"
                f"## Technical Skills\n"
                f"**Front-End:** React, Next.js, TypeScript, Tailwind CSS\n"
                f"**Back-End:** Python, FastAPI, Node.js, PostgreSQL\n"
                f"**AI & Automation Tools:** FastEmbed, Vector RAG, ChromaDB\n\n"
                f"## Professional Experience\n\n"
                f"### Nurix Hive | Software Engineer\n"
                f"*Dhaka, Bangladesh (Remote) | August 2025 – Present*\n"
                f"- Architected real-time dashboard components reducing page load latency by 42%.\n"
                f"- Slashed API response times by 35% through query indexing and Redis caching.\n"
                f"- Engineered automated CI/CD pipeline achieving 99.9% uptime over 6 months.\n"
                f"- Mentored 4 junior engineers on React and TypeScript architecture standards.\n\n"
                f"### Manaknight Digital | Full-Stack Developer\n"
                f"*Remote | January 2024 – July 2025*\n"
                f"- Developed 15+ responsive full-stack features using React and Node.js.\n"
                f"- Improved client test coverage from 60% to 92% across critical microservices.\n"
                f"- Reduced customer churn by 18% by redesigning checkout user flows.\n"
                f"- Optimized PostgreSQL database queries cutting 95th-percentile response time by 40%.\n\n"
                f"### MedLink Healthcare Private Limited | Frontend Developer\n"
                f"*Hybrid | June 2023 – December 2023*\n"
                f"- Built patient portal interfaces serving over 20,000 monthly active users.\n"
                f"- Accelerated UI rendering performance to consistent 60 fps on mobile browsers.\n"
                f"- Implemented secure HIPAA-compliant authentication flows with OAuth2.\n"
                f"- Standardized component library decreasing new feature turnaround time by 30%.\n\n"
                f"## Mentorship Experience\n\n"
                f"### Sadhinota Camp | Technical Mentor\n"
                f"*Dhaka, Bangladesh | 2023 – 2024*\n"
                f"- Mentored 30+ students in modern web development, algorithms, and git workflows.\n\n"
                f"## Education\n\n"
                f"**Bachelor of Science in Computer Science and Engineering**\n"
                f"Daffodil International University | Dhaka, Bangladesh\n\n"
                f"## Language\n\n"
                f"English (Fluent), Bengali (Native)\n"
            )

    with patch("app.resume_builder.generate_ai_response", side_effect=_fake_generate), \
         patch("app.email_generator.generate_ai_response", side_effect=_fake_generate):
        yield


def test_custom_job_description_with_company_and_title():
    """Test custom job description tailoring with explicit company and title."""
    jd_text = """
    We are seeking a Senior React Engineer with deep experience in Next.js, TypeScript, and state management.
    Requirements:
    - 4+ years experience with React, Next.js, Tailwind CSS, Redux/Zustand.
    - Strong API integration with REST/GraphQL and WebSockets.
    - Proven track record of improving web performance and Core Web Vitals.
    """
    payload = {
        "job_description": jd_text,
        "job_title": "Senior React Engineer",
        "company": "TechNova Corp",
        "recruiter_email": "careers@technova.io",
    }

    response = client.post("/api/job-description/tailor", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["job_title"] == "Senior React Engineer"
    assert data["company"] == "TechNova Corp"
    assert isinstance(data["match_score"], (int, float))
    assert 0.0 <= data["match_score"] <= 100.0

    # Test PDF naming convention: Yeasaleh_Resume_{clean_company}_{clean_title}.pdf
    expected_filename = "Yeasaleh_Resume_TechNova_Corp_Senior_React_Engineer.pdf"
    assert data["pdf_filename"] == expected_filename
    assert data["download_url"] == f"/api/resume/download/{expected_filename}"

    # Header in tailored resume markdown must use target job title
    assert "Senior React Engineer" in data["tailored_resume_markdown"]
    assert "# Yeasaleh | Senior React Engineer" in data["tailored_resume_markdown"]

    # Cold email validation
    cold_email = data["cold_email"]
    assert "Senior React Engineer" in cold_email["subject"]
    assert "TechNova Corp" in cold_email["subject"]
    assert "Senior React Engineer" in cold_email["body"]

    # Cover letter validation
    cover_letter = data["cover_letter"]
    assert len(cover_letter) > 100
    assert "Senior React Engineer" in cover_letter

    # Verify download endpoint
    dl_response = client.get(data["download_url"])
    assert dl_response.status_code == 200
    assert dl_response.headers["content-type"] == "application/pdf"
    assert len(dl_response.content) > 1000


def test_custom_job_description_without_company():
    """Test custom job description tailoring without company name."""
    jd_text = """
    Looking for a skilled Python FastAPI Backend Engineer to design microservices and AI agent workflows.
    Required: Python, FastAPI, PostgreSQL, Docker, Redis, Celery, Vector RAG.
    """
    payload = {
        "job_description": jd_text,
        "job_title": "FastAPI Backend Specialist",
        "company": None,
    }

    response = client.post("/api/job-description/tailor", json=payload)
    assert response.status_code == 200
    data = response.json()

    # When no company name is provided, format must be Yeasaleh_Resume_{clean_title}.pdf
    expected_filename = "Yeasaleh_Resume_FastAPI_Backend_Specialist.pdf"
    assert data["pdf_filename"] == expected_filename
    assert data["company"] is None
    assert data["download_url"] == f"/api/resume/download/{expected_filename}"

    # Ensure header has new title
    assert "# Yeasaleh | FastAPI Backend Specialist" in data["tailored_resume_markdown"]


def test_custom_job_description_title_inference_from_text():
    """Test inferring job title when omitted in the payload."""
    jd_text = """Job Title: Staff Frontend Architect
    We are looking for a Staff Frontend Architect to lead our client-side platform team.
    Stack: React, TypeScript, Micro-frontends, Vite, Next.js.
    """
    payload = {
        "job_description": jd_text,
    }

    response = client.post("/api/job-description/tailor", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["job_title"] == "Staff Frontend Architect"
    assert "Staff_Frontend_Architect" in data["pdf_filename"]
    assert "# Yeasaleh | Staff Frontend Architect" in data["tailored_resume_markdown"]


def test_custom_job_description_offline_fallback():
    """Verify endpoint functions seamlessly even when all external LLM providers are offline/exhausted."""
    from app.llm_manager import AllProvidersExhaustedError

    with patch("app.resume_builder.generate_ai_response", side_effect=AllProvidersExhaustedError("All API keys failed")), \
         patch("app.email_generator.generate_ai_response", side_effect=AllProvidersExhaustedError("All API keys failed")):

        payload = {
            "job_description": "We need a Senior Full Stack Engineer experienced with Next.js and FastAPI microservices.",
            "job_title": "Lead Software Architect",
            "company": "Aura Cloud",
        }

        response = client.post("/api/job-description/tailor", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "success"
        assert data["job_title"] == "Lead Software Architect"
        assert data["pdf_filename"] == "Yeasaleh_Resume_Aura_Cloud_Lead_Software_Architect.pdf"
        assert "# Yeasaleh | Lead Software Architect" in data["tailored_resume_markdown"]
        assert len(data["cover_letter"]) > 100
        assert "Lead Software Architect" in data["cover_letter"]
        assert "Lead Software Architect" in data["cold_email"]["subject"]

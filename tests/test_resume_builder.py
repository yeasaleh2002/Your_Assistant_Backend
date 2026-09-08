import os
from pathlib import Path
import sys
from unittest.mock import patch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.resume_builder import (
    ResumeBuilder,
    STRICT_ATS_PROMPT,
    format_markdown_for_reportlab,
)


@pytest.fixture
def sample_resume_file(tmp_path):
    f = tmp_path / "base_resume.txt"
    f.write_text(
        """
        ALEX RIVERA
        Senior Python & AI Platform Engineer
        alex.rivera@example.com | San Francisco, CA

        SUMMARY
        Senior Python engineer with 7+ years building FastAPI and RAG architectures.

        SKILLS
        Python, FastAPI, SQLAlchemy, ChromaDB, Docker

        EXPERIENCE
        Senior AI Engineer | HiveTech | 2022 - Present
        - Scaled ChromaDB vector search to sub-50ms latency.
        """,
        encoding="utf-8",
    )
    return f


def test_strict_ats_prompt_content():
    """Verify that the strict anti-hallucination ATS prompt requirement is exact."""
    expected = (
        "Rewrite the resume for 100% ATS compatibility. You MUST ONLY use skills and "
        "experiences present in the original resume. DO NOT hallucinate or add any fake skills."
    )
    assert STRICT_ATS_PROMPT == expected


def test_tailor_resume_invokes_llm_with_strict_constraint(sample_resume_file):
    """Verify tailor_resume passes the strict prompt as system_prompt to llm_manager."""
    builder = ResumeBuilder(base_resume_path=sample_resume_file)

    mock_llm_response = (
        "# ALEX RIVERA\n"
        "Senior Python Engineer | alex.rivera@example.com\n\n"
        "## Professional Summary\n"
        "Experienced Python and FastAPI engineer.\n\n"
        "## Technical Skills\n"
        "- Python, FastAPI, ChromaDB\n"
    )

    with patch("app.resume_builder.generate_ai_response", return_value=mock_llm_response) as mock_gen:
        result = builder.tailor_resume(
            job_title="Lead FastAPI Engineer",
            job_description="Looking for Python and FastAPI expert with ChromaDB experience.",
            company="Acme Corp",
        )

        assert result == mock_llm_response.strip()
        mock_gen.assert_called_once()

        args, kwargs = mock_gen.call_args
        assert kwargs["system_prompt"] == STRICT_ATS_PROMPT
        assert "Lead FastAPI Engineer at Acme Corp" in kwargs["prompt"]
        assert STRICT_ATS_PROMPT in kwargs["prompt"]


def test_markdown_sanitization_and_inline_tags():
    """Verify safe HTML escaping and markdown tag conversion."""
    raw = "Proficient in C++ & Python **FastAPI** and *SQLAlchemy* with `docker run`."
    formatted = format_markdown_for_reportlab(raw)
    assert "&amp;" in formatted
    assert "<b>FastAPI</b>" in formatted
    assert "<i>SQLAlchemy</i>" in formatted
    assert '<font face="Courier">docker run</font>' in formatted


def test_generate_pdf_creates_valid_pdf(tmp_path):
    """Verify ReportLab produces a valid PDF file matching standard PDF specifications."""
    builder = ResumeBuilder()
    markdown_content = """# JANE DOE
Senior Machine Learning Engineer | jane@example.com | github.com/janedoe

## Professional Summary
Accomplished Machine Learning Engineer specializing in distributed training and LLM architectures.

## Technical Skills
- Programming: Python, C++, CUDA
- Frameworks: PyTorch, FastAPI, Ray

## Professional Experience
### Apex AI | Principal ML Engineer | 2021 - Present
- Optimized transformer model inference by 40% using ONNX Runtime.
- Built automated evaluation pipeline for LLM agents.

## Education
- M.S. in Computer Science, Stanford University
"""
    output_pdf = tmp_path / "Jane_Doe_Resume.pdf"
    generated_path = builder.generate_pdf(markdown_content, output_pdf)

    assert generated_path.exists()
    assert generated_path.stat().st_size > 500

    # Validate PDF magic header
    with open(generated_path, "rb") as f:
        header = f.read(5)
    assert header == b"%PDF-"


def test_build_tailored_resume_pdf_pipeline(tmp_path, sample_resume_file):
    """Verify the entire end-to-end pipeline: load -> tailor -> generate PDF."""
    builder = ResumeBuilder(base_resume_path=sample_resume_file)

    mock_markdown = """# ALEX RIVERA
Senior AI Engineer | alex@example.com

## Professional Summary
Senior AI engineer tailoring experience for scalable Python systems.

## Professional Experience
- Led development of RAG agents with sub-50ms latency.
"""
    custom_pdf_name = tmp_path / "Tailored_Alex_Rivera.pdf"

    with patch("app.resume_builder.generate_ai_response", return_value=mock_markdown):
        result = builder.build_tailored_resume_pdf(
            job_title="Senior AI Platform Engineer",
            job_description="Seeking AI Platform Engineer with FastAPI and vector database skills.",
            company="DeepMind",
            output_filename=str(custom_pdf_name),
        )

        assert result["job_title"] == "Senior AI Platform Engineer"
        assert result["company"] == "DeepMind"
        assert "ALEX RIVERA" in result["tailored_markdown"]
        assert Path(result["pdf_path"]).exists()
        assert Path(result["pdf_path"]).stat().st_size > 500


def test_api_resume_tailor_and_pdf_endpoints():
    """Verify POST /api/resume/tailor and POST /api/resume/generate-pdf endpoints."""
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    # 1. Test /api/resume/tailor
    mock_md = "# ALEX RIVERA\n## Experience\n- Built AI systems"
    with patch("app.resume_builder.ResumeBuilder.tailor_resume", return_value=mock_md):
        res = client.post(
            "/api/resume/tailor",
            json={
                "job_title": "AI Architect",
                "job_description": "Architecting large scale Python backends with ChromaDB",
                "company": "OpenAI",
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["tailored_resume_markdown"] == mock_md

    # 2. Test /api/resume/generate-pdf
    res_pdf = client.post(
        "/api/resume/generate-pdf",
        json={
            "markdown_text": "# TEST USER\n## Summary\nExperienced developer.\n- Point 1",
            "filename": "test_download.pdf",
        },
    )
    assert res_pdf.status_code == 200
    assert res_pdf.headers["content-type"] == "application/pdf"
    assert res_pdf.content.startswith(b"%PDF-")
    # Content-Disposition should contain Yeasaleh_Resume_test_download.pdf
    assert "Yeasaleh_Resume_test_download.pdf" in res_pdf.headers["content-disposition"]

    # 3. Test GET /api/resume/download/{filename}
    res_dl = client.get("/api/resume/download/Yeasaleh_Resume_test_download.pdf")
    assert res_dl.status_code == 200
    assert res_dl.headers["content-type"] == "application/pdf"
    assert res_dl.content.startswith(b"%PDF-")


def test_yeasaleh_resume_filename_default(sample_resume_file):
    """Verify default filename starts with Yeasaleh_Resume_."""
    builder = ResumeBuilder(base_resume_path=sample_resume_file)
    with patch("app.resume_builder.generate_ai_response", return_value="# Tailored Resume"):
        result = builder.build_tailored_resume_pdf(
            job_title="Full Stack Engineer",
            job_description="React, Node.js, TypeScript",
            company="Google",
        )
        assert result["filename"] == "Yeasaleh_Resume_Google_Full_Stack_Engineer.pdf"



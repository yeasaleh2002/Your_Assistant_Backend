import os
import sys
import tempfile
from pathlib import Path
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.rag_engine import (
    FastEmbedEmbeddingFunction,
    MatchedJob,
    RAGEngine,
    MIN_MATCH_THRESHOLD,
)
from app.scraper import ScrapedJob


@pytest.fixture
def sample_resume_path(tmp_path):
    """Fixture providing a temporary resume text file."""
    resume_file = tmp_path / "resume.txt"
    resume_file.write_text(
        """
        ALEX RIVERA
        Senior Python & AI Platform Engineer | Full-Stack ML Systems
        Expert in Python, FastAPI, SQLAlchemy, ChromaDB, vector embeddings,
        RAG pipelines, microservices, and asynchronous distributed systems.
        Proven track record scaling low-latency ML applications and automated job search systems.
        """,
        encoding="utf-8",
    )
    return resume_file


@pytest.fixture
def rag_engine(tmp_path, sample_resume_path):
    """Fixture providing an isolated RAGEngine instance."""
    chroma_dir = tmp_path / "test_chroma"
    return RAGEngine(db_path=str(chroma_dir), resume_path=sample_resume_path)


def test_embedding_function_generates_dense_vectors():
    """Verify FastEmbedEmbeddingFunction generates normalized vectors of correct dimensionality (384)."""
    ef = FastEmbedEmbeddingFunction()
    docs = [
        "Senior Python Engineer with FastAPI and ChromaDB",
        "Chef preparing Mediterranean cuisine",
    ]
    embeddings = ef(docs)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert len(embeddings[1]) == 384



def test_embed_and_load_resume(rag_engine, sample_resume_path):
    """Verify loading and embedding base resume into ChromaDB."""
    text = rag_engine.embed_resume()
    assert "Senior Python & AI Platform Engineer" in text
    assert rag_engine.get_resume_text() == text

    # Verify document exists in Chroma collection
    res = rag_engine.resume_collection.get(ids=["base_resume"])
    assert len(res["ids"]) == 1
    assert "Alex Rivera" in res["documents"][0].title() or "ALEX RIVERA" in res["documents"][0]


def test_cosine_similarity_matching_and_strict_75_threshold(rag_engine):
    """
    Verify cosine similarity matching:
    - High-overlap AI/Python job scores > 75% and is returned.
    - Low-overlap job (Chef / Graphic Designer) scores < 75% and is pruned.
    """
    candidate_jobs = [
        ScrapedJob(
            title="Senior Python & AI Platform Engineer",
            company="Neural Works",
            description=(
                "We are hiring a Senior Python Engineer experienced in FastAPI, SQLAlchemy, "
                "ChromaDB, vector embeddings, and building RAG pipelines for scalable systems."
            ),
            job_link="https://neuralworks.example/jobs/1",
            career_page_link="https://neuralworks.example/careers",
        ),
        ScrapedJob(
            title="Executive Pastry Chef",
            company="Sweet Delights Bakery",
            description="Seeking a master pastry chef to bake artisan breads, croissants, and wedding cakes.",
            job_link="https://sweetdelights.example/jobs/chef",
            career_page_link="https://sweetdelights.example/careers",
        ),
        ScrapedJob(
            title="Senior Graphic Designer",
            company="Creative Studio",
            description="Design brochures, typography, print media, Figma UI mockups, and corporate branding assets.",
            job_link="https://creativestudio.example/jobs/designer",
            career_page_link="https://creativestudio.example/careers",
        ),
    ]

    # Run RAG matching pipeline with default 75% threshold
    matched = rag_engine.match_jobs(candidate_jobs, min_match_score=MIN_MATCH_THRESHOLD)

    # Only the Python/AI job should pass the > 75% threshold
    assert len(matched) == 1
    top_match = matched[0]
    assert top_match.title == "Senior Python & AI Platform Engineer"
    assert top_match.company == "Neural Works"
    assert top_match.match_score > 75.0

    # Ensure Chef and Graphic Designer were filtered out to save LLM tokens
    matched_titles = [m.title for m in matched]
    assert "Executive Pastry Chef" not in matched_titles
    assert "Senior Graphic Designer" not in matched_titles


def test_empty_candidates_returns_empty(rag_engine):
    """Verify empty input list safely returns empty results."""
    assert rag_engine.match_jobs([]) == []

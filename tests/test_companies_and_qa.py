import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database import SessionLocal
from app.models import Company
from app.scraper import is_role_relevant, get_target_roles

client = TestClient(app)

def test_company_directory_database():
    session = SessionLocal()
    try:
        total = session.query(Company).count()
        assert total >= 100, f"Expected at least 100 authentic verified companies, got {total}"
        
        # Verify zero duplicate names
        distinct_names = session.query(Company.name).distinct().count()
        assert distinct_names == total, f"Found duplicates: {total} total vs {distinct_names} distinct names"

        # Verify Arab countries present
        arab_countries = ["Saudi Arabia", "UAE", "Qatar", "Kuwait", "Bahrain", "Oman", "Egypt", "Jordan", "Morocco"]
        for country in arab_countries:
            count = session.query(Company).filter(Company.country.ilike(f"%{country}%")).count()
            assert count > 0, f"Expected verified companies in {country}, got {count}"

        # Verify Malaysia present
        my_count = session.query(Company).filter(Company.country.ilike("%Malaysia%")).count()
        assert my_count > 0, f"Expected verified companies in Malaysia, got {my_count}"

        # Verify companies founded in 2026 present
        c_2026 = session.query(Company).filter(Company.founded_year == 2026).count()
        assert c_2026 > 0, f"Expected companies founded in 2026, got {c_2026}"
    finally:
        session.close()

def test_company_api_search_by_country():
    # Test Saudi Arabia
    resp = client.get("/api/companies?country=Saudi%20Arabia&limit=20")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] > 0
    assert len(data["items"]) > 0
    for item in data["items"]:
        assert "saudi arabia" in item["country"].lower()

    # Test Malaysia
    resp = client.get("/api/companies?country=Malaysia&limit=20")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] > 0
    assert len(data["items"]) > 0
    for item in data["items"]:
        assert "malaysia" in item["country"].lower()

def test_company_api_search_by_year_2026():
    resp = client.get("/api/companies?year=2026&limit=25")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] > 0
    assert len(data["items"]) > 0
    for item in data["items"]:
        assert item["founded_year"] == 2026

def test_company_api_pagination():
    resp = client.get("/api/companies?page=1&limit=15")
    assert resp.status_code == 200
    data = resp.json()
    assert data["page"] == 1
    assert data["limit"] == 15
    assert len(data["items"]) == 15
    assert data["total_pages"] > 1

def test_qa_roles_allowed_in_scraper():
    # QA roles must be permitted
    assert is_role_relevant("Senior QA Engineer", "Automated testing with Cypress and Playwright") is True
    assert is_role_relevant("SDET", "Building end-to-end automation test suites") is True
    assert is_role_relevant("Quality Assurance Automation Engineer", "Selenium test framework") is True
    assert is_role_relevant("QA Tester", "Manual and automated web testing") is True
    assert is_role_relevant("Software Test Engineer", "Writing tests with Jest and Playwright") is True

    # Python roles still require FastAPI
    assert is_role_relevant("Python Backend Developer", "FastAPI and PostgreSQL") is True
    assert is_role_relevant("Python Developer", "Django and Flask only, microservices") is False

    # Target roles include QA
    target_roles = get_target_roles()
    assert any("qa" in r.lower() or "sdet" in r.lower() for r in target_roles)

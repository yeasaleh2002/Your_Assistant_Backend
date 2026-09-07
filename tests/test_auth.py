import os
import sys
from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.auth import create_access_token, decode_access_token, verify_admin_credentials
from app.main import app

client = TestClient(app)


def test_verify_admin_credentials():
    """Verify that verify_admin_credentials checks against environment values."""
    # Matches ADMIN_EMAIL and ADMIN_PASSWORD configured in .env
    valid = verify_admin_credentials("admin@yourassistant.com", "change_this_password")
    assert valid is True

    # Case insensitive email
    valid_caps = verify_admin_credentials("ADMIN@yourassistant.com", "change_this_password")
    assert valid_caps is True

    # Bad password
    invalid = verify_admin_credentials("admin@yourassistant.com", "wrong_password")
    assert invalid is False

    # Bad email
    invalid_email = verify_admin_credentials("other@example.com", "change_this_password")
    assert invalid_email is False


def test_jwt_token_encode_decode():
    """Verify JWT token encoding and decoding."""
    token = create_access_token({"sub": "admin@yourassistant.com", "role": "admin"})
    assert isinstance(token, str)

    payload = decode_access_token(token)
    assert payload["sub"] == "admin@yourassistant.com"
    assert payload["role"] == "admin"
    assert payload["iss"] == "your-assistant-auth"
    assert "exp" in payload


def test_login_endpoint_success_and_failure():
    """Verify POST /api/auth/login behavior."""
    # 1. Successful login
    success_payload = {
        "email": "admin@yourassistant.com",
        "password": "change_this_password",
    }
    res = client.post("/api/auth/login", json=success_payload)
    assert res.status_code == 200, res.text
    data = res.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["email"] == "admin@yourassistant.com"
    token = data["access_token"]

    # 2. Failed login (bad password)
    fail_payload = {
        "email": "admin@yourassistant.com",
        "password": "incorrect_password",
    }
    fail_res = client.post("/api/auth/login", json=fail_payload)
    assert fail_res.status_code == 401
    assert "Invalid email or password" in fail_res.json()["detail"]

    # 3. Test /api/auth/me with valid Bearer token
    me_res = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_res.status_code == 200
    me_data = me_res.json()
    assert me_data["email"] == "admin@yourassistant.com"
    assert me_data["role"] == "admin"
    assert me_data["authenticated"] is True

    # 4. Test /api/auth/me with missing/invalid token
    no_auth_res = client.get("/api/auth/me")
    assert no_auth_res.status_code in [401, 403]

    bad_token_res = client.get("/api/auth/me", headers={"Authorization": "Bearer invalid.fake.token"})
    assert bad_token_res.status_code == 401

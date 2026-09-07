import os
import sys
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.llm_manager import (
    AllProvidersExhaustedError,
    LLMManager,
    RateLimitExhaustedError,
    generate_ai_response,
)


def test_key_loading_from_env_vars(monkeypatch):
    """Verify loading and parsing comma-separated keys from environment variables."""
    monkeypatch.setenv("CLAUDE_KEYS", "c_key1, c_key2 , c_key3")
    monkeypatch.setenv("GEMINI_KEYS", "g_key1, g_key2")

    manager = LLMManager()
    assert manager.claude_keys == ["c_key1", "c_key2", "c_key3"]
    assert manager.gemini_keys == ["g_key1", "g_key2"]


def test_fallback_to_claude_key_2_when_key_1_rate_limited():
    """Verify that when Claude Key 1 is rate-limited (HTTP 429), it rotates to Key 2."""
    manager = LLMManager(
        claude_keys=["claude_key_1", "claude_key_2"],
        gemini_keys=["gemini_key_1"],
    )

    def mock_call_claude(api_key, prompt, system_prompt=None):
        if api_key == "claude_key_1":
            raise RateLimitExhaustedError("Key 1 rate-limited (429)")
        if api_key == "claude_key_2":
            return "Response from Claude Key 2"
        raise ValueError(f"Unexpected key: {api_key}")

    with patch.object(manager, "_call_claude", side_effect=mock_call_claude):
        result = manager.generate_ai_response("Help me draft a cover letter")

    assert result == "Response from Claude Key 2"


def test_fallback_to_gemini_when_all_claude_keys_fail():
    """Verify that when all Claude keys fail, the pipeline falls back to Gemini 1.5."""
    manager = LLMManager(
        claude_keys=["claude_key_1", "claude_key_2"],
        gemini_keys=["gemini_key_1"],
    )

    # All Claude keys fail
    with patch.object(manager, "_call_claude", side_effect=RateLimitExhaustedError("All Claude 429")):
        with patch.object(manager, "_call_gemini", return_value="Response from Gemini 1.5"):
            result = manager.generate_ai_response("Analyze job description")

    assert result == "Response from Gemini 1.5"


def test_gemini_key_rotation_when_gemini_key_1_fails():
    """Verify that Gemini rotates keys if Gemini Key 1 encounters rate limits."""
    manager = LLMManager(
        claude_keys=[],  # No Claude keys configured
        gemini_keys=["gemini_key_1", "gemini_key_2"],
    )

    def mock_call_gemini(api_key, prompt, system_prompt=None):
        if api_key == "gemini_key_1":
            raise RateLimitExhaustedError("Gemini Key 1 quota exhausted (429)")
        if api_key == "gemini_key_2":
            return "Response from Gemini Key 2"
        raise ValueError(f"Unexpected key: {api_key}")

    with patch.object(manager, "_call_gemini", side_effect=mock_call_gemini):
        result = manager.generate_ai_response("Extract required skills")

    assert result == "Response from Gemini Key 2"


def test_all_providers_exhausted_raises_error():
    """Verify that AllProvidersExhaustedError is raised if all Claude and Gemini keys fail."""
    manager = LLMManager(
        claude_keys=["c1"],
        gemini_keys=["g1"],
    )

    with patch.object(manager, "_call_claude", side_effect=Exception("Claude network error")):
        with patch.object(manager, "_call_gemini", side_effect=RateLimitExhaustedError("Gemini 429")):
            with pytest.raises(AllProvidersExhaustedError) as exc_info:
                manager.generate_ai_response("Test prompt")

    assert "All AI LLM providers and keys failed" in str(exc_info.value)


def test_unified_generate_ai_response_function():
    """Verify public generate_ai_response function abstraction."""
    mock_mgr = MagicMock()
    mock_mgr.generate_ai_response.return_value = "Unified AI Output"

    res = generate_ai_response("Draft email", system_prompt="Be concise", manager=mock_mgr)
    assert res == "Unified AI Output"
    mock_mgr.generate_ai_response.assert_called_once_with("Draft email", system_prompt="Be concise")


def test_api_generate_ai_endpoint():
    """Verify POST /api/ai/generate endpoint."""
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    with patch("app.main.generate_ai_response", return_value="Drafted Cover Letter"):
        response = client.post(
            "/api/ai/generate",
            json={"prompt": "Draft a cover letter for AI Engineer position", "system_prompt": "Professional tone"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["response"] == "Drafted Cover Letter"


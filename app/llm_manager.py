import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import requests

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger("your_assistant.llm_manager")

# API Endpoints
ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_CLAUDE_MODEL = "claude-3-5-sonnet-20241022"

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"


# ==============================================================================
# Custom Exceptions
# ==============================================================================

class LLMManagerError(Exception):
    """Base exception for LLM manager failures."""
    pass


class RateLimitExhaustedError(LLMManagerError):
    """Raised when an individual API key encounters HTTP 429 or quota exhaustion."""
    pass


class ProviderAuthError(LLMManagerError):
    """Raised when an API key is invalid or unauthorized."""
    pass


class AllProvidersExhaustedError(LLMManagerError):
    """Raised when every key across Claude 3 and Gemini 1.5 providers has failed."""
    pass


# ==============================================================================
# LLM Manager Core
# ==============================================================================

class LLMManager:
    """
    Resilient multi-tier LLM fallback manager:
    1. Sequentially rotates Claude 3 keys on rate-limits (HTTP 429 / 529).
    2. Seamlessly falls back to Gemini 1.5 when all Claude keys are exhausted.
    3. Sequentially rotates Gemini keys if needed.
    """

    def __init__(
        self,
        claude_keys: Optional[List[str]] = None,
        gemini_keys: Optional[List[str]] = None,
        claude_model: str = DEFAULT_CLAUDE_MODEL,
        gemini_model: str = DEFAULT_GEMINI_MODEL,
        timeout: int = 30,
    ):
        self.claude_keys = claude_keys if claude_keys is not None else self._load_keys("CLAUDE_KEYS", "ANTHROPIC_API_KEY")
        self.gemini_keys = gemini_keys if gemini_keys is not None else self._load_keys("GEMINI_KEYS", "GEMINI_API_KEY", "GOOGLE_API_KEY")
        self.claude_model = claude_model
        self.gemini_model = gemini_model
        self.timeout = timeout

        logger.info(
            "LLMManager initialized with %d Claude key(s) and %d Gemini key(s).",
            len(self.claude_keys),
            len(self.gemini_keys),
        )

    @staticmethod
    def _load_keys(primary_env: str, *fallback_envs: str) -> List[str]:
        """
        Parse comma-separated keys from environment variables.
        Supports:
          PRIMARY_ENV=key1,key2,key3
        or single fallback variables (e.g. ANTHROPIC_API_KEY).
        """
        raw = os.getenv(primary_env, "").strip()
        if raw:
            return [k.strip() for k in raw.split(",") if k.strip()]

        for fallback in fallback_envs:
            fb_val = os.getenv(fallback, "").strip()
            if fb_val:
                return [k.strip() for k in fb_val.split(",") if k.strip()]

        return []

    def _call_claude(
        self,
        api_key: str,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Execute request to Anthropic Claude 3 Messages API."""
        headers = {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": self.claude_model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            payload["system"] = system_prompt

        response = requests.post(
            ANTHROPIC_MESSAGES_URL,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )

        # 429 = Rate Limited, 529 = Overloaded
        if response.status_code in (429, 529):
            raise RateLimitExhaustedError(
                f"Claude rate-limited (HTTP {response.status_code}): {response.text}"
            )

        if response.status_code in (401, 403):
            raise ProviderAuthError(f"Claude authentication failed (HTTP {response.status_code})")

        if not response.ok:
            # Check for rate_limit_error in JSON response body
            try:
                err_data = response.json()
                if err_data.get("error", {}).get("type") == "rate_limit_error":
                    raise RateLimitExhaustedError(f"Claude rate limit error: {response.text}")
            except Exception:
                pass
            raise LLMManagerError(f"Claude error (HTTP {response.status_code}): {response.text}")

        data = response.json()
        content = data.get("content", [])
        if content and isinstance(content, list) and content[0].get("text"):
            return content[0]["text"].strip()

        raise LLMManagerError(f"Malformed Claude response format: {data}")

    def _call_gemini(
        self,
        api_key: str,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Execute request to Google Gemini 1.5 generateContent API."""
        url = f"{GEMINI_BASE_URL}/{self.gemini_model}:generateContent?key={api_key}"
        headers = {"content-type": "application/json"}

        contents = []
        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": f"System Instructions: {system_prompt}"}]})
            contents.append({"role": "model", "parts": [{"text": "Understood. I will follow these instructions."}]})

        contents.append({"role": "user", "parts": [{"text": prompt}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 1024,
            },
        }

        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)

        if response.status_code == 429:
            raise RateLimitExhaustedError(f"Gemini rate-limited (HTTP 429): {response.text}")

        if response.status_code in (400, 403):
            # Inspect if 400 is actually quota exhaustion (e.g. RESOURCE_EXHAUSTED)
            if "RESOURCE_EXHAUSTED" in response.text or "quota" in response.text.lower():
                raise RateLimitExhaustedError(f"Gemini quota exhausted: {response.text}")
            raise ProviderAuthError(f"Gemini auth/request error (HTTP {response.status_code})")

        if not response.ok:
            raise LLMManagerError(f"Gemini error (HTTP {response.status_code}): {response.text}")

        data = response.json()
        candidates = data.get("candidates", [])
        if candidates and isinstance(candidates, list):
            first_cand = candidates[0]
            parts = first_cand.get("content", {}).get("parts", [])
            if parts and isinstance(parts, list) and parts[0].get("text"):
                return parts[0]["text"].strip()

        raise LLMManagerError(f"Malformed Gemini response format: {data}")

    def generate_ai_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Unified generation pipeline with multi-tier fallback:
        Tier 1: Claude 3 with Key 1 -> If rate-limited/failed, try Key 2 -> Key N...
        Tier 2: Gemini 1.5 with Key 1 -> If rate-limited/failed, try Key 2 -> Key N...
        
        :param prompt: User instruction or input.
        :param system_prompt: Optional system personality/guidelines.
        :return: Generated text response.
        :raises AllProvidersExhaustedError: If every key and provider fails.
        """
        errors: List[str] = []

        # ======================================================================
        # Tier 1: Try Claude 3 Models
        # ======================================================================
        if self.claude_keys:
            logger.info("Attempting generation via Tier 1: Claude 3 (%s)...", self.claude_model)
            for idx, key in enumerate(self.claude_keys, start=1):
                masked_key = f"{key[:6]}...{key[-4:]}" if len(key) > 10 else "***"
                try:
                    logger.info("Calling Claude 3 with Key #%d (%s)...", idx, masked_key)
                    result = self._call_claude(key, prompt, system_prompt=system_prompt)
                    logger.info("Claude 3 successfully generated response using Key #%d.", idx)
                    return result
                except RateLimitExhaustedError as rle:
                    msg = f"Claude Key #{idx} rate-limited: {rle}"
                    logger.warning(msg)
                    errors.append(msg)
                except Exception as exc:
                    msg = f"Claude Key #{idx} failed: {exc}"
                    logger.warning(msg)
                    errors.append(msg)
        else:
            logger.info("No Claude API keys configured. Proceeding to Gemini fallback.")

        # ======================================================================
        # Tier 2: Fallback to Gemini 1.5 Models
        # ======================================================================
        if self.gemini_keys:
            logger.warning("All Claude options exhausted. Falling back to Tier 2: Gemini 1.5 (%s)...", self.gemini_model)
            for idx, key in enumerate(self.gemini_keys, start=1):
                masked_key = f"{key[:6]}...{key[-4:]}" if len(key) > 10 else "***"
                try:
                    logger.info("Calling Gemini 1.5 with Key #%d (%s)...", idx, masked_key)
                    result = self._call_gemini(key, prompt, system_prompt=system_prompt)
                    logger.info("Gemini 1.5 successfully generated response using Key #%d.", idx)
                    return result
                except RateLimitExhaustedError as rle:
                    msg = f"Gemini Key #{idx} rate-limited: {rle}"
                    logger.warning(msg)
                    errors.append(msg)
                except Exception as exc:
                    msg = f"Gemini Key #{idx} failed: {exc}"
                    logger.warning(msg)
                    errors.append(msg)
        else:
            logger.warning("No Gemini API keys configured.")

        # ======================================================================
        # Complete Failure Across All Providers
        # ======================================================================
        summary = "\n".join(f"- {err}" for err in errors)
        error_msg = (
            f"All AI LLM providers and keys failed to generate a response.\n"
            f"Attempted {len(self.claude_keys)} Claude key(s) and {len(self.gemini_keys)} Gemini key(s).\n"
            f"Error details:\n{summary if summary else 'No API keys configured for either Claude or Gemini.'}"
        )
        logger.error(error_msg)
        raise AllProvidersExhaustedError(error_msg)


# ==============================================================================
# Global Unified Function
# ==============================================================================

_global_manager: Optional[LLMManager] = None


def generate_ai_response(
    prompt: str,
    system_prompt: Optional[str] = None,
    manager: Optional[LLMManager] = None,
) -> str:
    """
    Unified public function to generate AI responses using the resilient fallback system.
    Tries Claude 3 (Key 1 -> Key 2) -> Gemini 1.5 (Key 1 -> Key 2).
    """
    global _global_manager
    if manager is None:
        if _global_manager is None:
            _global_manager = LLMManager()
        manager = _global_manager

    return manager.generate_ai_response(prompt, system_prompt=system_prompt)

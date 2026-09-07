from datetime import datetime, timedelta, timezone
import logging
import os
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt
from pydantic import BaseModel, ConfigDict, EmailStr, Field

logger = logging.getLogger("your_assistant.auth")

# ==============================================================================
# Auth Configuration
# ==============================================================================

JWT_SECRET = os.getenv(
    "JWT_SECRET",
    "080c28897f0a23ca02c407685998fc1c6e8197d3997510ba1d49d7f994f628ad",
)
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_DAYS = int(os.getenv("JWT_EXPIRE_DAYS", "7"))

security_bearer = HTTPBearer(auto_error=False)
security_bearer_strict = HTTPBearer(auto_error=True)


# ==============================================================================
# Pydantic Schemas
# ==============================================================================

class LoginRequest(BaseModel):
    """Payload model for admin authentication."""
    model_config = ConfigDict(str_strip_whitespace=True)

    email: EmailStr = Field(..., description="Admin email address", examples=["user@example.com"])
    password: str = Field(..., min_length=1, description="Admin password", examples=["secret123"])


class TokenResponse(BaseModel):
    """Response model returning signed JWT access token and user metadata."""
    access_token: str
    token_type: str = "bearer"
    expires_in_days: int = ACCESS_TOKEN_EXPIRE_DAYS
    user: Dict[str, Any]


class UserProfile(BaseModel):
    """Authenticated user info."""
    email: str
    role: str = "admin"
    authenticated: bool = True


# ==============================================================================
# Authentication & JWT Utilities
# ==============================================================================

def get_admin_credentials() -> Dict[str, str]:
    """Retrieve authorized admin credentials from environment."""
    return {
        "email": os.getenv("ADMIN_EMAIL", "").strip().lower(),
        "password": os.getenv("ADMIN_PASSWORD", "").strip(),
    }


def verify_admin_credentials(email: str, password: str) -> bool:
    """
    Verify user input against ADMIN_EMAIL and ADMIN_PASSWORD configured in .env.
    Performs constant-time comparison for security.
    """
    creds = get_admin_credentials()
    admin_email = creds["email"]
    admin_password = creds["password"]

    if not admin_email or not admin_password:
        logger.warning("ADMIN_EMAIL or ADMIN_PASSWORD is not set in .env. Login rejected.")
        return False

    email_clean = email.strip().lower()
    password_clean = password.strip()

    import hmac
    email_match = hmac.compare_digest(email_clean, admin_email)
    password_match = hmac.compare_digest(password_clean, admin_password)

    return email_match and password_match


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Generate a signed JWT token containing user identity and expiration.
    """
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS))
    to_encode.update({
        "iat": now,
        "exp": expire,
        "iss": "your-assistant-auth",
    })
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate a signed JWT token.
    Raises HTTPException 401 if invalid or expired.
    """
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            issuer="your-assistant-auth",
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_bearer),
) -> Optional[Dict[str, Any]]:
    """
    FastAPI dependency that extracts and verifies optional Bearer token.
    """
    if not credentials or not credentials.credentials:
        return None
    return decode_access_token(credentials.credentials)


async def require_authenticated_user(
    credentials: HTTPAuthorizationCredentials = Security(security_bearer_strict),
) -> Dict[str, Any]:
    """
    FastAPI dependency that enforces a valid Bearer token.
    Rejects unauthorized requests with HTTP 401.
    """
    return decode_access_token(credentials.credentials)

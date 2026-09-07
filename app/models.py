from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, HttpUrl, field_validator
from sqlalchemy import Column, DateTime, Float, Integer, String, Text

from app.database import Base


# ==============================================================================
# SQLAlchemy ORM Models
# ==============================================================================

class JobHistory(Base):
    """
    SQLAlchemy ORM table tracking AI job search history and recruiter matches.
    """
    __tablename__ = "job_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    title = Column(String(255), nullable=False, index=True)
    company = Column(String(255), nullable=False, index=True)
    job_link = Column(Text, nullable=False)
    career_page_link = Column(Text, nullable=True)
    match_score = Column(Float, nullable=False)
    recruiter_email = Column(String(255), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )

    def __repr__(self) -> str:
        return f"<JobHistory(id={self.id}, title='{self.title}', company='{self.company}', score={self.match_score})>"


# ==============================================================================
# Pydantic Schemas (Strict Input Validation & Security)
# ==============================================================================

class JobHistoryBase(BaseModel):
    """Base Pydantic model with strict validation to prevent malicious payloads."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",  # Reject unexpected fields to mitigate mass assignment & injection
    )

    title: str = Field(
        ...,
        min_length=2,
        max_length=255,
        description="Job title",
        examples=["Senior AI Platform Engineer"],
    )
    company: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Target company name",
        examples=["DeepMind"],
    )
    job_link: HttpUrl = Field(
        ...,
        description="Direct link to the job posting (http/https only)",
        examples=["https://example.com/careers/ai-engineer"],
    )
    career_page_link: Optional[HttpUrl] = Field(
        default=None,
        description="Company career portal link",
        examples=["https://example.com/careers"],
    )
    match_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Match score calculated by AI (between 0.0 and 100.0)",
        examples=[94.5],
    )
    recruiter_email: Optional[EmailStr] = Field(
        default=None,
        description="Optional verified recruiter contact email",
        examples=["recruiter@example.com"],
    )

    @field_validator("title", "company")
    @classmethod
    def sanitize_text(cls, value: str) -> str:
        """Prevent control characters, script tags, or dangerous characters in strings."""
        if "<script" in value.lower() or "</script>" in value.lower():
            raise ValueError("Malicious content detected in text field.")
        return value


class JobHistoryCreate(JobHistoryBase):
    """Payload model for creating a new job history record."""
    pass


class JobHistoryResponse(BaseModel):
    """Response model for returning serialized job history records."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    company: str
    job_link: str
    career_page_link: Optional[str] = None
    match_score: float
    recruiter_email: Optional[str] = None
    created_at: datetime

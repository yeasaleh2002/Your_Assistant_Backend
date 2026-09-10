from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator
from sqlalchemy import Column, Date, DateTime, Enum as SQLEnum, Float, Integer, String, Text
from sqlalchemy.ext.hybrid import hybrid_property

from app.database import Base


# ==============================================================================
# Application Status Enum
# ==============================================================================

class JobStatus(str, Enum):
    Pending = "Pending"
    Applied = "Applied"
    Interview = "Interview"
    Rejected = "Rejected"


# ==============================================================================
# SQLAlchemy ORM Models
# ==============================================================================

class Job(Base):
    """
    SQLAlchemy ORM model for storing and tracking job opportunities.
    Supports PostgreSQL, MySQL, and SQLite dialects.
    """
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    title = Column(String(255), nullable=False, index=True)
    company = Column(String(255), nullable=False, index=True)
    link = Column(Text, nullable=False, index=True)
    match_score = Column(Float, nullable=False, index=True)
    location = Column(String(255), nullable=False, default="Remote")
    status = Column(
        SQLEnum(JobStatus, native_enum=False),
        nullable=False,
        default=JobStatus.Pending,
        index=True,
    )
    scraped_date = Column(
        Date,
        nullable=False,
        default=date.today,
        index=True,
    )

    # Supplemental fields for ATS Resume & Cold Email generation
    description = Column(Text, nullable=True)
    recruiter_email = Column(String(255), nullable=True)
    career_page_link = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )

    # Backward-compatible property so references to job.job_link seamlessly map to job.link
    @hybrid_property
    def job_link(self) -> Any:
        return getattr(self, "link")

    @job_link.setter
    def job_link(self, value: Any) -> None:
        setattr(self, "link", value)

    def __repr__(self) -> str:
        return f"<Job(id={self.id}, title='{self.title}', company='{self.company}', status='{self.status}', score={self.match_score}, date='{self.scraped_date}')>"


# Backward compatibility alias
JobHistory = Job


# ==============================================================================
# Pydantic Schemas (Strict Input Validation & Security)
# ==============================================================================

class JobStatusEnum(str, Enum):
    Pending = "Pending"
    Applied = "Applied"
    Interview = "Interview"
    Rejected = "Rejected"


class JobUpdateStatus(BaseModel):
    """Payload model for updating application status."""
    status: JobStatusEnum = Field(
        ...,
        description="Updated application status: Pending, Applied, Interview, or Rejected",
        examples=["Applied"],
    )


class JobBase(BaseModel):
    """Base Pydantic model with strict validation to prevent malicious payloads."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    title: str = Field(
        ...,
        min_length=2,
        max_length=255,
        description="Job title",
        examples=["Senior Full Stack Engineer"],
    )
    company: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Target company name",
        examples=["Vercel Ecosystem Partner"],
    )
    link: Optional[str] = Field(
        default=None,
        min_length=5,
        description="Direct link to the job posting",
        examples=["https://example.com/careers/engineer"],
    )
    job_link: Optional[str] = Field(
        default=None,
        min_length=5,
        description="Direct link to the job posting (backward-compatible alias)",
        examples=["https://example.com/careers/engineer"],
    )
    match_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Match score calculated by AI (between 0.0 and 100.0)",
        examples=[88.5],
    )
    location: str = Field(
        default="Remote",
        max_length=255,
        description="Job location or Remote status",
        examples=["Remote"],
    )
    status: JobStatusEnum = Field(
        default=JobStatusEnum.Pending,
        description="Application status: Pending, Applied, Interview, Rejected",
        examples=["Pending"],
    )
    scraped_date: Optional[date] = Field(
        default_factory=date.today,
        description="Date the job was scraped (YYYY-MM-DD)",
        examples=["2026-09-07"],
    )
    description: Optional[str] = Field(default=None, description="Job description text")
    career_page_link: Optional[str] = Field(default=None, description="Company career portal link")
    recruiter_email: Optional[EmailStr] = Field(default=None, description="Recruiter contact email")

    @model_validator(mode="before")
    @classmethod
    def resolve_link_alias(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "job_link" in data and "link" not in data:
                data["link"] = data["job_link"]
            elif "link" in data and "job_link" not in data:
                data["job_link"] = data["link"]
            if not data.get("link") and not data.get("job_link"):
                raise ValueError("Either 'link' or 'job_link' must be provided.")
        return data

    @field_validator("link", "job_link")
    @classmethod
    def validate_url(cls, value: Optional[str]) -> Optional[str]:
        """Validate URL scheme."""
        if value is not None:
            if not (value.startswith("http://") or value.startswith("https://")):
                raise ValueError("Link must be a valid HTTP or HTTPS URL.")
        return value

    @field_validator("title", "company")
    @classmethod
    def sanitize_text(cls, value: str) -> str:
        """Prevent control characters or script tags in strings."""
        if "<script" in value.lower() or "</script>" in value.lower():
            raise ValueError("Malicious content detected in text field.")
        return value


class JobCreate(JobBase):
    """Payload model for programmatic job creation."""
    pass


class JobResponse(BaseModel):
    """Response model for returning serialized job records."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    company: str
    link: str
    match_score: float
    location: str
    status: JobStatusEnum
    scraped_date: date
    description: Optional[str] = None
    recruiter_email: Optional[str] = None
    career_page_link: Optional[str] = None
    created_at: Optional[datetime] = None

    # Backward-compatible attribute for legacy consumers expecting job_link
    @property
    def job_link(self) -> str:
        return self.link


# Backward compatibility aliases
JobHistoryBase = JobBase
JobHistoryCreate = JobCreate
JobHistoryResponse = JobResponse


# ==============================================================================
# Software Startup & Remote Tech Companies Directory
# ==============================================================================

class Company(Base):
    """
    SQLAlchemy ORM model for storing curated global software startups and remote tech companies.
    Enforces uniqueness on company name and website to prevent duplicate entries.
    """
    __tablename__ = "companies"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(255), nullable=False, unique=True, index=True)
    website = Column(String(255), nullable=False)
    careers_page = Column(Text, nullable=True)
    contact_email = Column(String(255), nullable=True)
    country = Column(String(100), nullable=False, index=True)
    city = Column(String(100), nullable=True)
    founded_year = Column(Integer, nullable=False, index=True)
    industry = Column(String(100), default="Software & Technology", nullable=False)
    tech_stack = Column(Text, nullable=True)
    remote_policy = Column(String(50), default="Remote-First", nullable=False)
    employee_count = Column(String(50), nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )

    def __repr__(self) -> str:
        return f"<Company(id={self.id}, name='{self.name}', country='{self.country}', founded={self.founded_year})>"


class CompanyBase(BaseModel):
    """Base Pydantic model for software startup company."""
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(..., min_length=1, max_length=255, description="Company name")
    website: str = Field(..., description="Company official website URL")
    careers_page: Optional[str] = Field(default=None, description="Direct link to careers or job portal")
    contact_email: Optional[str] = Field(default=None, description="Contact or HR/recruiting email")
    country: str = Field(..., description="Country of origin / HQ (e.g. USA, Canada, Singapore, UK)")
    city: Optional[str] = Field(default=None, description="Headquarters city")
    founded_year: int = Field(..., ge=1990, le=2026, description="Year company was founded")
    industry: str = Field(default="Software & Technology", description="Industry sector")
    tech_stack: Optional[str] = Field(default=None, description="Primary technologies & frameworks")
    remote_policy: str = Field(default="Remote-First", description="Remote working policy")
    employee_count: Optional[str] = Field(default=None, description="Estimated team size range")
    description: Optional[str] = Field(default=None, description="Company overview and product mission")


class CompanyCreate(CompanyBase):
    pass


class CompanyResponse(CompanyBase):
    """Response model for company record."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: Optional[datetime] = None


class CompanyPaginationResponse(BaseModel):
    """Paginated list of software companies."""
    total: int = Field(..., description="Total companies matching query")
    page: int = Field(..., description="Current page number (1-indexed)")
    limit: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total pages available")
    items: List[CompanyResponse] = Field(..., description="List of companies on current page")

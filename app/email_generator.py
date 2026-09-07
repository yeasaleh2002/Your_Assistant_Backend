import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from app.llm_manager import generate_ai_response

logger = logging.getLogger("your_assistant.email_generator")

DEFAULT_RESUME_FILE = Path("data") / "resume.txt"
DEFAULT_MISSING_EMAIL = "[Recruiter Email]"

# High-precision regex pattern for extracting email addresses from text
EMAIL_REGEX = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"
)

# Common non-recruiter domains or false positive suffixes to ignore
IGNORED_PATTERNS = {
    "example.com",
    "sentry.io",
    "w3.org",
    "schema.org",
    "domain.com",
}


class EmailDraft(BaseModel):
    """Structured JSON schema for generated cold outreach email."""
    model_config = ConfigDict(str_strip_whitespace=True)

    email: str = Field(
        ...,
        description="Recruiter's contact email or [Recruiter Email] if not found",
    )
    subject: str = Field(
        ...,
        description="Engaging, high-open-rate subject line tailored to the job role",
    )
    body: str = Field(
        ...,
        description="Personalized cold email body emphasizing genuine skills from the resume",
    )


def extract_recruiter_email(text: str) -> str:
    """
    Extract a recruiter or hiring contact email from the job description using regex.
    Returns '[Recruiter Email]' if no valid contact email is discovered.
    """
    if not text:
        return DEFAULT_MISSING_EMAIL

    matches = EMAIL_REGEX.findall(text)
    for email_match in matches:
        cleaned = email_match.strip().lower()
        domain = cleaned.split("@")[-1]
        if domain not in IGNORED_PATTERNS and not cleaned.endswith((".png", ".jpg", ".gif")):
            return email_match.strip()

    return DEFAULT_MISSING_EMAIL


class EmailGenerator:
    """
    Service generating highly personalized, ATS/recruiter-focused cold outreach
    emails highlighting genuine skills from the user's base resume.
    """

    def __init__(self, base_resume_path: Union[str, Path] = DEFAULT_RESUME_FILE):
        self.base_resume_path = Path(base_resume_path)

    def load_base_resume(self) -> str:
        """Load text from the base resume file."""
        if not self.base_resume_path.exists():
            raise FileNotFoundError(f"Base resume not found at: {self.base_resume_path}")
        return self.base_resume_path.read_text(encoding="utf-8").strip()

    def generate_cold_email(
        self,
        job_title: str,
        job_description: str,
        company: Optional[str] = None,
        recruiter_email: Optional[str] = None,
        base_resume_text: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Generate a tailored cold email for a specific job posting.
        
        1. Resolves recruiter email (provided -> regex extracted -> [Recruiter Email]).
        2. Prompts LLM to write an engaging subject line and concise body emphasizing
           real skills from the user's resume matching the role.
        3. Returns JSON dict: {"email": "...", "subject": "...", "body": "..."}.
        """
        resume_content = base_resume_text or self.load_base_resume()

        # Step 1: Resolve recruiter email
        resolved_email = recruiter_email or extract_recruiter_email(job_description)
        company_name = company or "the team"

        # Step 2: Formulate AI Prompt
        system_prompt = (
            "You are an elite career strategist and executive talent partner. "
            "Write a highly engaging, concise, and persuasive cold outreach email from the candidate "
            "to the recruiter or hiring manager. "
            "CRITICAL RULES:\n"
            "1. You MUST ONLY reference real skills, metrics, and experiences present in the candidate's resume. "
            "DO NOT fabricate or hallucinate any skills or experiences.\n"
            "2. Highlight the 2-3 most compelling skill overlaps with the target job requirements.\n"
            "3. Keep the email punchy, professional, and under 175 words.\n"
            "4. Include an engaging, high-open-rate subject line.\n"
            "5. You MUST return ONLY a valid JSON object with keys: 'email', 'subject', 'body'."
        )

        user_prompt = f"""
TARGET ROLE:
- Job Title: {job_title}
- Company: {company_name}
- Recruiter Contact Email: {resolved_email}
- Job Description:
{job_description}

CANDIDATE BASE RESUME:
{resume_content}

OUTPUT FORMAT:
Return ONLY a valid JSON object formatted exactly as:
{{
  "email": "{resolved_email}",
  "subject": "<engaging, tailored subject line>",
  "body": "<concise, personalized email body with greeting, tailored value proposition, call-to-action, and candidate sign-off>"
}}
"""

        logger.info("Generating personalized cold email for '%s' at '%s'...", job_title, company_name)

        # Step 3: Call LLM
        raw_response = generate_ai_response(
            prompt=user_prompt.strip(),
            system_prompt=system_prompt,
        )

        # Step 4: Parse JSON from LLM response
        parsed_data = self._parse_json_response(raw_response, default_email=resolved_email, default_subject=f"Inquiry: {job_title} - {company_name}")
        return parsed_data

    @staticmethod
    def _parse_json_response(
        raw_text: str,
        default_email: str,
        default_subject: str,
    ) -> Dict[str, str]:
        """
        Extract and validate JSON object from LLM response, stripping markdown fences if present.
        """
        cleaned = raw_text.strip()

        # Remove markdown code fences ```json ... ``` if present
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        # Search for first '{' and last '}'
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)

        try:
            data = json.loads(cleaned)
            email = str(data.get("email") or default_email).strip()
            subject = str(data.get("subject") or default_subject).strip()
            body = str(data.get("body") or "").strip()

            if not body:
                body = raw_text.strip()

            draft = EmailDraft(email=email, subject=subject, body=body)
            return draft.model_dump()
        except Exception as exc:
            logger.warning("Failed to parse clean JSON from LLM response (%s). Using fallback parsing.", exc)
            return {
                "email": default_email,
                "subject": default_subject,
                "body": raw_text.strip(),
            }

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


WORLD_CLASS_EMAIL_WRITER_PROMPT = """You are a World-Class Job Application Email Writer, Professional Recruiter Communication Specialist, and Hiring Communication Expert.

Your job is to generate concise, personalized, professional, and high-converting job application emails based on the candidate's resume, the target job description, and the company information.

Your goal is to make the email relevant enough that a recruiter immediately understands:
* Who the candidate is
* What role they are applying for
* Why they are relevant
* What makes them a strong candidate
* What action the recruiter should take next

### 1. PERSONALIZATION
Never generate a generic email when a job description is available.
Analyze: Job title, Company name, Required skills, Preferred skills, Responsibilities, Experience requirements, Industry/domain, Important keywords.
Then personalize the email around the candidate's actual experience.
You MUST ONLY reference real skills and experiences present in the candidate's resume. DO NOT fabricate or hallucinate any skills, metrics, or experiences.

### 2. SUBJECT LINE
Generate a concise and professional subject line.
Prefer formats such as:
Application for [Job Title] — [Candidate Name]
or
Application — [Job Title] | [Candidate Name]
When appropriate, create a more compelling but still professional subject line based on the candidate's strongest relevant qualification.
Avoid: Clickbait, Excessive capitalization, Emojis, Generic subjects such as "Job Application".

### 3. EMAIL STRUCTURE
Keep the email concise and recruiter-friendly.
Recommended structure:
1. Professional greeting
2. Clear statement of the position being applied for
3. Short introduction of the candidate
4. 2–3 highly relevant qualifications or achievements
5. Why the candidate is relevant to the specific role/company
6. Mention attached resume when applicable
7. Clear call to action (e.g. "I'd welcome the opportunity to discuss how my experience could contribute to your team.")
8. Professional closing

The default email should generally be around 120–200 words.

### 4. IMPACT
Prioritize achievements over generic responsibilities.
Use measurable achievements when the candidate has provided legitimate metrics. Never invent metrics.
If metrics are unavailable, use strong qualitative impact.

### 5. KEYWORD RELEVANCE
Naturally incorporate important keywords from the job description when the candidate genuinely possesses those skills.
Do NOT keyword-stuff the email. The email should sound naturally written by a professional candidate, not generated for an ATS.

### 6. GRAMMAR & LANGUAGE
Check grammar, spelling, sentence structure, punctuation, professional tone, clarity, and conciseness.
Remove unnecessary phrases and filler. Avoid overly complicated vocabulary.

### 7. REPETITION
Detect and remove repeated skills, technologies, achievements, and phrases.

### 8. TONE
Default tone: Professional + Confident + Concise + Natural.
Do not sound desperate, arrogant, robotic, overly formal, generic, or AI-generated.
Avoid phrases such as:
"I am writing to express my keen interest..."
"I believe I would be a perfect fit..."
"I am extremely passionate..."
"Please find my attached resume for your kind consideration..."
Prefer natural professional language.

### 9. CALL TO ACTION
End with a natural, polite CTA.

### 10. FINAL QUALITY CHECK & OUTPUT FORMAT
Return ONLY a valid JSON object with keys: 'email', 'subject', 'body'.
"""


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
        2. Prompts LLM using World-Class Job Application Email Writer instructions.
        3. Returns JSON dict: {"email": "...", "subject": "...", "body": "..."}.
        """
        resume_content = base_resume_text or self.load_base_resume()

        # Step 1: Resolve recruiter email
        resolved_email = recruiter_email or extract_recruiter_email(job_description)
        company_name = company or "the team"

        # Step 2: Formulate AI Prompt
        system_prompt = WORLD_CLASS_EMAIL_WRITER_PROMPT

        user_prompt = f"""
TARGET ROLE:
- Job Title: {job_title}
- Company: {company_name}
- Recruiter Contact Email: {resolved_email}
- Job Description:
{job_description}

CANDIDATE BASE RESUME:
{resume_content}

CRITICAL RULES:
1. Candidate Name: Extract the candidate's name from the resume (e.g. Yeasaleh).
2. Professional Subject Line: Prefer format like 'Application for {job_title} — Yeasaleh' or highlight a top relevant qualification.
3. Word Count: Keep between 120 and 200 words.
4. Tone: Professional + Confident + Concise + Natural. Avoid clichés.
5. Truthfulness: Strictly highlight real qualifications and achievements from the resume without fabricating metrics.

OUTPUT FORMAT:
Return ONLY a valid JSON object formatted exactly as:
{{
  "email": "{resolved_email}",
  "subject": "<engaging, professional subject line>",
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

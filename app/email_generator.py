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


WORLD_CLASS_EMAIL_WRITER_PROMPT = """You are a World-Class Software Engineer, Technical Recruiter, and Elite Direct-Outreach Specialist.

Your job is to write high-converting, deeply humanized cold emails from software engineer Yeasaleh directly to hiring managers, engineering leads, and technical recruiters.

CRITICAL DIRECTIVE: IT MUST SOUND LIKE A REAL, TALENTED HUMAN ENGINEER WRITING A THOUGHTFUL DIRECT EMAIL—NEVER AN AUTOMATED API SCRIPT, AI BOT, OR GENERIC COVER LETTER TEMPLATE.
You MUST ONLY reference real skills and experiences present in the candidate's resume. DO NOT fabricate or hallucinate any skills, metrics, or experiences.

### 1. BANNED PHRASES & AI CLICHÉS (STRICTLY PROHIBITED)
Never use:
- "I am writing to express my keen interest / enthusiastic application..."
- "I believe I am an ideal / perfect fit for this position..."
- "I am extremely passionate about..."
- "Look no further..."
- "Please find attached my resume for your perusal / consideration..."
- "In conclusion..."
- "Hope this email finds you well" (Use a warmer, more direct opening instead)

### 2. HUMANIZED STRUCTURE & FLOW
1. NATURAL OPENING:
   - "Hi [Hiring Team / Engineering Team],"
   - Open with immediate context: Mention seeing their opening for [Job Title] at [Company Name] and noticing their work on [specific product focus or technology requirement from job description].
2. DIRECT RELEVANCE & VALUE PROPOSITION:
   - In 1 conversational sentence, bridge why Yeasaleh's actual engineering experience directly solves problems in their stack (Next.js, React, Node.js, TypeScript, FastAPI, or cloud systems).
3. 2-3 PUNCHY, QUANTIFIED PROOF POINTS (Use bullets for readability):
   - Highlight 2-3 concrete achievements directly from Yeasaleh's verified resume:
     * e.g., Next.js & React: Screen scaling utilities eliminating layout shifts (0.01 CLS) and 60fps micro-animations.
     * e.g., Node.js & Database: APIs and MySQL/Postgres architectures scaling across 50,000+ active users with 99.9% uptime.
     * e.g., Payments & Automation: Multi-tier subscription billing with Stripe/LemonSqueezy or AI pipeline automations cutting manual workloads by 85%.
   - Only include metrics and technologies that are relevant to this specific role and genuinely present in the resume.
4. PORTFOLIO & CODE PROOF:
   - Provide direct links to review live engineering work:
     "You can also explore my live project work and architectural write-ups at https://yeasaleh.xyz and GitHub at https://github.com/yeasaleh2002."
5. LOW-FRICTION, CONFIDENT CALL TO ACTION:
   - "I've attached my tailored resume. Would you be open to a brief 10-minute introductory conversation sometime this week or next?"
6. PROFESSIONAL HUMAN SIGN-OFF:
   Best regards,
   Yeasaleh
   Software Developer
   +8801735782467 | yeasaleh.contact@gmail.com
   Portfolio: https://yeasaleh.xyz | LinkedIn: https://www.linkedin.com/in/yea-saleh

### 3. SUBJECT LINE EXCELLENCE
Create a compelling, professional subject line that stands out in an inbox and gets opened:
- Format options:
  * Application: [Job Title] — Yeasaleh (Next.js / Node.js)
  * [Job Title] role at [Company] — Yeasaleh
  * Engineering candidate for [Job Title] — Yeasaleh
Avoid spammy clickbait, all-caps, or emojis.

### 4. WORD COUNT & TONE
- Length: 120-190 words. Short, punchy, respectful of recruiter time.
- Tone: Natural, confident, direct, technically credible.

### 5. OUTPUT FORMAT
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
1. Candidate Identity: Candidate is Yeasaleh, Software Developer based in Dhaka, Bangladesh (Remote).
2. Human Conversational Tone: Write like an elite engineer reaching out directly. Do NOT use AI buzzwords or standard cover letter templates.
3. Quantified Achievements: Extract 2-3 specific, relevant metrics from Yeasaleh's actual experience (e.g. 0.01 CLS score with Next.js/React, 50k+ user systems with Node.js/MySQL, 85% workflow automation, Stripe/LemonSqueezy integrations).
4. Portfolio & GitHub: Include natural references to https://yeasaleh.xyz and https://github.com/yeasaleh2002.
5. Length: 120-190 words. Short, punchy, respectful of recruiter time.
6. Sign-off:
   Best regards,
   Yeasaleh
   {job_title}
   +8801735782467 | yeasaleh.contact@gmail.com
   Portfolio: https://yeasaleh.xyz | LinkedIn: https://www.linkedin.com/in/yea-saleh

OUTPUT FORMAT:
Return ONLY a valid JSON object formatted exactly as:
{{
  "email": "{resolved_email}",
  "subject": "<engaging, professional subject line>",
  "body": "<concise, personalized human email body>"
}}
"""

        logger.info("Generating personalized cold email for '%s' at '%s'...", job_title, company_name)

        # Step 3: Call LLM with graceful fallback
        default_subject = f"{job_title} Application - Yeasaleh | {company_name}"
        try:
            raw_response = generate_ai_response(
                prompt=user_prompt.strip(),
                system_prompt=system_prompt,
            )
            parsed_data = self._parse_json_response(raw_response, default_email=resolved_email, default_subject=default_subject)
            return parsed_data
        except Exception as exc:
            logger.warning("LLM call failed for cold email (%s), using deterministic fallback.", exc)
            fallback_body = (
                f"Hi {company_name} Hiring Team,\n\n"
                f"I am writing to express my enthusiastic interest in the {job_title} position. "
                f"With extensive full-stack experience building high-performance applications with React, Next.js, "
                f"and FastAPI, I have delivered microservices scaling to tens of thousands of users while optimizing "
                f"critical latency by up to 40%.\n\n"
                f"I would welcome the opportunity to discuss how my skill set can deliver immediate impact for {company_name}.\n\n"
                f"Best regards,\nYeasaleh\n{job_title}\n+8801735782467 | yeasaleh.contact@gmail.com\n"
                f"Portfolio: https://yeasaleh.xyz | LinkedIn: https://www.linkedin.com/in/yea-saleh"
            )
            return {
                "email": resolved_email,
                "subject": default_subject,
                "body": fallback_body,
            }

    def generate_cover_letter(
        self,
        job_title: str,
        job_description: str,
        company: Optional[str] = None,
        base_resume_text: Optional[str] = None,
    ) -> str:
        """
        Generate a professional, high-impact, ATS-optimized cover letter
        tailored to the target job title and description.
        """
        resume_content = base_resume_text or self.load_base_resume()
        company_name = (company or "Hiring Team").strip()

        system_prompt = (
            "You are an elite technical career advisor and professional cover letter writer. "
            f"Write a compelling, humanized, 3-4 paragraph technical cover letter for Yeasaleh applying "
            f"for the role of {job_title} at {company_name}. "
            "Highlight verified achievements from the resume (e.g. 0.01 CLS, 50k+ user scale, 85% workflow automation, React, Next.js, Node.js, FastAPI). "
            f"Use the target title '{job_title}' instead of generic 'Software Developer'. "
            "Write in standard American English, professional, direct, and zero generic clichés."
        )

        user_prompt = f"""
TARGET ROLE: {job_title}
COMPANY: {company_name}

JOB DESCRIPTION:
{job_description}

CANDIDATE BASE RESUME:
{resume_content}

INSTRUCTIONS:
- Address the hiring manager or technical recruiting team.
- Emphasize how candidate's specific background aligns with the requirements of {job_title}.
- Include direct links: Portfolio (https://yeasaleh.xyz), GitHub (https://github.com/yeasaleh2002), and LinkedIn (https://www.linkedin.com/in/yea-saleh).
- Sign-off as:
  Sincerely,
  Yeasaleh
  {job_title}
  +8801735782467 | yeasaleh.contact@gmail.com
"""

        logger.info("Generating tailored cover letter for '%s' at '%s'...", job_title, company_name)
        try:
            cover_letter = generate_ai_response(
                prompt=user_prompt.strip(),
                system_prompt=system_prompt,
            )
            return cover_letter.strip()
        except Exception as exc:
            logger.warning("LLM call failed for cover letter (%s), using deterministic fallback.", exc)
            return (
                f"Dear {company_name} Hiring Team,\n\n"
                f"I am writing to submit my application for the {job_title} position. "
                f"With demonstrated expertise in modern full-stack development, distributed API architecture, and performance optimization, "
                f"I am confident in my ability to bring immediate technical value to {company_name}.\n\n"
                f"Throughout my work at Nurix Hive and Manaknight Digital, I have specialized in engineering resilient backends, "
                f"responsive front-ends, and automated data pipelines that reduce latency and accelerate operational workflows.\n\n"
                f"Thank you for your time and consideration. You can explore my live work at https://yeasaleh.xyz.\n\n"
                f"Sincerely,\nYeasaleh\n{job_title}\n+8801735782467 | yeasaleh.contact@gmail.com"
            )

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

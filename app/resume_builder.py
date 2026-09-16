import html
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union

from reportlab.lib import colors  # type: ignore
from reportlab.lib.pagesizes import A4, letter  # type: ignore
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # type: ignore
from reportlab.pdfbase import pdfmetrics  # type: ignore
from reportlab.pdfbase.pdfmetrics import registerFontFamily  # type: ignore
from reportlab.pdfbase.ttfonts import TTFont  # type: ignore
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer  # type: ignore

import json
from pydantic import BaseModel, Field

from app.llm_manager import generate_ai_response
from app.rag_engine import RAGEngine

logger = logging.getLogger("your_assistant.resume_builder")

is_vercel = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))
DEFAULT_RESUME_FILE = Path("data") / "resume.txt"
OUTPUT_DIR = Path("/tmp/output") if is_vercel else Path("output")

_CALIBRI_REGISTERED = False
_PRIMARY_FONT = "Helvetica"
_PRIMARY_BOLD_FONT = "Helvetica-Bold"
_PRIMARY_ITALIC_FONT = "Helvetica-Oblique"


def get_calibri_font_family() -> tuple[str, str, str]:
    """
    Register Calibri TrueType font family with ReportLab.
    Searches repository data/fonts, system fonts, and falls back gracefully
    to standard Helvetica if font files are absent.
    """
    global _CALIBRI_REGISTERED, _PRIMARY_FONT, _PRIMARY_BOLD_FONT, _PRIMARY_ITALIC_FONT
    if _CALIBRI_REGISTERED:
        return _PRIMARY_FONT, _PRIMARY_BOLD_FONT, _PRIMARY_ITALIC_FONT

    search_dirs = [
        Path(__file__).resolve().parent.parent / "data" / "fonts",
        Path("data") / "fonts",
        Path(os.environ.get("WINDIR", "C:\\Windows")) / "Fonts",
        Path("/usr/share/fonts/truetype"),
        Path("/usr/share/fonts/truetype/msttcorefonts"),
    ]

    calibri_reg: Optional[Path] = None
    calibri_b: Optional[Path] = None
    calibri_i: Optional[Path] = None
    calibri_bi: Optional[Path] = None

    for d in search_dirs:
        if not d.exists():
            continue
        reg_candidates = list(d.glob("[cC][aA][lL][iI][bB][rR][iI].[tT][tT][fF]"))
        bold_candidates = list(d.glob("[cC][aA][lL][iI][bB][rR][iI][bB].[tT][tT][fF]"))
        italic_candidates = list(d.glob("[cC][aA][lL][iI][bB][rR][iI][iI].[tT][tT][fF]"))
        bi_candidates = list(d.glob("[cC][aA][lL][iI][bB][rR][iI][zZ].[tT][tT][fF]"))

        if reg_candidates and not calibri_reg:
            calibri_reg = reg_candidates[0]
        if bold_candidates and not calibri_b:
            calibri_b = bold_candidates[0]
        if italic_candidates and not calibri_i:
            calibri_i = italic_candidates[0]
        if bi_candidates and not calibri_bi:
            calibri_bi = bi_candidates[0]

    if calibri_reg and calibri_reg.exists():
        try:
            pdfmetrics.registerFont(TTFont("Calibri", str(calibri_reg)))
            b_font = str(calibri_b) if calibri_b and calibri_b.exists() else str(calibri_reg)
            i_font = str(calibri_i) if calibri_i and calibri_i.exists() else str(calibri_reg)
            bi_font = str(calibri_bi) if calibri_bi and calibri_bi.exists() else b_font

            pdfmetrics.registerFont(TTFont("Calibri-Bold", b_font))
            pdfmetrics.registerFont(TTFont("Calibri-Italic", i_font))
            pdfmetrics.registerFont(TTFont("Calibri-BoldItalic", bi_font))

            registerFontFamily(
                "Calibri",
                normal="Calibri",
                bold="Calibri-Bold",
                italic="Calibri-Italic",
                boldItalic="Calibri-BoldItalic",
            )
            _PRIMARY_FONT = "Calibri"
            _PRIMARY_BOLD_FONT = "Calibri-Bold"
            _PRIMARY_ITALIC_FONT = "Calibri-Italic"
            _CALIBRI_REGISTERED = True
            logger.info("Successfully registered Calibri font family from: %s", calibri_reg)
        except Exception as err:
            logger.warning("Failed to register Calibri TTFont (%s). Falling back to Helvetica.", err)
            _PRIMARY_FONT = "Helvetica"
            _PRIMARY_BOLD_FONT = "Helvetica-Bold"
            _PRIMARY_ITALIC_FONT = "Helvetica-Oblique"
            _CALIBRI_REGISTERED = True
    else:
        logger.warning("Calibri font file not found in search paths. Falling back to Helvetica.")
        _PRIMARY_FONT = "Helvetica"
        _PRIMARY_BOLD_FONT = "Helvetica-Bold"
        _PRIMARY_ITALIC_FONT = "Helvetica-Oblique"
        _CALIBRI_REGISTERED = True

    return _PRIMARY_FONT, _PRIMARY_BOLD_FONT, _PRIMARY_ITALIC_FONT

# Prompt Constraint for ATS compliance and anti-hallucination
STRICT_ATS_PROMPT = (
    "Rewrite the resume for 100% ATS compatibility. You MUST ONLY use skills and "
    "experiences present in the original resume. DO NOT hallucinate or add any fake skills."
)

FACT_CHECKER_SYSTEM_PROMPT = """You are a Strict ATS Fact-Checking Auditor and Anti-Hallucination Gatekeeper.

Your sole duty is to compare extracted Job Description (JD) keywords and requirements against the Candidate's Verified Context retrieved from ChromaDB.

CRITICAL DIRECTIVES:
1. ZERO ASSUMPTIONS: If a tool, language, framework, qualification, or metric in the JD is NOT explicitly mentioned or directly proven in the Candidate Context, you MUST classify it under `dropped_keywords`.
2. NEVER GUESS OR COMPENSATE: If the candidate lacks a skill, do NOT invent or substitute it. It must be explicitly dropped.
3. OUTPUT FORMAT: Output strictly valid JSON matching:
{
  "verified_keywords": ["keyword1", "keyword2"],
  "dropped_keywords": ["dropped_requirement1", "dropped_requirement2"],
  "proven_facts": ["proven fact 1", "proven fact 2"]
}
"""

EXPERT_RESUME_STRATEGIST_PROMPT = """You are an Expert Resume Writer and ATS Optimization Strategist. Your objective is to craft a highly compelling, professional resume tailored to a specific Job Description (JD), utilizing ONLY the provided Source Context.

As a professional resume writer, you must ensure the language is action-oriented, quantifiable, and impactful, while acting as a strict ATS gatekeeper to prevent any false information.

CRITICAL CONSTRAINTS:
1. ZERO HALLUCINATION: You must never invent, assume, or add skills, jobs, degrees, or metrics that are not explicitly stated in the Source Context.
2. NO FAKE SKILLS: If the JD requires a skill or experience the candidate lacks, DO NOT include it or try to compensate for it. Any keywords listed under FORBIDDEN_DROPPED_KEYWORDS must NEVER appear anywhere in the output.
3. ALIGNMENT & REWRITING: Identify keywords in the JD that naturally match the Source Context. Rewrite the candidate's bullet points to highlight these overlapping areas using the exact terminology from the JD, adhering to the "Action Verb + Task + Impact/Metric" formula (e.g., Accomplished [X], as measured by [Y], by doing [Z]).
4. FORMATTING: Output the final text using standard ATS headers: "Work Experience", "Education", and "Technical Skills". Do not use columns, tables, or complex formatting.
"""

EXPERT_COVER_LETTER_STRATEGIST_PROMPT = """You are an Expert Resume Writer and Executive Career Strategist. Write a concise, 3-paragraph cover letter based on the provided Candidate Resume and Job Description.

CRITICAL CONSTRAINTS:
1. ZERO HALLUCINATION: Base all claims strictly on the Candidate Resume. Do not invent enthusiasm or experience for tools the candidate hasn't used.
2. STRUCTURE: 
   - Paragraph 1: State the exact role applied for and a strong hook summarizing the candidate's most relevant core competency.
   - Paragraph 2: Highlight 1-2 specific achievements from the candidate's resume that directly solve the core problems outlined in the Job Description. Use metrics if available.
   - Paragraph 3: Brief conclusion and professional call to action.
3. ATS OPTIMIZATION: Seamlessly integrate 3-5 high-value keywords from the Job Description into the narrative.
4. TONE: Professional, confident, and direct. Avoid overly flowery language or clichés.
"""

WORLD_CLASS_ATS_PROMPT = """You are a World-Class ATS Resume Architect, Professional Resume Writer, Technical Recruiter, and ATS Optimization Specialist.

Your job is to analyze, build, rewrite, and optimize resumes to guarantee:
1. Top ATS (Applicant Tracking System) compatibility score (> 85% on Jobscan, Teal, ResumeWorded)
2. Exceptional human recruiter readability, hiring impact, and professional polish

You must behave like an expert resume reviewer who understands how modern ATS systems parse, rank, and match resumes against job descriptions.

CORE RESPONSIBILITIES

### 1. ATS Optimization & Parsing Architecture
* Ensure 100% compatibility with modern ATS parsers (Workday, Greenhouse, Lever, Taleo, iCIMS).
* Use standard, ATS-recognized section headings: Header, Professional Summary, Technical Skills, Professional Experience, Mentorship Experience, Education, Language.
* Optimize keywords naturally around the target job description based solely on skills the candidate genuinely has.
* Never keyword-stuff. Ensure keywords appear naturally in context with concrete engineering accomplishments.
* Avoid tables, text boxes, graphics, icons, columns, headers/footers, and non-standard symbols that break ATS parsing.
* Maintain clean, predictable, single-column hierarchy with standard bullet characters.

### 2. Mandatory Quantifiable Impact & Google X-Y-Z Formula (CRITICAL)
ATS checkers strictly penalize resumes lacking quantifiable achievements.
* EVERY SINGLE BULLET POINT in the Professional Experience and Mentorship sections MUST follow the Google X-Y-Z formula:
  "Accomplished [X], as measured by [Y], by doing [Z]"
* EVERY SINGLE BULLET POINT MUST CONTAIN EXPLICIT QUANTIFIABLE METRICS:
  - Percentages (e.g., "by 25%", "by 35%", "by 40%", "by 48%")
  - Performance/latency gains (e.g., "reduced API latency by 35%", "achieved 0.01 CLS score", "boosted frame rates to 60fps")
  - Scale & throughput (e.g., "50,000+ active users", "10,000+ weekly CRM interactions", "45+ developers mentored")
  - System reliability & efficiency (e.g., "sustained 99.9% uptime", "slashed load times by 48%", "automated 85% of workflows")
  - Time/cost savings (e.g., "reduced deployment cycles by 30%", "cut ticket resolution time by 15%")
* ZERO purely qualitative bullets are allowed in experience sections. An experience bullet lacking an explicit number, percentage, or scale metric is an automatic failure.
* Retain and adapt the verified metric magnitudes from the candidate's base resume, grounding them directly in the target role's technical requirements.

### 3. Strict USA English Grammar, Spelling & Language Quality
* All text MUST use standard American English (USA English) spelling and grammar conventions.
* REQUIRED AMERICAN SPELLINGS: "optimized" (never "optimised"), "analyzed" (never "analysed"), "prioritized" (never "prioritised"), "modeled" (never "modelled"), "behavior" (never "behaviour"), "catalog" (never "catalogue"), "program" (never "programme"), "synchronize" (never "synchronise"), "full-stack", "front-end", "back-end".
* ZERO spelling mistakes, typos, or grammatical errors are permitted.
* Active, precise action verbs only:
  - Current role (Nurix Hive): Active present-tense verbs (e.g., Architect, Engineer, Build, Automate, Deploy, Maintain, Program).
  - Past roles (Manaknight Digital, MedLink, Sadhinota Camp): Strong past-tense action verbs (e.g., Architected, Engineered, Spearheaded, Designed, Delivered, Accelerated, Reduced, Slashed, Automated, Mentored).
* Eliminate passive voice, clichés ("hard-working", "passionate"), and first-person pronouns (I, me, my).

### 4. Resume Parsing & Factual Integrity
* Preserve the candidate's exact employment history, company names, job titles, dates, locations, and educational credentials.
* Never fabricate fake companies, degrees, or certifications.

### 5. Resume Quality Control Audit
Before outputting, verify:
- ATS CHECK: Standard headings, single-column markdown, high keyword alignment.
- QUANTIFICATION CHECK: Did EVERY SINGLE bullet point include an explicit metric (%, ms, numbers, scale)? YES.
- LANGUAGE CHECK: 100% USA English spelling, flawless grammar, correct verb tenses.
- RECRUITER CHECK: Punchy 1-2 line bullets, immediate engineering impact visible.
"""


class FactCheckResult(BaseModel):
    """Result of Phase 1 LLM Fact-Checking & Anti-Hallucination Gatekeeper."""
    verified_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords from the JD that strictly exist in or are proven by the candidate context."
    )
    dropped_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords or requirements in the JD that the candidate lacks and MUST NEVER be added."
    )
    proven_facts: List[str] = Field(
        default_factory=list,
        description="Directly proven achievements, technologies, and metrics from candidate context."
    )


def format_markdown_for_reportlab(text: str) -> str:
    """
    Sanitize text and convert markdown inline markup (**bold**, *italic*)
    into ReportLab-supported XML tags (<b>, <i>).
    """
    # Escape basic XML/HTML special characters
    safe_text = html.escape(text, quote=False)

    # Convert **bold** to <b>bold</b>
    safe_text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe_text)
    # Convert *italic* to <i>italic</i>
    safe_text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", safe_text)
    # Convert `code` to font
    safe_text = re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', safe_text)

    return safe_text


class ResumeBuilder:
    """
    Engine that coordinates LLM-powered ATS resume tailoring and
    professional PDF generation.
    """

    def __init__(self, base_resume_path: Union[str, Path] = DEFAULT_RESUME_FILE):
        self.base_resume_path = Path(base_resume_path)

    def load_base_resume(self) -> str:
        """Load text from the base resume file."""
        if not self.base_resume_path.exists():
            raise FileNotFoundError(f"Base resume not found at: {self.base_resume_path}")
        return self.base_resume_path.read_text(encoding="utf-8").strip()

    def validate_and_filter_keywords(
        self,
        job_description: str,
        candidate_context: str,
        use_llm: bool = False,
    ) -> FactCheckResult:
        """
        Phase 1: Retrieval & Fact-Checking (Anti-Hallucination)
        1. Extracts candidate technical terms and requirements from the Job Description.
        2. Retrieves proven candidate context from ChromaDB.
        3. Cross-examines JD terms against Candidate Context:
           - ONLY selects keywords that strictly exist in or are directly proven by ChromaDB context.
           - Explicitly drops any JD requirements the candidate lacks.
        """
        raw_terms = re.findall(r"\b[A-Za-z0-9+#.-]{2,}\b", job_description)
        stop_words = {
            "the", "and", "with", "for", "that", "this", "from", "you", "will", "our",
            "are", "have", "experience", "role", "team", "years", "work", "looking",
            "candidate", "about", "what", "their", "must", "should", "ability",
        }
        candidate_terms = [t for t in dict.fromkeys(raw_terms) if t.lower() not in stop_words and len(t) > 2]

        # Retrieve proven atomic chunks from ChromaDB
        proven_chunks: List[str] = []
        try:
            rag = RAGEngine(resume_path=self.base_resume_path)
            proven_chunks = rag.retrieve_proven_context(candidate_terms[:25])
        except Exception as exc:
            logger.debug("ChromaDB chunk retrieval note: %s", exc)

        evidence_text = "\n\n".join(proven_chunks) if proven_chunks else candidate_context

        if use_llm:
            try:
                prompt = (
                    f"### JOB DESCRIPTION:\n{job_description}\n\n"
                    f"### CANDIDATE CONTEXT (FROM CHROMADB):\n{evidence_text}\n\n"
                    f"Cross-examine every JD requirement against Candidate Context. Output JSON only."
                )
                raw_resp = generate_ai_response(
                    prompt=prompt,
                    system_prompt=FACT_CHECKER_SYSTEM_PROMPT,
                )
                json_match = re.search(r"\{.*\}", raw_resp, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(0))
                    return FactCheckResult(**data)
            except Exception as exc:
                logger.warning("LLM fact-checking step failed (%s). Falling back to deterministic filter.", exc)

        # High-precision deterministic lexical verification fallback
        evidence_lower = evidence_text.lower()
        verified: List[str] = []
        dropped: List[str] = []

        for term in candidate_terms:
            t_lower = term.lower()
            # Strict word boundary check
            pattern = rf"\b{re.escape(t_lower)}\b"
            if re.search(pattern, evidence_lower):
                verified.append(term)
            else:
                dropped.append(term)

        # Extract verified experience facts
        facts = [
            line.strip("- ").strip()
            for line in evidence_text.splitlines()
            if line.strip().startswith("- ") and any(c.isdigit() for c in line)
        ]

        return FactCheckResult(
            verified_keywords=verified[:20],
            dropped_keywords=dropped[:25],
            proven_facts=facts[:8],
        )

    def tailor_resume(
        self,
        job_title: str,
        job_description: str,
        company: Optional[str] = None,
        base_resume_text: Optional[str] = None,
        run_llm_fact_check: bool = False,
    ) -> str:
        """
        Use LLM Manager to rewrite the base resume specifically for a matched job opportunity.
        Enforces Phase 1 Anti-Hallucination Fact-Checking and Phase 2 ATS Optimization:
        - Weaves in validated keywords naturally without inventing new facts.
        - Strictly drops unproven JD requirements.
        - Requires 'Action Verb + Task + Impact/Metric' format with concrete metrics.
        - Standard ATS headers: Work Experience, Education, Technical Skills.
        """
        resume_content = base_resume_text or self.load_base_resume()
        company_info = f" at {company}" if company else ""

        # Phase 1: Fact-Checking & Anti-Hallucination Gatekeeper
        fact_check = self.validate_and_filter_keywords(
            job_description=job_description,
            candidate_context=resume_content,
            use_llm=run_llm_fact_check,
        )

        user_prompt = f"""
TARGET JOB DETAILS:
- Title: {job_title}{company_info}
- Description & Requirements:
{job_description}

CANDIDATE BASE RESUME (GROUND TRUTH):
{resume_content}

PHASE 1 VALIDATED KEYWORDS (Strictly proven in Candidate Context - permitted to weave in):
{", ".join(fact_check.verified_keywords) if fact_check.verified_keywords else "Use candidate verified tools only"}

FORBIDDEN DROPPED REQUIREMENTS (Lacking in candidate context - DO NOT INVENT OR ADD):
{", ".join(fact_check.dropped_keywords[:20]) if fact_check.dropped_keywords else "None"}

MANDATORY INSTRUCTIONS:
1. Maintain the EXACT section structure and formatting from the candidate's base resume:
   - Header: Use '# Yeasaleh | {job_title}' (DO NOT use 'Software Developer' if the target role title is '{job_title}'). Followed by Location, Phone, Email, LinkedIn, GitHub, Portfolio.
   - Professional Summary: Align the summary opening directly to the target role of '{job_title}'.
   - Technical Skills (Front-End, Back-End, AI & Automation Tools)
   - Professional Experience (Nurix Hive, Manaknight Digital, MedLink Healthcare Private Limited)
   - Mentorship Experience (Sadhinota Camp)
   - Education
   - Language

2. FOR EACH OF THE 3 PROFESSIONAL ROLES:
   You MUST write AT LEAST 4 bullet points.
   CRITICAL MANDATORY REQUIREMENT FOR ATS SCORING (> 85%):
   - EVERY SINGLE BULLET POINT MUST follow the "Action Verb + Task + Impact/Metric" formula (Google X-Y-Z formula: "Accomplished [X], as measured by [Y], by doing [Z]").
   - EVERY SINGLE BULLET POINT MUST CONTAIN CONCRETE QUANTIFIABLE METRICS (percentages %, latency ms, frame rates fps, scale numbers, active user counts, throughput, or time/cost savings).
   - An experience bullet WITHOUT an explicit quantifiable metric (numbers or percentages) is strictly forbidden.
   - Tailor the candidate's verified metrics to directly target the required technologies and domain of the target job description.

3. ZERO HALLUCINATION & NO FAKE SKILLS:
   - You must NEVER invent or assume skills, tools, companies, degrees, or metrics not in the Source Context.
   - If the JD requires a skill the candidate lacks (listed in FORBIDDEN DROPPED REQUIREMENTS), DO NOT include it or try to compensate for it.

4. USA ENGLISH SPELLING & GRAMMAR:
   - Use standard American English (USA English) spelling exclusively: "optimized", "analyzed", "prioritized", "modeled", "behavior", "catalog", "full-stack", "front-end", "back-end".
   - Current role (Nurix Hive): Active present-tense verbs (Architect, Engineer, Build, Automate, Program, Deploy).
   - Past roles (Manaknight Digital, MedLink Healthcare): Strong past-tense action verbs (Architected, Engineered, Spearheaded, Built, Reduced, Slashed, Accelerated).
   - Zero spelling mistakes, zero typos, and flawless grammar throughout.

5. FOR TECHNICAL SKILLS:
   Update and re-order the skills in each category (Front-End, Back-End, AI & Automation Tools) to highlight the technologies most relevant to the target job description.

6. FORMATTING OUTPUT (Standard ATS Markdown):
   # Yeasaleh | {job_title}
   Dhaka, Bangladesh | +8801735782467 | yeasaleh.contact@gmail.com
   https://www.linkedin.com/in/yea-saleh | https://github.com/yeasaleh2002 | https://yeasaleh.xyz

   ## Professional Summary
   [Tailored 3-4 sentence summary emphasizing background as a {job_title} matching the target role, adhering strictly to USA English]

   ## Technical Skills
   **Front-End:** [Tailored front-end skills matching the job description from candidate context]
   **Back-End:** [Tailored back-end skills matching the job description from candidate context]
   **AI & Automation Tools:** [Tailored tools & integrations matching the job description from candidate context]

   ## Professional Experience

   ### Nurix Hive | Software Engineer
   *Dhaka, Bangladesh (Remote) | August 2025 – Present*
   - [Action Verb] + [Task] + [Impact/Metric % or number]
   - [Action Verb] + [Task] + [Impact/Metric % or number]
   - [Action Verb] + [Task] + [Impact/Metric % or number]
   - [Action Verb] + [Task] + [Impact/Metric % or number]

   ### Manaknight Digital | Web Developer
   *Toronto, Canada (Remote) | December 2023 – July 2025*
   - [Action Verb] + [Task] + [Impact/Metric % or number]
   - [Action Verb] + [Task] + [Impact/Metric % or number]
   - [Action Verb] + [Task] + [Impact/Metric % or number]
   - [Action Verb] + [Task] + [Impact/Metric % or number]

   ### MedLink Healthcare Private Limited | Software Engineer
   *Hyderabad, India (Remote) | March 2022 – December 2023*
   - [Action Verb] + [Task] + [Impact/Metric % or number]
   - [Action Verb] + [Task] + [Impact/Metric % or number]
   - [Action Verb] + [Task] + [Impact/Metric % or number]
   - [Action Verb] + [Task] + [Impact/Metric % or number]

   ## Mentorship Experience

   ### Sadhinota Camp | Support Mentor (Voluntary)
   *Dhaka, Bangladesh | September 2024 – April 2025*
   - Mentored 45+ students in full-stack architecture, improving student code quality by 40% through personalized code reviews and guidance.
   - Formulated practical learning resources focused on frontend performance and error handling, enabling learners to reduce runtime errors by 30%.

   ## Education

   ### B.Sc. in Computer Science & Engineering (2023 - Present)
   City University, Bangladesh

   ## Language
   Bangla (Native), English (Professional Working Proficiency)

CRITICAL CONSTRAINT:
{STRICT_ATS_PROMPT}
"""

        full_user_prompt = f"{EXPERT_RESUME_STRATEGIST_PROMPT}\n\n{WORLD_CLASS_ATS_PROMPT}\n\n{user_prompt.strip()}"

        logger.info("Requesting ATS resume tailoring for '%s%s'...", job_title, company_info)
        try:
            tailored_markdown = generate_ai_response(
                prompt=full_user_prompt.strip(),
                system_prompt=STRICT_ATS_PROMPT,
            )
            return tailored_markdown.strip()
        except Exception as exc:
            logger.warning(
                "LLM generation failed in tailor_resume (%s). Using high-fidelity base resume fallback with target role '%s'.",
                exc,
                job_title,
            )
            fallback = resume_content.replace("Yeasaleh | Software Developer", f"Yeasaleh | {job_title}")
            fallback = fallback.replace("Software Developer specializing", f"{job_title} specializing")
            if not fallback.strip().startswith("# "):
                fallback = f"# {fallback}"
            return fallback.strip()

    def generate_cover_letter(
        self,
        job_title: str,
        job_description: str,
        company: Optional[str] = None,
        base_resume_text: Optional[str] = None,
    ) -> str:
        """
        Generate a concise, 3-paragraph ATS-optimized cover letter based on
        Candidate Resume and Job Description with zero hallucination.
        """
        resume_content = base_resume_text or self.load_base_resume()
        company_name = (company or "Hiring Team").strip()
        fact_check = self.validate_and_filter_keywords(job_description, resume_content, use_llm=False)

        user_prompt = f"""
TARGET ROLE: {job_title} at {company_name}

JOB DESCRIPTION:
{job_description}

CANDIDATE RESUME (GROUND TRUTH):
{resume_content}

VALIDATED OVERLAPPING KEYWORDS:
{", ".join(fact_check.verified_keywords) if fact_check.verified_keywords else "Full-stack web architecture, performance optimization"}

FORBIDDEN DROPPED KEYWORDS (DO NOT USE):
{", ".join(fact_check.dropped_keywords[:15]) if fact_check.dropped_keywords else "None"}

Write the concise, 3-paragraph cover letter strictly following the structure and zero-hallucination constraints:
- Paragraph 1: State role applied for and hook summarizing core competency.
- Paragraph 2: Highlight 1-2 specific achievements solving JD problems with metrics.
- Paragraph 3: Brief conclusion and professional call to action.
- Sign-off as Yeasaleh, {job_title}.
"""
        logger.info("Generating ATS cover letter for '%s' at '%s'...", job_title, company_name)
        try:
            return generate_ai_response(
                prompt=user_prompt.strip(),
                system_prompt=EXPERT_COVER_LETTER_STRATEGIST_PROMPT,
            ).strip()
        except Exception as exc:
            logger.warning("Cover letter generation failed (%s), using fallback.", exc)
            return (
                f"Dear {company_name} Hiring Team,\n\n"
                f"I am writing to express my enthusiastic interest in the {job_title} position at {company_name}. "
                f"With extensive full-stack experience architecting scalable Next.js and Node.js applications, "
                f"I specialize in engineering high-performance APIs and optimizing user interfaces for mission-critical platforms.\n\n"
                f"At Nurix Hive and Manaknight Digital, I engineered modular systems that accelerated API response times by 25%, "
                f"slashed initial page load times by 48%, and automated 85% of repetitive workflows while supporting 50,000+ active users.\n\n"
                f"I welcome the opportunity to discuss how my technical expertise can solve key engineering challenges at {company_name}. "
                f"You can explore my technical projects at https://yeasaleh.xyz.\n\n"
                f"Sincerely,\nYeasaleh\n{job_title}\n+8801735782467 | yeasaleh.contact@gmail.com"
            )

    def generate_pdf(
        self,
        markdown_text: str,
        output_path: Union[str, Path],
    ) -> Path:
        """
        Convert structured Markdown resume into an ATS-optimized, publication-ready PDF
        matching the exact typography and format of data/Yeasaleh_Resume.docx:
        - Font family: Calibri
        - Candidate Name & Dynamic Target Role: 22 pt Bold
        - Section headings: 18 pt Bold
        - Experience company name: 14 pt Bold
        - Normal text (Summary, Skills, Bullets, Meta, Education, Language): 12 pt
        - Strict 100% black text for maximum ATS scannability
        """
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        normal_font, bold_font, italic_font = get_calibri_font_family()

        doc = SimpleDocTemplate(
            str(out_file),
            pagesize=A4,     # Standard A4 matching data/Yeasaleh_Resume.docx
            leftMargin=36,   # 0.5 inch margins for ATS scannability
            rightMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        styles = getSampleStyleSheet()

        # Strict 100% black text for ATS parsing fidelity
        text_color = colors.black

        name_style = ParagraphStyle(
            "ResumeName",
            parent=styles["Normal"],
            fontName=bold_font,
            fontSize=22,
            leading=26,
            textColor=text_color,
            alignment=0,
            spaceAfter=3,
        )

        contact_style = ParagraphStyle(
            "ResumeContact",
            parent=styles["Normal"],
            fontName=normal_font,
            fontSize=12,
            leading=15,
            textColor=text_color,
            spaceAfter=2,
        )

        heading_style = ParagraphStyle(
            "ResumeSectionHeading",
            parent=styles["Normal"],
            fontName=bold_font,
            fontSize=18,
            leading=22,
            textColor=text_color,
            spaceBefore=10,
            spaceAfter=4,
            keepWithNext=True,
        )

        company_role_style = ParagraphStyle(
            "ResumeCompanyRole",
            parent=styles["Normal"],
            fontName=normal_font,
            fontSize=12,
            leading=16,
            textColor=text_color,
            spaceBefore=6,
            spaceAfter=1,
            keepWithNext=True,
        )

        role_meta_style = ParagraphStyle(
            "ResumeRoleMeta",
            parent=styles["Normal"],
            fontName=italic_font,
            fontSize=12,
            leading=15,
            textColor=text_color,
            spaceAfter=3,
            keepWithNext=True,
        )

        body_style = ParagraphStyle(
            "ResumeBody",
            parent=styles["Normal"],
            fontName=normal_font,
            fontSize=12,
            leading=15.5,
            textColor=text_color,
            spaceAfter=3,
        )

        bullet_style = ParagraphStyle(
            "ResumeBullet",
            parent=body_style,
            fontName=normal_font,
            fontSize=12,
            leading=15,
            leftIndent=14,
            firstLineIndent=-10,
            spaceAfter=2.5,
        )

        story: List[Any] = []
        lines = markdown_text.splitlines()
        seen_first_section = False

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            # Header 1: Candidate Name & Dynamic Role
            # (matches "# Yeasaleh | Title" or "Yeasaleh | Title" before any section)
            if line.startswith("# ") or (not seen_first_section and line.startswith("Yeasaleh | ")):
                header_raw = line[2:].strip() if line.startswith("# ") else line
                name_text = format_markdown_for_reportlab(header_raw)
                story.append(Paragraph(f"<b>{name_text}</b>", name_style))
                continue

            # Section Titles: ## Title or recognized exact section name
            is_sec_heading = line.startswith("## ") or line in [
                "Professional Summary",
                "Summary",
                "Technical Skills",
                "Skills",
                "Professional Experience",
                "Work Experience",
                "Experience",
                "Mentorship Experience",
                "Education",
                "Language",
                "Languages",
            ]
            if is_sec_heading:
                seen_first_section = True
                raw_title = line[3:].strip() if line.startswith("## ") else line
                sec_text = format_markdown_for_reportlab(raw_title)
                story.append(Paragraph(f"<b>{sec_text}</b>", heading_style))
                continue

            # Subheadings (Company / Role / Project): ### Company | Role or recognized company line
            is_company_subheading = line.startswith("### ") or (
                seen_first_section
                and " | " in line
                and not line.startswith("-")
                and not line.startswith("*")
                and not line.startswith("•")
            )
            if is_company_subheading:
                raw_sub = line[4:].strip() if line.startswith("### ") else line
                # Check for "Company | Role" structure
                if " | " in raw_sub:
                    parts = raw_sub.split(" | ", 1)
                    comp_name = format_markdown_for_reportlab(parts[0].strip())
                    role_name = format_markdown_for_reportlab(parts[1].strip())
                    # 14 pt bold for company name, 12 pt for role
                    formatted_sub = f'<font size="14"><b>{comp_name}</b></font> | <font size="12">{role_name}</font>'
                else:
                    sub_formatted = format_markdown_for_reportlab(raw_sub)
                    formatted_sub = f'<font size="14"><b>{sub_formatted}</b></font>'
                story.append(Paragraph(formatted_sub, company_role_style))
                continue

            # Contact line / metadata below name before the first section
            if not seen_first_section:
                contact_text = format_markdown_for_reportlab(line)
                story.append(Paragraph(contact_text, contact_style))
                continue

            # Italic Role metadata (Location, Dates) e.g. *Dhaka, Bangladesh...* or line with dates/location
            if (
                (line.startswith("*") and line.endswith("*"))
                or ("|" in line and any(yr in line for yr in ["2022", "2023", "2024", "2025", "Present"]))
                or line.startswith("Dhaka, Bangladesh")
                or line.startswith("Toronto, Canada")
                or line.startswith("Hyderabad, India")
            ):
                clean_meta = line.strip("*").strip()
                meta_text = format_markdown_for_reportlab(clean_meta)
                story.append(Paragraph(f"<i>{meta_text}</i>", role_meta_style))
                continue

            # Bullet points
            if line.startswith("- ") or line.startswith("* ") or line.startswith("• "):
                raw_bullet = line[2:].strip() if (line.startswith("- ") or line.startswith("* ")) else line[1:].strip()
                bullet_content = format_markdown_for_reportlab(raw_bullet)
                formatted_bullet = f"&bull;&nbsp;&nbsp;{bullet_content}"
                story.append(Paragraph(formatted_bullet, bullet_style))
                continue

            # Regular paragraph text (Summary, Skills lines, Education details, Language)
            para_line = line
            for skill_cat in ["Front-End:", "Back-End:", "AI & Automation Tools:", "Languages & Frameworks:", "Databases & Tools:"]:
                if para_line.startswith(skill_cat) and not para_line.startswith(f"**{skill_cat}"):
                    para_line = f"**{skill_cat}** {para_line[len(skill_cat):].strip()}"
                    break

            para_text = format_markdown_for_reportlab(para_line)
            story.append(Paragraph(para_text, body_style))

        # Build document
        doc.build(story)
        logger.info("Generated ATS-optimized resume PDF at: %s", out_file)
        return out_file

    def build_tailored_resume_pdf(
        self,
        job_title: str,
        job_description: str,
        company: Optional[str] = None,
        base_resume_text: Optional[str] = None,
        output_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Orchestrate complete pipeline:
        1. Phase 1: Fact-Checking & Anti-Hallucination Gatekeeper.
        2. Phase 2: ATS-Optimized Markdown Tailoring.
        3. Phase 3: Machine-Readable Single-Column PDF Generation.
        """
        resume_content = base_resume_text or self.load_base_resume()

        # Step 1: Fact-Checking & Anti-Hallucination Filter
        fact_check = self.validate_and_filter_keywords(
            job_description=job_description,
            candidate_context=resume_content,
            use_llm=False,
        )

        # Step 2: Tailor Markdown
        tailored_markdown = self.tailor_resume(
            job_title=job_title,
            job_description=job_description,
            company=company,
            base_resume_text=base_resume_text,
        )

        # Step 3: Generate PDF
        clean_title = re.sub(r"[^a-zA-Z0-9]+", "_", job_title or "Engineer").strip("_")
        clean_company = re.sub(r"[^a-zA-Z0-9]+", "_", company).strip("_") if company and company.strip() else ""
        if output_filename:
            out_p = Path(output_filename)
            if out_p.is_absolute() or len(out_p.parts) > 1:
                pdf_path = out_p
                fname = out_p.name
            else:
                raw = output_filename
                if raw.startswith("Resume_"):
                    fname = f"Yeasaleh_{raw}"
                elif not raw.startswith("Yeasaleh_Resume"):
                    fname = f"Yeasaleh_Resume_{raw}"
                else:
                    fname = raw
                pdf_path = OUTPUT_DIR / fname
        else:
            if clean_company and clean_company.lower() not in ["none", "null", "company"]:
                fname = f"Yeasaleh_Resume_{clean_company}_{clean_title}.pdf"
            else:
                fname = f"Yeasaleh_Resume_{clean_title}.pdf"
            pdf_path = OUTPUT_DIR / fname

        generated_path = self.generate_pdf(tailored_markdown, pdf_path)

        return {
            "job_title": job_title,
            "company": company,
            "tailored_markdown": tailored_markdown,
            "pdf_path": str(generated_path),
            "filename": fname,
            "verified_keywords": fact_check.verified_keywords,
            "dropped_keywords": fact_check.dropped_keywords,
        }

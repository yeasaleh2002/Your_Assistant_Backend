import html
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union

from reportlab.lib import colors  # type: ignore
from reportlab.lib.pagesizes import letter  # type: ignore
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # type: ignore
from reportlab.platypus import HRFlowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer  # type: ignore

from app.llm_manager import generate_ai_response

logger = logging.getLogger("your_assistant.resume_builder")

is_vercel = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))
DEFAULT_RESUME_FILE = Path("data") / "resume.txt"
OUTPUT_DIR = Path("/tmp/output") if is_vercel else Path("output")

# Prompt Constraint for ATS compliance and anti-hallucination
STRICT_ATS_PROMPT = (
    "Rewrite the resume for 100% ATS compatibility. You MUST ONLY use skills and "
    "experiences present in the original resume. DO NOT hallucinate or add any fake skills."
)

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

    def tailor_resume(
        self,
        job_title: str,
        job_description: str,
        company: Optional[str] = None,
        base_resume_text: Optional[str] = None,
    ) -> str:
        """
        Use LLM Manager to rewrite the base resume specifically for a matched job opportunity.
        Strictly preserves the candidate's resume format, requires at least 4 bullet points per job,
        and updates the skills section.
        """
        resume_content = base_resume_text or self.load_base_resume()
        company_info = f" at {company}" if company else ""

        user_prompt = f"""
TARGET JOB DETAILS:
- Title: {job_title}{company_info}
- Description & Requirements:
{job_description}

CANDIDATE BASE RESUME:
{resume_content}

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
   - EVERY SINGLE BULLET POINT MUST CONTAIN CONCRETE QUANTIFIABLE METRICS (percentages %, latency ms, frame rates fps, scale numbers, active user counts, throughput, or time/cost savings) following the Google X-Y-Z formula: "Accomplished [X], as measured by [Y], by doing [Z]".
   - An experience bullet WITHOUT an explicit quantifiable metric (numbers or percentages) is strictly forbidden.
   - Tailor the candidate's verified metrics to directly target the required technologies and domain of the target job description.

3. USA ENGLISH SPELLING & GRAMMAR:
   - Use standard American English (USA English) spelling exclusively: "optimized", "analyzed", "prioritized", "modeled", "behavior", "catalog", "full-stack", "front-end", "back-end".
   - Current role (Nurix Hive): Active present-tense verbs (Architect, Engineer, Build, Automate, Program, Deploy).
   - Past roles (Manaknight Digital, MedLink Healthcare): Strong past-tense action verbs (Architected, Engineered, Spearheaded, Built, Reduced, Slashed, Accelerated).
   - Zero spelling mistakes, zero typos, and flawless grammar throughout.

4. FOR TECHNICAL SKILLS:
   Update and re-order the skills in each category (Front-End, Back-End, AI & Automation Tools) to highlight the technologies most relevant to the target job description.

5. FORMATTING OUTPUT (Standard ATS Markdown):
   # Yeasaleh | {job_title}
   Dhaka, Bangladesh | +8801735782467 | yeasaleh.contact@gmail.com
   https://www.linkedin.com/in/yea-saleh | https://github.com/yeasaleh2002 | https://yeasaleh.xyz

   ## Professional Summary
   [Tailored 3-4 sentence summary emphasizing background as a {job_title} matching the target role, adhering strictly to USA English]

   ## Technical Skills
   **Front-End:** [Tailored front-end skills matching the job description]
   **Back-End:** [Tailored back-end skills matching the job description]
   **AI & Automation Tools:** [Tailored tools & integrations matching the job description]

   ## Professional Experience

   ### Nurix Hive | Software Engineer
   *Dhaka, Bangladesh (Remote) | August 2025 – Present*
   - [Tailored bullet point 1 with explicit quantifiable metric % or number]
   - [Tailored bullet point 2 with explicit quantifiable metric % or number]
   - [Tailored bullet point 3 with explicit quantifiable metric % or number]
   - [Tailored bullet point 4 with explicit quantifiable metric % or number]

   ### Manaknight Digital | Web Developer
   *Toronto, Canada (Remote) | December 2023 – July 2025*
   - [Tailored bullet point 1 with explicit quantifiable metric % or number]
   - [Tailored bullet point 2 with explicit quantifiable metric % or number]
   - [Tailored bullet point 3 with explicit quantifiable metric % or number]
   - [Tailored bullet point 4 with explicit quantifiable metric % or number]

   ### MedLink Healthcare Private Limited | Software Engineer
   *Hyderabad, India (Remote) | March 2022 – December 2023*
   - [Tailored bullet point 1 with explicit quantifiable metric % or number]
   - [Tailored bullet point 2 with explicit quantifiable metric % or number]
   - [Tailored bullet point 3 with explicit quantifiable metric % or number]
   - [Tailored bullet point 4 with explicit quantifiable metric % or number]

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

        full_user_prompt = f"{WORLD_CLASS_ATS_PROMPT}\n\n{user_prompt.strip()}"

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

    def generate_pdf(
        self,
        markdown_text: str,
        output_path: Union[str, Path],
    ) -> Path:
        """
        Convert structured Markdown resume into an ATS-optimized, publication-ready PDF
        using ReportLab. All text is strictly 100% black color for maximum ATS parsing fidelity.
        """
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(
            str(out_file),
            pagesize=letter,
            leftMargin=36,   # 0.5 inch margins for ATS scannability
            rightMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        styles = getSampleStyleSheet()

        # Custom Typography and Color Palette - STRICT ALL-BLACK TEXT FOR ATS COMPATIBILITY
        primary_color = colors.black      # 100% Black (#000000)
        secondary_color = colors.black    # 100% Black (#000000)
        text_color = colors.black         # 100% Black (#000000)
        meta_color = colors.black         # 100% Black (#000000)
        line_color = colors.black         # 100% Black (#000000)

        name_style = ParagraphStyle(
            "ResumeName",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=primary_color,
            alignment=0,
            spaceAfter=3,
        )

        contact_style = ParagraphStyle(
            "ResumeContact",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11.5,
            textColor=meta_color,
            spaceAfter=2,
        )

        heading_style = ParagraphStyle(
            "ResumeSectionHeading",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=13.5,
            textColor=secondary_color,
            spaceBefore=7,
            spaceAfter=2,
            textTransform="uppercase",
            keepWithNext=True,
        )

        subheading_style = ParagraphStyle(
            "ResumeSubheading",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9.5,
            leading=12.5,
            textColor=primary_color,
            spaceBefore=4,
            spaceAfter=1,
            keepWithNext=True,
        )

        role_meta_style = ParagraphStyle(
            "ResumeRoleMeta",
            parent=styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8.5,
            leading=11,
            textColor=meta_color,
            spaceAfter=3,
            keepWithNext=True,
        )

        body_style = ParagraphStyle(
            "ResumeBody",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=text_color,
            spaceAfter=3,
        )

        bullet_style = ParagraphStyle(
            "ResumeBullet",
            parent=body_style,
            leftIndent=12,
            firstLineIndent=-8,
            spaceAfter=2,
            leading=11.5,
        )

        story: List[Any] = []
        lines = markdown_text.splitlines()
        seen_first_section = False

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            # Header 1: Candidate Name & Title
            if line.startswith("# "):
                name_text = format_markdown_for_reportlab(line[2:].strip())
                story.append(Paragraph(name_text, name_style))
                continue

            # Header 2: Section Titles
            if line.startswith("## "):
                seen_first_section = True
                raw_title = line[3:].strip()

                # User requirement: Put mentorship experience cleanly at the top of the next page
                # to prevent the section heading or partial bullets from splitting awkwardly across pages
                if "mentor" in raw_title.lower():
                    story.append(PageBreak())

                sec_text = format_markdown_for_reportlab(raw_title)
                story.append(Paragraph(sec_text, heading_style))
                story.append(
                    HRFlowable(
                        width="100%",
                        thickness=0.5,
                        color=line_color,
                        spaceBefore=1,
                        spaceAfter=4,
                    )
                )
                continue

            # Header 3: Subheadings (Company / Role / Project)
            if line.startswith("### "):
                sub_text = format_markdown_for_reportlab(line[4:].strip())
                story.append(Paragraph(sub_text, subheading_style))
                continue

            # Contact line / metadata below name before the first section
            if not seen_first_section:
                contact_text = format_markdown_for_reportlab(line)
                story.append(Paragraph(contact_text, contact_style))
                continue

            # Italic Role metadata (Location, Dates) e.g. *Dhaka, Bangladesh...* or line with dates
            if (line.startswith("*") and line.endswith("*")) or ("|" in line and any(yr in line for yr in ["2022", "2023", "2024", "2025", "Present"])):
                clean_meta = line.strip("*").strip()
                meta_text = format_markdown_for_reportlab(clean_meta)
                story.append(Paragraph(meta_text, role_meta_style))
                continue

            # Bullet points
            if line.startswith("- ") or line.startswith("* "):
                bullet_content = format_markdown_for_reportlab(line[2:].strip())
                formatted_bullet = f"&bull;&nbsp; {bullet_content}"
                story.append(Paragraph(formatted_bullet, bullet_style))
                continue

            # Regular paragraph text (Summary, Skills lines, etc.)
            para_text = format_markdown_for_reportlab(line)
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
        1. Tailors resume via LLM with strict anti-hallucination prompt.
        2. Converts tailored Markdown into professional ATS-friendly PDF.
        """
        # Step 1: Tailor Markdown
        tailored_markdown = self.tailor_resume(
            job_title=job_title,
            job_description=job_description,
            company=company,
            base_resume_text=base_resume_text,
        )

        # Step 2: Generate PDF
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
        }

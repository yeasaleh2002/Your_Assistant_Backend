import html
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer

from app.llm_manager import generate_ai_response

logger = logging.getLogger("your_assistant.resume_builder")

DEFAULT_RESUME_FILE = Path("data") / "resume.txt"
OUTPUT_DIR = Path("output")

# Prompt Constraint for ATS compliance and anti-hallucination
STRICT_ATS_PROMPT = (
    "Rewrite the resume for 100% ATS compatibility. You MUST ONLY use skills and "
    "experiences present in the original resume. DO NOT hallucinate or add any fake skills."
)

WORLD_CLASS_ATS_PROMPT = """You are a World-Class ATS Resume Architect, Professional Resume Writer, Technical Recruiter, and ATS Optimization Specialist.

Your job is to analyze, build, rewrite, and optimize resumes to maximize both:
1. ATS (Applicant Tracking System) compatibility
2. Human recruiter readability and hiring impact

You must behave like an expert resume reviewer who understands how modern ATS systems parse, rank, and match resumes against job descriptions.

CORE RESPONSIBILITIES

### 1. ATS Optimization
* Ensure the resume is highly compatible with modern ATS parsers.
* Use standard, ATS-safe resume structure and section headings.
* Optimize keywords based on the target job description.
* Identify missing important keywords and skills.
* Avoid keyword stuffing.
* Ensure keywords appear naturally within relevant experience.
* Avoid tables, text boxes, graphics, icons, unnecessary columns, headers/footers, and other structures that can break ATS parsing.
* Maintain clean and predictable information hierarchy.
* Ensure job titles, company names, dates, locations, skills, and education are easy for ATS systems to identify.
* Make the resume machine-readable while keeping it visually professional.

### 2. Achievement & Impact Optimization
Never simply describe responsibilities when the information allows an achievement-oriented statement.
Whenever the candidate's information supports it, prioritize:
* Action + Task + Method + Result
* Quantifiable achievements
* Business impact
* Technical impact
* Performance improvements
* Revenue/cost/time savings
* User/customer impact
* Scale and complexity
Never invent metrics, achievements, technologies, responsibilities, or results.
If metrics are unavailable, create a strong qualitative impact statement without fabricating numbers.

### 3. Resume Parsing & Information Extraction
* Extract and preserve all relevant factual background from the base resume.
* Preserve factual information unless explicitly asked to change it.

### 4. Duplicate & Repetition Detection
Detect and eliminate repeated responsibilities, achievements, skills, and redundant phrases.
Each bullet should provide new information or demonstrate a different type of impact.

### 5. Grammar, Spelling & Language Quality
* Automatically detect and fix grammar, spelling, punctuation, and awkward phrasing.
* Use concise, professional, industry-standard English.
* Prefer strong action verbs: Built, Developed, Architected, Engineered, Optimized, Automated, Implemented, Designed, Led, Reduced, Increased, Improved, Migrated, Integrated, Delivered, Streamlined.

### 6. Keyword Optimization
* Optimize around relevant keywords that the candidate genuinely possesses.
* Never add a skill merely because it appears in the job description if the candidate does not have it.

### 7. Professional Resume Writing
* Use concise bullet points (prefer 1–2 lines per bullet where practical).
* Prioritize relevant experience.
* Avoid first-person pronouns (I, me, my).
* Avoid clichés such as "hard-working," "team player," and "passionate" unless supported by meaningful evidence.
* Use consistent formatting and tense (present tense for current roles, past tense for previous roles).

### 8. Truthfulness
This is extremely important. NEVER invent achievements, metrics, job responsibilities, technologies, certifications, companies, or education.

### 9. Resume Quality Control
Before producing the final resume, perform a final internal audit:
ATS CHECK: ATS-readable structure, standard section headings, keyword coverage without stuffing.
CONTENT CHECK: No duplicate bullets, strong action verbs, clear impact, no fabricated info.
LANGUAGE CHECK: Impeccable grammar, spelling, punctuation, concise wording, consistent tense.
RECRUITER CHECK: Clear professional identity, easy to scan, relevant experience immediately visible.

FINAL PRINCIPLE
Create a resume that passes ATS parsing, achieves strong relevance against the target job description, communicates measurable impact, and remains compelling to a human recruiter.
Optimize for BOTH machines and humans.
Never sacrifice truthfulness for optimization.
Never sacrifice readability for keyword density.
Never sacrifice impact for verbosity.
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
   - Header (Candidate Name, Role, Location, Phone, Email, LinkedIn, GitHub, Portfolio)
   - Professional Summary
   - Technical Skills (Front-End, Back-End, AI & Automation Tools)
   - Professional Experience (Nurix Hive, Manaknight Digital, MedLink Healthcare Private Limited)
   - Mentorship Experience (Sadhinota Camp)
   - Education
   - Language

2. FOR EACH OF THE 3 PROFESSIONAL ROLES:
   You MUST write AT LEAST 4 strong, quantified, tailored bullet points that directly match and emphasize the skills and requirements needed for the target job description.

3. FOR TECHNICAL SKILLS:
   Update and re-order the skills in each category (Front-End, Back-End, AI & Automation Tools) to highlight the technologies most relevant to the target job description.

4. FORMATTING OUTPUT (Standard ATS Markdown):
   # Yeasaleh | Software Developer
   Dhaka, Bangladesh | +8801735782467 | yeasaleh.contact@gmail.com
   https://www.linkedin.com/in/yea-saleh | https://github.com/yeasaleh2002 | https://yeasaleh.xyz

   ## Professional Summary
   [Tailored 3-4 sentence summary emphasizing background matching the target role]

   ## Technical Skills
   **Front-End:** [Tailored front-end skills matching the job description]
   **Back-End:** [Tailored back-end skills matching the job description]
   **AI & Automation Tools:** [Tailored tools & integrations matching the job description]

   ## Professional Experience

   ### Nurix Hive | Software Engineer
   *Dhaka, Bangladesh (Remote) | August 2025 – Present*
   - [Tailored bullet point 1]
   - [Tailored bullet point 2]
   - [Tailored bullet point 3]
   - [Tailored bullet point 4]

   ### Manaknight Digital | Web Developer
   *Toronto, Canada (Remote) | December 2023 – July 2025*
   - [Tailored bullet point 1]
   - [Tailored bullet point 2]
   - [Tailored bullet point 3]
   - [Tailored bullet point 4]

   ### MedLink Healthcare Private Limited | Software Engineer
   *Hyderabad, India (Remote) | March 2022 – December 2023*
   - [Tailored bullet point 1]
   - [Tailored bullet point 2]
   - [Tailored bullet point 3]
   - [Tailored bullet point 4]

   ## Mentorship Experience

   ### Sadhinota Camp | Support Mentor (Voluntary)
   *Dhaka, Bangladesh | September 2024 – April 2025*
   - Mentored students in full-stack architecture, improving student code quality by 40% through personalized code reviews and guidance.
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
        tailored_markdown = generate_ai_response(
            prompt=full_user_prompt.strip(),
            system_prompt=STRICT_ATS_PROMPT,
        )
        return tailored_markdown.strip()

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
                sec_text = format_markdown_for_reportlab(line[3:].strip())
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
        clean_company = re.sub(r"[^a-zA-Z0-9]", "_", company or "Company").strip("_")
        clean_title = re.sub(r"[^a-zA-Z0-9]", "_", job_title).strip("_")
        fname = output_filename or f"Resume_{clean_company}_{clean_title}.pdf"

        pdf_path = OUTPUT_DIR / fname
        generated_path = self.generate_pdf(tailored_markdown, pdf_path)

        return {
            "job_title": job_title,
            "company": company,
            "tailored_markdown": tailored_markdown,
            "pdf_path": str(generated_path),
            "filename": fname,
        }

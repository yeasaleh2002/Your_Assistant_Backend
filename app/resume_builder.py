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

# STRICT Prompt Constraint required for zero hallucination and 100% ATS compliance
STRICT_ATS_PROMPT = (
    "Rewrite the resume for 100% ATS compatibility. You MUST ONLY use skills and "
    "experiences present in the original resume. DO NOT hallucinate or add any fake skills."
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

    def tailor_resume(
        self,
        job_title: str,
        job_description: str,
        company: Optional[str] = None,
        base_resume_text: Optional[str] = None,
    ) -> str:
        """
        Use LLM Manager to rewrite the base resume specifically for a matched job opportunity.
        Strictly enforces ATS compatibility and zero hallucination.
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

INSTRUCTIONS:
1. Re-organize, emphasize, and highlight the candidate's existing experience and technical keywords that directly match the target job.
2. Structure the resume in standard ATS Markdown format:
   # [Candidate Name]
   [Job Title | Contact Information, Email, LinkedIn, GitHub]
   
   ## Professional Summary
   [Tailored 3-4 sentence summary using existing background]

   ## Technical Skills
   [Categorized list: Languages, Frameworks, Cloud, Tools]

   ## Professional Experience
   [Company, Role, Dates]
   - [Bullet points with metrics and direct relevance to the job]

   ## Education & Certifications
   [Degrees and credentials]

CRITICAL CONSTRAINT:
{STRICT_ATS_PROMPT}
"""

        logger.info("Requesting ATS resume tailoring for '%s%s'...", job_title, company_info)
        tailored_markdown = generate_ai_response(
            prompt=user_prompt.strip(),
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
        using ReportLab.
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

        # Custom Typography and Color Palette
        primary_color = colors.HexColor("#0F172A")    # Slate 900
        secondary_color = colors.HexColor("#1E3A8A")  # Navy Blue
        text_color = colors.HexColor("#1E293B")       # Slate 800
        line_color = colors.HexColor("#CBD5E1")       # Slate 300

        name_style = ParagraphStyle(
            "ResumeName",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=primary_color,
            alignment=0,
            spaceAfter=2,
        )

        contact_style = ParagraphStyle(
            "ResumeContact",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#475569"),
            spaceAfter=8,
        )

        heading_style = ParagraphStyle(
            "ResumeSectionHeading",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=secondary_color,
            spaceBefore=8,
            spaceAfter=3,
            textTransform="uppercase",
        )

        subheading_style = ParagraphStyle(
            "ResumeSubheading",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=primary_color,
            spaceBefore=4,
            spaceAfter=2,
        )

        body_style = ParagraphStyle(
            "ResumeBody",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=text_color,
            spaceAfter=4,
        )

        bullet_style = ParagraphStyle(
            "ResumeBullet",
            parent=body_style,
            leftIndent=14,
            firstLineIndent=-10,
            spaceAfter=3,
        )

        story: List[Any] = []
        lines = markdown_text.splitlines()

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            # Header 1: Candidate Name
            if line.startswith("# "):
                name_text = format_markdown_for_reportlab(line[2:].strip())
                story.append(Paragraph(name_text, name_style))
                continue

            # Header 2: Section Titles
            if line.startswith("## "):
                sec_text = format_markdown_for_reportlab(line[3:].strip())
                story.append(Paragraph(sec_text, heading_style))
                story.append(
                    HRFlowable(
                        width="100%",
                        thickness=0.75,
                        color=line_color,
                        spaceBefore=1,
                        spaceAfter=5,
                    )
                )
                continue

            # Header 3: Subheadings (Company / Role / Project)
            if line.startswith("### "):
                sub_text = format_markdown_for_reportlab(line[4:].strip())
                story.append(Paragraph(sub_text, subheading_style))
                continue

            # Bullet points
            if line.startswith("- ") or line.startswith("* "):
                bullet_content = format_markdown_for_reportlab(line[2:].strip())
                formatted_bullet = f"&bull;&nbsp; {bullet_content}"
                story.append(Paragraph(formatted_bullet, bullet_style))
                continue

            # Contact line / metadata below name
            if len(story) <= 2 and ("|" in line or "@" in line or "github" in line.lower()):
                contact_text = format_markdown_for_reportlab(line)
                story.append(Paragraph(contact_text, contact_style))
                continue

            # Regular paragraph text
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

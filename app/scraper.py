import logging
import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.database import SessionLocal, get_existing_recent_links

logger = logging.getLogger("your_assistant.scraper")

SERPAPI_URL = "https://serpapi.com/search.json"
DEFAULT_FALLBACK_QUERY = "Software Engineer remote"

# Common job aggregator domains where the domain is NOT the company's own site
AGGREGATOR_DOMAINS = {
    "linkedin.com",
    "indeed.com",
    "glassdoor.com",
    "ziprecruiter.com",
    "monster.com",
    "dice.com",
    "simplyhired.com",
    "careerbuilder.com",
    "google.com",
    "serpapi.com",
    "upwork.com",
    "fiverr.com",
    "wellfound.com",
}


class ScrapedJob(BaseModel):
    """Normalized data representation of a scraped job listing."""
    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Employer / Company name")
    description: str = Field(default="", description="Job description or snippet")
    job_link: str = Field(..., description="Direct job application or post link")
    career_page_link: str = Field(..., description="Predicted or resolved company career portal URL")


def predict_career_page_url(company: str, job_link: Optional[str] = None) -> str:
    """
    Predict or construct the company's official career page URL.
    
    1. If the job link belongs directly to the employer or a dedicated ATS
       (e.g., jobs.lever.co/company or boards.greenhouse.io/company), extract the portal link.
    2. Otherwise, clean the company name and construct standard conventions (e.g. domain.com/careers).
    """
    # 1. Inspect direct job_link if provided
    if job_link:
        try:
            parsed = urlparse(job_link)
            host = parsed.netloc.lower()
            if host.startswith("www."):
                host = host[4:]

            # Check if it's hosted on standard ATS platforms
            path_parts = [p for p in parsed.path.split("/") if p]
            if "greenhouse.io" in host and path_parts:
                return f"https://boards.greenhouse.io/{path_parts[0]}"
            if "lever.co" in host and path_parts:
                return f"https://jobs.lever.co/{path_parts[0]}"
            if "ashbyhq.com" in host and path_parts:
                return f"https://jobs.ashbyhq.com/{path_parts[0]}"
            if "workday.com" in host:
                return f"{parsed.scheme}://{parsed.netloc}"

            # If not a known generic aggregator, use the job_link's base domain
            is_aggregator = any(host == agg or host.endswith("." + agg) for agg in AGGREGATOR_DOMAINS)
            if not is_aggregator and "." in host:
                return f"{parsed.scheme}://{host}/careers"
        except Exception:
            pass

    # 2. Fallback: Clean company name and predict domain.com/careers
    cleaned_name = re.sub(
        r"\b(inc|incorporated|llc|ltd|limited|corp|corporation|technologies|tech|solutions|co|company)\b",
        "",
        company,
        flags=re.IGNORECASE,
    )
    # Remove punctuation and whitespace
    slug = re.sub(r"[^a-zA-Z0-9]", "", cleaned_name).lower()
    if not slug:
        slug = re.sub(r"[^a-zA-Z0-9]", "", company).lower() or "company"

    return f"https://{slug}.com/careers"


class JobScraper:
    """
    Job scraper using SerpApi Google Jobs engine.
    Supports query prioritization, data extraction, career URL prediction,
    and automated 7-day database deduplication.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SERPAPI_API_KEY", "")

    def build_query(self, job_keyword: Optional[str] = None) -> str:
        """
        Formulate search query based on input keyword:
        - If provided, prioritize exact match with quotes.
        - If blank, fallback to default query.
        """
        if job_keyword and job_keyword.strip():
            cleaned_keyword = job_keyword.strip()
            return f'"{cleaned_keyword}"'
        return DEFAULT_FALLBACK_QUERY

    def _extract_job_link(self, job_dict: Dict[str, Any]) -> Optional[str]:
        """Extract the most direct application link from SerpApi job data."""
        # 1. Try apply_options (often has direct employer / application links)
        apply_options = job_dict.get("apply_options", [])
        if apply_options and isinstance(apply_options, list):
            first_apply = apply_options[0]
            if isinstance(first_apply, dict) and first_apply.get("link"):
                return first_apply["link"]

        # 2. Try related_links
        related_links = job_dict.get("related_links", [])
        if related_links and isinstance(related_links, list):
            first_related = related_links[0]
            if isinstance(first_related, dict) and first_related.get("link"):
                return first_related["link"]

        # 3. Try share_link
        if job_dict.get("share_link"):
            return job_dict["share_link"]

        # 4. Fallback: Google search link using job_id if present
        job_id = job_dict.get("job_id")
        if job_id:
            return f"https://www.google.com/search?ibp=htl;jobs#fpstate=tldetail&htidocid={job_id}"

        return None

    def fetch_serpapi_jobs(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Perform HTTP request to SerpApi Google Jobs engine."""
        if not self.api_key:
            logger.warning("SERPAPI_API_KEY is not configured. Returning empty results.")
            return []

        params = {
            "engine": "google_jobs",
            "q": query,
            "api_key": self.api_key,
            "hl": "en",
        }

        try:
            response = requests.get(SERPAPI_URL, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            jobs = data.get("jobs_results", [])
            return jobs[:limit]
        except requests.RequestException as exc:
            logger.error("Error fetching jobs from SerpApi: %s", exc)
            return []

    def scrape_jobs(
        self,
        job_keyword: Optional[str] = None,
        limit: int = 10,
        db: Optional[Session] = None,
    ) -> List[ScrapedJob]:
        """
        Execute job scraping workflow:
        1. Formulate search query (prioritize job_keyword or fallback).
        2. Fetch results from SerpApi Google Jobs.
        3. Extract fields (title, company, description, link).
        4. Predict company career page URL.
        5. Check database for links saved in the last 7 days and skip duplicates.
        """
        query = self.build_query(job_keyword)
        logger.info("Executing job scrape with query: '%s'", query)

        raw_jobs = self.fetch_serpapi_jobs(query, limit=limit)
        if not raw_jobs:
            logger.info("No raw jobs returned for query: '%s'", query)
            return []

        # Parse candidates
        candidates: List[ScrapedJob] = []
        for item in raw_jobs:
            title = (item.get("title") or "").strip()
            company = (item.get("company_name") or "Unknown Company").strip()
            description = (item.get("description") or "").strip()
            link = self._extract_job_link(item)

            if not title or not link:
                continue

            career_link = predict_career_page_url(company=company, job_link=link)

            candidates.append(
                ScrapedJob(
                    title=title,
                    company=company,
                    description=description,
                    job_link=link,
                    career_page_link=career_link,
                )
            )

        if not candidates:
            return []

        # Duplicate Check: Check links recorded in JobHistory within the last 7 days
        all_links = [c.job_link for c in candidates]
        existing_recent_links = get_existing_recent_links(links=all_links, days=7, db=db)

        filtered_jobs: List[ScrapedJob] = []
        for candidate in candidates:
            if candidate.job_link in existing_recent_links:
                logger.info("Skipping duplicate job link seen in last 7 days: %s", candidate.job_link)
                continue
            filtered_jobs.append(candidate)

        logger.info(
            "Scrape complete. Retrieved %d total, filtered %d duplicates, returning %d unique jobs.",
            len(candidates),
            len(candidates) - len(filtered_jobs),
            len(filtered_jobs),
        )
        return filtered_jobs

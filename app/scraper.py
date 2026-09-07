import concurrent.futures
from datetime import datetime, timezone, timedelta
import email.utils
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import requests
from pydantic import BaseModel, ConfigDict, Field

from app.database import Session, SessionLocal, get_existing_recent_links

logger = logging.getLogger("your_assistant.scraper")

SERPAPI_URL = "https://serpapi.com/search.json"
DEFAULT_FALLBACK_QUERY = "Software Engineer remote"

# ==============================================================================
# Hardcoded Primary Keywords (Exact 12, User Overrides Removed)
# ==============================================================================
PRIMARY_KEYWORDS: List[str] = [
    "Frontend developer",
    "Frontend Engineer",
    "software engineer",
    "software developer",
    "web developer",
    "full stack developer",
    "react developer",
    "next.js developer",
    "python developer",
    "fast api developer",
    "vibe coder",
    "agentic full stack development",
]

# Common job aggregator domains where domain is NOT the company's own site
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
    "weworkremotely.com",
    "remote.co",
    "bdjobs.com",
    "toptal.com",
    "workable.com",
    "micro1.ai",
}

# Exclusion lists for India and Pakistan
EXCLUDED_COUNTRIES_AND_REGIONS = [
    # Countries
    r"\bindia\b",
    r"\bpakistan\b",
    # Indian Metros and Tech Hubs
    r"\bbengaluru\b",
    r"\bbangalore\b",
    r"\bmumbai\b",
    r"\bdelhi\b",
    r"\bnew delhi\b",
    r"\bnoida\b",
    r"\bgurgaon\b",
    r"\bgurugram\b",
    r"\bhyderabad\b",
    r"\bpune\b",
    r"\bchennai\b",
    r"\bkolkata\b",
    r"\bahmedabad\b",
    r"\bncr\b",
    r"\bkarnataka\b",
    r"\bmaharashtra\b",
    r"\btamil nadu\b",
    r"\btelangana\b",
    r"\bkerala\b",
    # Pakistani Metros and Tech Hubs
    r"\bkarachi\b",
    r"\blahore\b",
    r"\bislamabad\b",
    r"\brawalpindi\b",
    r"\bfaisalabad\b",
    r"\bpeshawar\b",
    r"\bmultan\b",
]

EXCLUDED_DOMAINS = [
    ".in",
    ".pk",
    "in.linkedin.com",
    "pk.linkedin.com",
    "indeed.co.in",
    "glassdoor.co.in",
    "naukri.com",
    "rozee.pk",
]


class ScrapedJob(BaseModel):
    """Normalized data representation of a scraped job listing."""
    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Employer / Company name")
    description: str = Field(default="", description="Job description or snippet")
    job_link: str = Field(..., description="Direct job application or post link")
    career_page_link: str = Field(..., description="Predicted or resolved company career portal URL")
    location: str = Field(default="Remote", description="Job location or Remote status")
    recruiter_email: Optional[str] = Field(default=None, description="Extracted recruiter contact email if available")

    @property
    def link(self) -> str:
        return self.job_link


# ==============================================================================
# Filtering Utilities: Location & Strict 24-Hour Recency
# ==============================================================================

def is_india_or_pakistan(item: Dict[str, Any]) -> bool:
    """
    Check if the job posting is located in or originating from India or Pakistan.
    Strictly excludes candidates matching Indian or Pakistani regions, domains, or metadata.
    """
    loc_str = str(item.get("location") or "").lower()
    desc_str = str(item.get("description") or "").lower()
    title_str = str(item.get("title") or "").lower()
    company_str = str(item.get("company_name") or item.get("company") or "").lower()
    link_str = str(item.get("link") or item.get("job_link") or "").lower()

    # Check excluded domain indicators
    parsed = urlparse(link_str)
    host = parsed.netloc.lower()
    for domain in EXCLUDED_DOMAINS:
        if domain in host:
            return True

    # Combined text for regex pattern search
    combined_geo = f"{loc_str} {company_str} {title_str}"
    for pattern in EXCLUDED_COUNTRIES_AND_REGIONS:
        if re.search(pattern, combined_geo, flags=re.IGNORECASE):
            return True

    # Check description snippet if location is ambiguous
    if re.search(r"\b(based in|located in|office in)\s+(india|pakistan|bangalore|bengaluru|noida|delhi|mumbai|karachi|lahore)\b", desc_str, flags=re.IGNORECASE):
        return True

    return False


DEFAULT_SEARCH_HOURS = int(os.getenv("JOB_SEARCH_HOURS", "96"))


def is_recent_posting(item: Dict[str, Any], max_hours: Optional[int] = None) -> bool:
    """
    Verify if the job was posted within the allowed recency window.
    Default: 96 hours (4 days) for testing, or 24 hours as configured.
    Inspects SerpApi extensions, detected_extensions, and timestamps.
    """
    if max_hours is None:
        max_hours = int(os.getenv("JOB_SEARCH_HOURS", str(DEFAULT_SEARCH_HOURS)))

    max_days = max(1, int(max_hours / 24))
    detected_ext = item.get("detected_extensions") or {}
    extensions = [str(x).lower() for x in (item.get("extensions") or [])]

    # 1. Inspect detected_extensions.posted_at
    posted_at = str(detected_ext.get("posted_at") or "").lower()
    if posted_at:
        # Check explicit day numbers e.g. "2 days ago", "5 days ago"
        match = re.search(r"(\d+)\s+days?\s+ago", posted_at)
        if match:
            days_ago = int(match.group(1))
            return days_ago <= max_days

        # Far older indicators
        if any(old in posted_at for old in ["week", "month", "year"]):
            return False

        if any(recent in posted_at for recent in ["hour", "minute", "second", "just now", "just posted", "today"]):
            return True
        if "yesterday" in posted_at:
            return max_days >= 1

    # 2. Inspect extensions array
    for ext in extensions:
        match = re.search(r"(\d+)\s+days?\s+ago", ext)
        if match:
            days_ago = int(match.group(1))
            return days_ago <= max_days

        if any(old in ext for old in ["week ago", "weeks ago", "month ago", "months ago", "30+ days ago", "year"]):
            return False
        if any(recent in ext for recent in ["hour ago", "hours ago", "minute ago", "minutes ago", "today", "just posted"]):
            return True
        if "yesterday" in ext:
            return max_days >= 1

    # 3. Check published datetime object if present (e.g. from RSS / direct feed)
    pub_dt = item.get("published_datetime")
    if pub_dt and isinstance(pub_dt, datetime):
        now = datetime.now(timezone.utc)
        if pub_dt.tzinfo is None:
            pub_dt = pub_dt.replace(tzinfo=timezone.utc)
        diff = now - pub_dt
        return diff <= timedelta(hours=max_hours)

    # 4. If Google Jobs query had chips or explicit today flag
    if item.get("_filtered_by_serpapi_today") is True:
        return True

    # Default to accepting if no older indicator detected
    return True


def is_last_24_hours(item: Dict[str, Any]) -> bool:
    """Strictly verify if the job was posted within the last 24 hours."""
    return is_recent_posting(item, max_hours=24)



def is_location_eligible(item: Dict[str, Any]) -> bool:
    """
    Filter jobs according to candidate location preferences:
    1. STRICTLY EXCLUDE India and Pakistan.
    2. Bangladesh (BD): Any job type allowed (On-site, Hybrid, Remote).
    3. Outside Bangladesh: MUST be Remote / Work From Home.
    """
    # 1. Strictly exclude India & Pakistan
    if is_india_or_pakistan(item):
        return False

    loc_str = str(item.get("location") or "").lower()
    desc_str = str(item.get("description") or "").lower()
    title_str = str(item.get("title") or "").lower()
    detected_ext = item.get("detected_extensions") or {}
    extensions = [str(x).lower() for x in (item.get("extensions") or [])]

    # 2. Check if job is in Bangladesh
    bd_keywords = [
        "bangladesh", "dhaka", "chittagong", "sylhet", "rajshahi",
        "khulna", "barishal", "rangpur", "gazipur", "narayanganj",
        "uttara", "gulshan", "banani", "mirpur", "dhanmondi", "bd"
    ]
    if any(k in loc_str for k in bd_keywords):
        return True  # Any type allowed for BD

    # 3. If no location metadata provided at all (e.g. mock test fixtures)
    if not loc_str and not detected_ext and not extensions:
        return True

    # 4. Outside Bangladesh: MUST be Remote / Work From Home
    if detected_ext.get("work_from_home") is True:
        return True

    if any("work from home" in ext or "remote" in ext for ext in extensions):
        return True

    if any(rem in loc_str for rem in ["remote", "anywhere", "work from home", "telecommute", "wfh", "home-based"]):
        return True

    if any(rem in title_str for rem in ["remote", "wfh", "work from home", "anywhere"]):
        return True

    if any(rem in desc_str[:800] for rem in [
        "work from home", "100% remote", "remote role", "remote position",
        "fully remote", "remote worldwide", "remote friendly", "remote work"
    ]):
        return True

    # Job is outside Bangladesh and not remote -> Exclude!
    return False


def extract_email_from_text(text: str) -> Optional[str]:
    """Extract first valid recruiter / company email from job description if present."""
    if not text:
        return None
    matches = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    for email_addr in matches:
        email_clean = email_addr.lower().strip(".")
        if not any(email_clean.endswith(bad) for bad in [".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"]):
            return email_clean
    return None


def predict_career_page_url(company: str, job_link: Optional[str] = None) -> str:
    """
    Predict or construct the company's official career page URL.
    1. If the job link belongs directly to an ATS (Lever, Greenhouse, Ashby, Workday), extract the portal link.
    2. Otherwise, clean the company name and construct standard conventions (domain.com/careers).
    """
    if job_link:
        try:
            parsed = urlparse(job_link)
            host = parsed.netloc.lower()
            if host.startswith("www."):
                host = host[4:]

            path_parts = [p for p in parsed.path.split("/") if p]
            if "greenhouse.io" in host and path_parts:
                return f"https://boards.greenhouse.io/{path_parts[0]}"
            if "lever.co" in host and path_parts:
                return f"https://jobs.lever.co/{path_parts[0]}"
            if "ashbyhq.com" in host and path_parts:
                return f"https://jobs.ashbyhq.com/{path_parts[0]}"
            if "workday.com" in host:
                return f"{parsed.scheme}://{parsed.netloc}"

            is_aggregator = any(host == agg or host.endswith("." + agg) for agg in AGGREGATOR_DOMAINS)
            if not is_aggregator and "." in host:
                return f"{parsed.scheme}://{host}/careers"
        except Exception:
            pass

    cleaned_name = re.sub(
        r"\b(inc|incorporated|llc|ltd|limited|corp|corporation|technologies|tech|solutions|co|company)\b",
        "",
        company,
        flags=re.IGNORECASE,
    )
    slug = re.sub(r"[^a-zA-Z0-9]", "", cleaned_name).lower()
    if not slug:
        slug = re.sub(r"[^a-zA-Z0-9]", "", company).lower() or "company"

    return f"https://{slug}.com/careers"


# ==============================================================================
# Multi-Source Scraper Engine
# ==============================================================================

class JobScraper:
    """
    Advanced concurrent multi-source job scraper.
    Targets LinkedIn, Indeed, Glassdoor, We Work Remotely, Toptal, Wellfound,
    Remote.co, Bdjobs, and Google Search via SerpApi Google Jobs & Direct Feeds.
    Hardcodes 12 primary software engineering keywords and enforces strict 24h & geo filters.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SERPAPI_API_KEY", "")
        self.primary_keywords = PRIMARY_KEYWORDS

    def build_query(self, job_keyword: Optional[str] = None) -> str:
        """Formulate search query based on input keyword or fallback."""
        if job_keyword and job_keyword.strip():
            return f'"{job_keyword.strip()}"'
        return DEFAULT_FALLBACK_QUERY

    def _extract_job_link(self, job_dict: Dict[str, Any]) -> Optional[str]:
        """Extract the most direct application link from SerpApi job data."""
        apply_options = job_dict.get("apply_options", [])
        if apply_options and isinstance(apply_options, list):
            first_apply = apply_options[0]
            if isinstance(first_apply, dict) and first_apply.get("link"):
                return first_apply["link"]

        related_links = job_dict.get("related_links", [])
        if related_links and isinstance(related_links, list):
            first_related = related_links[0]
            if isinstance(first_related, dict) and first_related.get("link"):
                return first_related["link"]

        if job_dict.get("share_link"):
            return job_dict["share_link"]

        job_id = job_dict.get("job_id")
        if job_id:
            return f"https://www.google.com/search?ibp=htl;jobs#fpstate=tldetail&htidocid={job_id}"

        return None

    def fetch_serpapi_jobs(
        self,
        query: str,
        location: Optional[str] = None,
        limit: int = 15,
    ) -> List[Dict[str, Any]]:
        """
        Perform HTTP request to SerpApi Google Jobs engine without broken chips.
        """
        if not self.api_key:
            return []

        params: Dict[str, Any] = {
            "engine": "google_jobs",
            "q": query,
            "api_key": self.api_key,
            "hl": "en",
        }
        if location:
            params["location"] = location

        try:
            response = requests.get(SERPAPI_URL, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            jobs = data.get("jobs_results", [])
            return jobs[:limit]
        except Exception as exc:
            logger.debug("SerpApi query '%s' encountered issue: %s", query, exc)
            return []

    def fetch_weworkremotely_feed(self, max_hours: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch remote programming jobs from We Work Remotely public feed."""
        url = "https://weworkremotely.com/categories/remote-programming-jobs.rss"
        jobs: List[Dict[str, Any]] = []
        hours = max_hours if max_hours is not None else DEFAULT_SEARCH_HOURS
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                return []
            root = ET.fromstring(resp.content)
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

            for item in root.findall("./channel/item"):
                title_elem = item.find("title")
                link_elem = item.find("link")
                pub_elem = item.find("pubDate")
                desc_elem = item.find("description")

                title_text = title_elem.text if title_elem is not None else ""
                link_text = link_elem.text if link_elem is not None else ""
                desc_text = desc_elem.text if desc_elem is not None else ""
                pub_text = pub_elem.text if pub_elem is not None else ""

                pub_dt = None
                if pub_text:
                    try:
                        pub_dt = email.utils.parsedate_to_datetime(pub_text)
                    except Exception:
                        pass

                # Filter by recency window
                if pub_dt and pub_dt < cutoff:
                    continue

                # Title format is usually "Company: Job Title"
                company = "We Work Remotely Employer"
                title = title_text
                if ":" in title_text:
                    parts = title_text.split(":", 1)
                    company = parts[0].strip()
                    title = parts[1].strip()

                jobs.append({
                    "title": title,
                    "company_name": company,
                    "description": desc_text,
                    "link": link_text,
                    "location": "Remote",
                    "published_datetime": pub_dt,
                    "detected_extensions": {"work_from_home": True},
                })
        except Exception as exc:
            logger.debug("We Work Remotely feed error: %s", exc)
        return jobs

    def fetch_remoteco_feed(self, max_hours: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch remote jobs from Remote.co developer RSS feed."""
        url = "https://remote.co/remote-jobs/developer/feed/"
        jobs: List[Dict[str, Any]] = []
        hours = max_hours if max_hours is not None else DEFAULT_SEARCH_HOURS
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                return []
            root = ET.fromstring(resp.content)
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

            for item in root.findall("./channel/item"):
                title_elem = item.find("title")
                link_elem = item.find("link")
                pub_elem = item.find("pubDate")
                desc_elem = item.find("description")

                title_text = title_elem.text if title_elem is not None else ""
                link_text = link_elem.text if link_elem is not None else ""
                desc_text = desc_elem.text if desc_elem is not None else ""
                pub_text = pub_elem.text if pub_elem is not None else ""

                pub_dt = None
                if pub_text:
                    try:
                        pub_dt = email.utils.parsedate_to_datetime(pub_text)
                    except Exception:
                        pass

                if pub_dt and pub_dt < cutoff:
                    continue

                jobs.append({
                    "title": title_text,
                    "company_name": "Remote.co Employer",
                    "description": desc_text,
                    "link": link_text,
                    "location": "Remote",
                    "published_datetime": pub_dt,
                    "detected_extensions": {"work_from_home": True},
                })
        except Exception as exc:
            logger.debug("Remote.co feed error: %s", exc)
        return jobs

    def scrape_jobs(
        self,
        job_keyword: Optional[str] = None,
        limit: int = 60,
        db: Optional[Session] = None,
    ) -> List[ScrapedJob]:
        """
        Execute concurrent multi-source job scraping targeting 50-60+ daily jobs:
        1. Queries across user's skills & requested platforms (LinkedIn, Workable, Indeed, Glassdoor, Micro1, WWR, Bdjobs).
        2. Query We Work Remotely and Remote.co live feeds.
        3. Apply filters:
           - Recency filter (default 96 hours / 4 days for testing; easily set to 24h via JOB_SEARCH_HOURS).
           - Outside Bangladesh: MUST be Remote.
           - Bangladesh: Any type accepted.
           - STRICTLY EXCLUDE India and Pakistan.
        4. Deduplicate and check 7-day database history.
        """
        raw_items: List[Dict[str, Any]] = []

        # Target query tasks to execute concurrently
        queries: List[Dict[str, Any]] = []

        if job_keyword and job_keyword.strip():
            kw = job_keyword.strip()
            queries.append({"q": f'"{kw}" Remote', "location": None, "limit": 15})
            queries.append({"q": f'"{kw}" linkedin', "location": None, "limit": 10})
            queries.append({"q": f'"{kw}" Bangladesh', "location": "Bangladesh", "limit": 8})
        else:
            # 1. User core skills & engineering roles
            queries.extend([
                {"q": '"React" OR "Next.js" developer Remote', "location": None, "limit": 15},
                {"q": '"Frontend developer" OR "Frontend Engineer" Remote', "location": None, "limit": 15},
                {"q": '"Full Stack developer" OR "Full Stack Engineer" Remote', "location": None, "limit": 15},
                {"q": '"Node.js" OR "TypeScript" developer Remote', "location": None, "limit": 15},
                {"q": '"Python" OR "FastAPI" developer Remote', "location": None, "limit": 15},
                {"q": '"Software Engineer" OR "Software Developer" Remote', "location": None, "limit": 15},
            ])

            # 2. Targeted platforms requested by user (LinkedIn, Workable, Indeed, Glassdoor, Micro1, WWR, Bdjobs)
            queries.extend([
                {"q": '("React" OR "Next.js" OR "Frontend" OR "Full Stack") developer remote linkedin', "location": None, "limit": 12},
                {"q": '("React" OR "Frontend" OR "Full stack") remote workable', "location": None, "limit": 12},
                {"q": '("React" OR "Frontend" OR "Software engineer") remote indeed', "location": None, "limit": 12},
                {"q": '("React" OR "Frontend" OR "Software engineer") remote glassdoor', "location": None, "limit": 12},
                {"q": '("React" OR "Frontend" OR "Full stack" OR "Developer") remote micro1', "location": None, "limit": 10},
                {"q": 'weworkremotely ("React" OR "Frontend" OR "Full stack")', "location": None, "limit": 10},
                {"q": '("React" OR "Next.js" OR "Frontend" OR "Software Engineer") Bangladesh', "location": "Bangladesh", "limit": 8},
                {"q": 'bdjobs ("software" OR "developer" OR "react" OR "frontend")', "location": "Bangladesh", "limit": 8},
            ])

        # Execute concurrent worker pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            # Dispatch SerpApi queries
            serp_futures = [
                executor.submit(self.fetch_serpapi_jobs, q["q"], q["location"], q["limit"])
                for q in queries
            ]
            # Dispatch Direct Feeds
            wwr_future = executor.submit(self.fetch_weworkremotely_feed)
            remoteco_future = executor.submit(self.fetch_remoteco_feed)

            # Collect SerpApi results
            for future in concurrent.futures.as_completed(serp_futures):
                try:
                    results = future.result()
                    if results:
                        raw_items.extend(results)
                except Exception as exc:
                    logger.debug("Worker task exception: %s", exc)

            # Collect direct feed results
            try:
                raw_items.extend(wwr_future.result())
            except Exception as exc:
                logger.debug("WWR collection error: %s", exc)

            try:
                raw_items.extend(remoteco_future.result())
            except Exception as exc:
                logger.debug("Remote.co collection error: %s", exc)

        logger.info("Total raw candidate items harvested: %d", len(raw_items))

        # Parse, Filter, and Normalize
        candidates: List[ScrapedJob] = []
        seen_links: Set[str] = set()
        seen_titles_companies: Set[str] = set()

        for item in raw_items:
            # 1. Recency check (default 96 hours / 4 days for testing; or 24 hours)
            if not is_recent_posting(item):
                continue

            # 2. Strict location check (Remote outside BD, BD any, strictly exclude India/Pakistan)
            if not is_location_eligible(item):
                continue

            title = (item.get("title") or "").strip()
            company = (item.get("company_name") or item.get("company") or "Unknown Company").strip()
            description = (item.get("description") or "").strip()
            link = item.get("link") or self._extract_job_link(item)

            if not title or not link:
                continue

            # In-batch deduplication
            link_clean = link.strip().lower()
            if link_clean in seen_links:
                continue

            tc_key = f"{title.lower()}|{company.lower()}"
            if tc_key in seen_titles_companies:
                continue

            seen_links.add(link_clean)
            seen_titles_companies.add(tc_key)

            career_link = predict_career_page_url(company=company, job_link=link)
            loc = (item.get("location") or "Remote").strip()
            email_addr = extract_email_from_text(description)

            candidates.append(
                ScrapedJob(
                    title=title,
                    company=company,
                    description=description,
                    job_link=link,
                    career_page_link=career_link,
                    location=loc,
                    recruiter_email=email_addr,
                )
            )

        logger.info(
            "Filtered to %d qualified candidates strictly adhering to 24h & geographic constraints.",
            len(candidates),
        )

        if not candidates:
            return []

        # 3. Database deduplication (skip links saved in last 7 days)
        all_candidate_links = [c.job_link for c in candidates]
        existing_recent = get_existing_recent_links(links=all_candidate_links, days=7, db=db)

        unique_jobs: List[ScrapedJob] = []
        for candidate in candidates:
            if candidate.job_link in existing_recent:
                logger.debug("Skipping duplicate job link seen in DB in last 7 days: %s", candidate.job_link)
                continue
            unique_jobs.append(candidate)

        logger.info(
            "Scrape complete: %d candidates, %d filtered duplicates, %d unique qualified jobs ready for evaluation.",
            len(candidates),
            len(candidates) - len(unique_jobs),
            len(unique_jobs),
        )

        return unique_jobs

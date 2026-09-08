import logging
import os
from pathlib import Path
import shutil
from datetime import date, datetime, timedelta, timezone
import importlib
from typing import Any, Generator, Optional

from sqlalchemy import create_engine

# Dynamically import sqlalchemy.orm to maintain full static analyzer compatibility across Python environments
_orm: Any = importlib.import_module("sqlalchemy.orm")
declarative_base = _orm.declarative_base
sessionmaker = _orm.sessionmaker
Session = _orm.Session

logger = logging.getLogger("your_assistant.database")

is_vercel = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))
default_db_url = "sqlite:////tmp/jobs.db" if is_vercel else "sqlite:///./jobs.db"
DATABASE_URL = os.getenv("DATABASE_URL", default_db_url).strip()

# On Vercel / serverless, filesystem outside /tmp is strictly read-only.
# SQLite must always write to /tmp/jobs.db. If jobs.db exists in repo, copy it over.
if is_vercel and DATABASE_URL.startswith("sqlite"):
    tmp_db = Path("/tmp/jobs.db")
    if not tmp_db.exists():
        repo_db = Path(__file__).resolve().parent.parent / "jobs.db"
        if repo_db.exists():
            try:
                shutil.copy2(repo_db, tmp_db)
                logger.info("Copied repository seeded jobs.db to writable /tmp/jobs.db")
            except Exception as exc:
                logger.warning("Could not copy bundled jobs.db to /tmp: %s", exc)
    DATABASE_URL = "sqlite:////tmp/jobs.db"

# Normalize Heroku/Render legacy postgres:// to postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Configure dialect-specific engine parameters
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=False,
    )
    logger.info("Configured SQLite database engine: %s", DATABASE_URL)
else:
    # Production PostgreSQL or MySQL connection pooling
    engine = create_engine(
        DATABASE_URL,
        pool_size=5 if is_vercel else 10,
        max_overflow=10 if is_vercel else 20,
        pool_pre_ping=True,
        pool_recycle=1800,
        echo=False,
    )
    # Mask password for logging
    masked_url = DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL.split("://")[0]
    logger.info("Configured production SQL database engine connected to: ...@%s", masked_url)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def delete_records_older_than(days: int = 7, db: Optional[Session] = None) -> int:
    """
    Automated cleanup function to delete Job records older than the specified retention period.
    """
    from app.models import Job

    should_close_session = False
    if db is None:
        db = SessionLocal()
        should_close_session = True

    try:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        logger.info("Executing retention cleanup: deleting records older than %s (%d days)", cutoff_date.isoformat(), days)

        deleted_count = (
            db.query(Job)
            .filter(Job.created_at < cutoff_date)
            .delete(synchronize_session=False)
        )
        db.commit()
        logger.info("Retention cleanup completed: %d expired job records deleted.", deleted_count)
        return deleted_count
    except Exception as exc:
        db.rollback()
        logger.error("Failed to execute retention cleanup: %s", exc, exc_info=True)
        raise
    finally:
        if should_close_session:
            db.close()


def delete_jobs_by_date(target_date: date, db: Optional[Session] = None) -> int:
    """
    Delete all scraped jobs for a specific date (YYYY-MM-DD).
    
    :param target_date: The date to clean up.
    :param db: Optional SQLAlchemy Session.
    :return: Number of deleted records.
    """
    from app.models import Job

    should_close_session = False
    if db is None:
        db = SessionLocal()
        should_close_session = True

    try:
        deleted_count = (
            db.query(Job)
            .filter(Job.scraped_date == target_date)
            .delete(synchronize_session=False)
        )
        db.commit()
        logger.info("Deleted %d job records for date %s", deleted_count, target_date.isoformat())
        return deleted_count
    except Exception as exc:
        db.rollback()
        logger.error("Failed to delete jobs for date %s: %s", target_date, exc, exc_info=True)
        raise
    finally:
        if should_close_session:
            db.close()


def get_existing_recent_links(links: list[str], days: int = 7, db: Optional[Session] = None) -> set[str]:
    """
    Check a list of job URLs against JobHistory records saved in the last `days` days.
    
    :param links: List of job URLs to inspect.
    :param days: Retention lookback window in days (default: 7).
    :param db: Optional SQLAlchemy Session.
    :return: Set of URLs that already exist within the retention window.
    """
    if not links:
        return set()

    from app.models import Job

    should_close_session = False
    if db is None:
        db = SessionLocal()
        should_close_session = True

    try:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        existing_rows = (
            db.query(Job.link)
            .filter(
                Job.created_at >= cutoff_date,
                Job.link.in_(links),
            )
            .all()
        )
        return {row[0] for row in existing_rows}
    finally:
        if should_close_session:
            db.close()


def is_job_link_recent(job_link: str, days: int = 7, db: Optional[Session] = None) -> bool:
    """
    Check if a specific job link exists in Job records within the last `days` days.
    """
    existing = get_existing_recent_links([job_link], days=days, db=db)
    return job_link in existing



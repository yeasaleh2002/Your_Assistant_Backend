import logging
import os
from datetime import datetime, timedelta, timezone
import importlib
from typing import Any, Generator, Optional

from sqlalchemy import create_engine

# Dynamically import sqlalchemy.orm to maintain full static analyzer compatibility across Python environments
_orm: Any = importlib.import_module("sqlalchemy.orm")
declarative_base = _orm.declarative_base
sessionmaker = _orm.sessionmaker
Session = _orm.Session

logger = logging.getLogger("your_assistant.database")

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./jobs.db")

# SQLite requires check_same_thread=False for multi-threaded access in FastAPI
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    echo=False,
)

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
    Automated cleanup function to delete JobHistory records older than the specified retention period.
    
    Can be called directly by background schedulers (creates its own session)
    or invoked with an existing session.
    
    :param days: Retention threshold in days (default: 7).
    :param db: Optional SQLAlchemy Session.
    :return: Number of deleted records.
    """
    # Deferred import to prevent circular dependency with models.py
    from app.models import JobHistory

    should_close_session = False
    if db is None:
        db = SessionLocal()
        should_close_session = True

    try:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        logger.info("Executing retention cleanup: deleting records older than %s (%d days)", cutoff_date.isoformat(), days)

        deleted_count = (
            db.query(JobHistory)
            .filter(JobHistory.created_at < cutoff_date)
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

    from app.models import JobHistory

    should_close_session = False
    if db is None:
        db = SessionLocal()
        should_close_session = True

    try:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        existing_rows = (
            db.query(JobHistory.job_link)
            .filter(
                JobHistory.created_at >= cutoff_date,
                JobHistory.job_link.in_(links),
            )
            .all()
        )
        return {row[0] for row in existing_rows}
    finally:
        if should_close_session:
            db.close()


def is_job_link_recent(job_link: str, days: int = 7, db: Optional[Session] = None) -> bool:
    """
    Check if a specific job link exists in JobHistory within the last `days` days.
    """
    existing = get_existing_recent_links([job_link], days=days, db=db)
    return job_link in existing


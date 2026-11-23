"""
Database layer for Resume Tailor Agent using SQLAlchemy.
Configured for SQLite by default, Postgres-ready via the DATABASE_URL environment variable.
"""
import os
import json
from datetime import datetime
from typing import Generator, Dict, Any, List, Optional
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship

from app.config import settings

# Engine configuration: SQLite needs check_same_thread=False for FastAPI concurrency
connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class User(Base):
    """Represents a student or job seeker using the application."""
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")
    resumes = relationship("Resume", back_populates="user", cascade="all, delete-orphan")
    preferences = relationship("UserPreference", back_populates="user", cascade="all, delete-orphan")
    summaries = relationship("SessionSummary", back_populates="user", cascade="all, delete-orphan")


class ChatSession(Base):
    """Tracks a distinct conversation session."""
    __tablename__ = "sessions"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    status = Column(String, default="active")
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="sessions")
    traces = relationship("StepTrace", back_populates="session", cascade="all, delete-orphan")


class Resume(Base):
    """Metadata and raw/parsed contents of an uploaded resume."""
    __tablename__ = "resumes"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    raw_text = Column(Text, nullable=True)
    parsed_profile_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="resumes")


class StepTrace(Base):
    """
    Structured step-by-step audit log for the hand-written agent loop.
    Logs tool name, latency, token consumption, and errors for full observability.
    """
    __tablename__ = "step_traces"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False, index=True)
    step = Column(Integer, nullable=False)
    tool = Column(String, nullable=False)
    args_json = Column(Text, nullable=False)
    latency_ms = Column(Float, default=0.0)
    tokens = Column(Integer, default=0)
    error = Column(Text, nullable=True)
    path_used = Column(String, default="tool_calling")  # 'tool_calling' or 'deterministic_fallback'
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="traces")


class UserPreference(Base):
    """Cross-session user preference store (e.g. target role, tone, preferred domains)."""
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    key = Column(String, nullable=False)
    value = Column(String, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="preferences")


class SessionSummary(Base):
    """Stores short 3-5 line summaries of previous conversations to preserve context cleanly."""
    __tablename__ = "session_summaries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    summary_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="summaries")


class JDCache(Base):
    """
    Caches parsed Job Description requirements by the SHA-256 hash of the JD text.
    Why: Prevents redundant LLM parsing calls when comparing multiple resumes to the same JD.
    """
    __tablename__ = "jd_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    text_hash = Column(String, unique=True, index=True, nullable=False)
    requirements_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db() -> None:
    """Create all tables in the database if they do not exist."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for obtaining a thread-safe database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def delete_user_data(user_id: str, db: Session) -> Dict[str, Any]:
    """
    Completely purges all data belonging to a user (GDPR / personal privacy requirement).
    Deletes resume files from disk, step traces, sessions, preferences, and summaries.
    """
    resumes = db.query(Resume).filter(Resume.user_id == user_id).all()
    files_deleted = 0
    for r in resumes:
        if r.file_path and os.path.exists(r.file_path):
            try:
                os.remove(r.file_path)
                files_deleted += 1
            except OSError:
                pass

    user = db.query(User).filter(User.id == user_id).first()
    if user:
        db.delete(user)
        db.commit()
        return {"status": "success", "user_id": user_id, "files_deleted": files_deleted}
    return {"status": "not_found", "user_id": user_id, "files_deleted": files_deleted}

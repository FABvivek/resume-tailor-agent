"""
Cross-session memory management for Resume Tailor Agent.
Handles:
1. User preferences (e.g. target role, tone, domain interest).
2. Cross-session summaries: generating 3-5 line summaries at session conclusion
   and loading the most recent 3 summaries at session start instead of unbounded history.
"""
import logging
from typing import Dict, List, Optional
from sqlalchemy.orm import Session

from app.db import UserPreference, SessionSummary, User
from app.llm import chat

logger = logging.getLogger(__name__)


def save_preference(user_id: str, key: str, value: str, db: Session) -> Dict[str, str]:
    """
    Saves or updates a persistent user preference tag.
    Ensures user record exists first.
    """
    # Ensure user exists
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        user = User(id=user_id)
        db.add(user)
        db.flush()

    pref = db.query(UserPreference).filter(
        UserPreference.user_id == user_id,
        UserPreference.key == key
    ).first()

    if pref:
        pref.value = value
    else:
        pref = UserPreference(user_id=user_id, key=key, value=value)
        db.add(pref)

    db.commit()
    logger.info(f"Saved preference for user '{user_id}': {key}={value}")
    return {"key": key, "value": value}


def recall_preferences(user_id: str, db: Session) -> Dict[str, str]:
    """
    Retrieves all stored preference key-value pairs for a user.
    """
    prefs = db.query(UserPreference).filter(UserPreference.user_id == user_id).all()
    return {p.key: p.value for p in prefs}


def save_session_summary(session_id: str, user_id: str, summary_text: str, db: Session) -> SessionSummary:
    """
    Persists a concise 3-5 line session summary into the database.
    """
    # Ensure user exists
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        user = User(id=user_id)
        db.add(user)
        db.flush()

    summary = SessionSummary(session_id=session_id, user_id=user_id, summary_text=summary_text)
    db.add(summary)
    db.commit()
    db.refresh(summary)
    logger.info(f"Stored session summary for session '{session_id}', user '{user_id}'.")
    return summary


def get_recent_summaries(user_id: str, db: Session, limit: int = 3) -> List[str]:
    """
    Loads the last N session summaries for a user in chronological order.
    Why: Injects context without blowing through model context limits or token budgets.
    """
    records = (
        db.query(SessionSummary)
        .filter(SessionSummary.user_id == user_id)
        .order_by(SessionSummary.created_at.desc())
        .limit(limit)
        .all()
    )
    # Return chronologically (oldest to newest among the recent slice)
    return [r.summary_text for r in reversed(records)]


def generate_session_summary(
    conversation_text: str,
    session_id: str,
    user_id: str,
    db: Session
) -> str:
    """
    Invokes the LLM to synthesize a concise 3-5 line summary of the session:
    - Candidate's target role or focus
    - Key gaps identified
    - Rewrites or action items agreed upon
    Stores the summary directly into the database.
    """
    prompt = (
        "Summarize the following resume tailoring session in 3 to 5 concise bullet points.\n"
        "Focus on: target role, main skills matched/missing, key rewrites discussed, and learning plan.\n\n"
        f"Session Transcript:\n{conversation_text}\n\n"
        "Summary:"
    )

    try:
        resp = chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        summary_text = (resp.content or "Tailoring session completed.").strip()
    except Exception as exc:
        logger.warning(f"Could not generate LLM session summary ({exc}); using fallback.")
        summary_text = "Session completed: analyzed resume against job requirements and suggested grounded rewrites."

    save_session_summary(session_id=session_id, user_id=user_id, summary_text=summary_text, db=db)
    return summary_text

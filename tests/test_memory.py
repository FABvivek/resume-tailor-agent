"""
Tests for cross-session memory in app.memory.
Asserts that preferences and session summaries persist and reload across two separate sessions.
"""
from app.memory import (
    save_preference, recall_preferences,
    save_session_summary, get_recent_summaries
)
from app.db import User


def test_user_preferences_crud(db_session):
    user_id = "test_candidate_1"

    # 1. Save preferences
    save_preference(user_id=user_id, key="target_role", value="Backend Engineer", db=db_session)
    save_preference(user_id=user_id, key="preferred_framework", value="FastAPI", db=db_session)

    # 2. Recall preferences
    prefs = recall_preferences(user_id=user_id, db=db_session)
    assert prefs["target_role"] == "Backend Engineer"
    assert prefs["preferred_framework"] == "FastAPI"

    # 3. Update existing preference
    save_preference(user_id=user_id, key="target_role", value="AI Engineer", db=db_session)
    updated_prefs = recall_preferences(user_id=user_id, db=db_session)
    assert updated_prefs["target_role"] == "AI Engineer"


def test_memory_across_two_sessions(db_session):
    """Simulates session 1 saving context, and session 2 recalling the last 3 summaries and preferences."""
    user_id = "multi_session_user"

    # --- SESSION 1 ---
    session_1_id = "sess_001"
    save_preference(user_id=user_id, key="tone", value="concise", db=db_session)
    save_session_summary(
        session_id=session_1_id,
        user_id=user_id,
        summary_text="Candidate analyzed backend intern JD. Identified missing Docker experience.",
        db=db_session
    )

    # --- SESSION 2 ---
    session_2_id = "sess_002"
    # Recall preferences established in Session 1
    recalled_prefs = recall_preferences(user_id=user_id, db=db_session)
    assert recalled_prefs.get("tone") == "concise"

    # Retrieve prior summaries (should contain Session 1 summary)
    summaries = get_recent_summaries(user_id=user_id, db=db_session, limit=3)
    assert len(summaries) == 1
    assert "missing Docker experience" in summaries[0]

    # Session 2 appends another summary
    save_session_summary(
        session_id=session_2_id,
        user_id=user_id,
        summary_text="Candidate added Docker project to resume. Prepared bullet rewrites.",
        db=db_session
    )

    # Subsequent check shows both summaries
    all_summaries = get_recent_summaries(user_id=user_id, db=db_session, limit=3)
    assert len(all_summaries) == 2
    assert "Docker project to resume" in all_summaries[1]

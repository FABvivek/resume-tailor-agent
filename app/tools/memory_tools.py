"""
Tool: memory_tools
Provides callable tool functions for the hand-written agent loop
to record and recall user preferences across sessions.
"""
from typing import Dict, Any
from sqlalchemy.orm import Session

from app.memory import save_preference as _save_pref, recall_preferences as _recall_pref


def save_user_preference(user_id: str, key: str, value: str, db: Session) -> Dict[str, Any]:
    """
    Saves a persistent preference for the user (e.g. role focus, preferred frameworks, target company).
    """
    res = _save_pref(user_id=user_id, key=key, value=value, db=db)
    return {"status": "saved", "preference": res}


def recall_user_preferences(user_id: str, db: Session) -> Dict[str, Any]:
    """
    Recalls all stored preferences for the user.
    """
    prefs = _recall_pref(user_id=user_id, db=db)
    return {"status": "retrieved", "preferences": prefs}

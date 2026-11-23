"""
Tests for JD caching in app.tools.analyze_jd.
Asserts that identical JD text triggers a cache hit, preventing redundant LLM invocations.
"""
from unittest.mock import patch
from app.schemas import Requirement, NormalizedResponse
from app.tools.analyze_jd import analyze_jd, compute_jd_hash
from app.db import JDCache


def test_analyze_jd_cache_hit_and_miss(db_session):
    jd_text = (
        "Role: Backend Intern.\n"
        "Must have: Python, FastAPI, and SQL.\n"
        "Nice to have: Docker and Redis."
    )

    expected_llm_json = """{
        "requirements": [
            {"text": "Python and FastAPI", "importance": "must"},
            {"text": "SQL databases", "importance": "must"},
            {"text": "Docker and Redis", "importance": "nice"}
        ]
    }"""

    mock_response = NormalizedResponse(
        content=expected_llm_json,
        prompt_tokens=50,
        completion_tokens=30,
        total_tokens=80
    )

    # First call: Cache miss -> triggers chat()
    with patch("app.tools.analyze_jd.chat", return_value=mock_response) as mock_chat:
        reqs1 = analyze_jd(text=jd_text, db=db_session)
        assert len(reqs1) == 3
        assert reqs1[0].text == "Python and FastAPI"
        assert mock_chat.call_count == 1

        # Check entry written into SQLite JDCache
        h = compute_jd_hash(jd_text)
        cached_row = db_session.query(JDCache).filter(JDCache.text_hash == h).first()
        assert cached_row is not None

    # Second call with the same JD text: Cache hit -> chat() MUST NOT be called!
    with patch("app.tools.analyze_jd.chat") as mock_chat_second:
        reqs2 = analyze_jd(text=jd_text, db=db_session)
        assert len(reqs2) == 3
        assert reqs2[0].text == "Python and FastAPI"
        # Zero LLM invocations on cache hit
        assert mock_chat_second.call_count == 0

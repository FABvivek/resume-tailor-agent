"""
Tests for the hand-written agent loop in app.agent.
Asserts:
1. Native tool-calling path executes tools and logs audit traces to StepTrace.
2. Deterministic fallback pipeline triggers seamlessly when a local model emits no tools on step 0.
3. Budget guardrails (MAX_STEPS / MAX_SESSION_TOKENS) terminate execution cleanly.
"""
from unittest.mock import patch
from app.schemas import ChatRequest, NormalizedResponse, ToolCall
from app.agent import run_agent_loop
from app.db import StepTrace, Resume


def test_agent_tool_calling_path(db_session):
    """Simulates model emitting a tool call on turn 1, then returning final response on turn 2."""
    user_id = "agent_user_1"
    session_id = "sess_tools_101"

    # Pre-register a resume in DB
    res = Resume(
        id="file_101",
        user_id=user_id,
        filename="resume.txt",
        file_path="dummy.txt",
        parsed_profile_json='{"name": "Sam", "skills": ["Python"], "projects": [], "education": []}'
    )
    db_session.add(res)
    db_session.commit()

    # Turn 1: Model requests parse_resume tool
    call_1 = NormalizedResponse(
        content="",
        tool_calls=[ToolCall(id="tc_1", name="parse_resume", arguments={"file_id": "file_101"})],
        total_tokens=100
    )
    # Turn 2: Model emits final response
    call_2 = NormalizedResponse(
        content="I have parsed your profile. Your Python experience is noted.",
        tool_calls=[],
        total_tokens=150
    )

    with patch("app.agent.chat", side_effect=[call_1, call_2]), \
         patch("app.agent.generate_session_summary", return_value="Session completed."):
        req = ChatRequest(
            session_id=session_id,
            user_id=user_id,
            message="Check my resume.",
            file_id="file_101"
        )
        resp = run_agent_loop(request=req, db=db_session)

        assert "I have parsed your profile" in resp.response
        assert resp.path_used == "tool_calling"

        # Verify step trace logged in SQLite table
        traces = db_session.query(StepTrace).filter(StepTrace.session_id == session_id).all()
        assert len(traces) == 1
        assert traces[0].tool == "parse_resume"
        assert traces[0].step == 1
        assert traces[0].path_used == "tool_calling"


def test_agent_deterministic_fallback_path(db_session):
    """
    Simulates a small local model that emits plain text on turn 1 without tool calling.
    Agent must fall back to deterministic pipeline (parse -> analyze -> match -> rewrite).
    """
    user_id = "agent_user_2"
    session_id = "sess_fallback_102"

    # Register resume in DB
    res = Resume(
        id="file_102",
        user_id=user_id,
        filename="resume.txt",
        file_path="dummy.txt",
        parsed_profile_json='{"name": "Sam", "skills": ["Python"], "projects": [], "education": []}'
    )
    db_session.add(res)
    db_session.commit()

    # Step 0 chat returns plain text with zero tool calls
    blank_call = NormalizedResponse(
        content="Hello, I will look at your resume.",
        tool_calls=[],
        total_tokens=50
    )

    with patch("app.agent.chat", return_value=blank_call), \
         patch("app.agent.generate_session_summary", return_value="Session completed."):
        with patch("app.tools.analyze_jd.analyze_jd") as mock_jd:
            from app.schemas import Requirement
            mock_jd.return_value = [Requirement(text="Python 3", importance="must")]

            req = ChatRequest(
                session_id=session_id,
                user_id=user_id,
                message="Analyze my resume for this JD.",
                file_id="file_102",
                jd_text="Must know Python 3."
            )
            resp = run_agent_loop(request=req, db=db_session)

            assert resp.path_used == "deterministic_fallback"
            assert "Skill Gap Overview" in resp.response

            # Check that StepTrace recorded deterministic steps
            traces = db_session.query(StepTrace).filter(StepTrace.session_id == session_id).all()
            assert len(traces) >= 2
            assert all(t.path_used == "deterministic_fallback" for t in traces)


def test_agent_budget_guard(db_session):
    """Asserts that agent terminates cleanly if token budget is exceeded."""
    user_id = "agent_user_3"
    session_id = "sess_budget_103"

    huge_token_call = NormalizedResponse(
        content="",
        tool_calls=[ToolCall(id="tc_1", name="recall_preferences", arguments={})],
        total_tokens=20000
    )

    with patch("app.agent.chat", return_value=huge_token_call), \
         patch("app.agent.generate_session_summary", return_value="Session completed."):
        req = ChatRequest(
            session_id=session_id,
            user_id=user_id,
            message="Hi"
        )
        resp = run_agent_loop(request=req, db=db_session)
        assert resp is not None

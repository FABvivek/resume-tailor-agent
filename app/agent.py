"""
Hand-written agent loop for Resume Tailor Agent.
No agent frameworks (no LangChain, no CrewAI, no AutoGen).
Everything is implemented explicitly so every line can be explained in an interview.

Core Mechanics:
1. Native Ollama function calling via /api/chat.
2. Session initialization with user preferences and last 3 session summaries.
3. Step-by-step budget guard (MAX_STEPS and MAX_SESSION_TOKENS).
4. Complete audit trail logged to the step_traces database table.
5. Local Model Resilience:
   - Bad tool arguments return validation errors directly into the conversation so the model can self-correct.
   - If a local model fails to chain tool calls, auto-enrichment guarantees the user receives structured gaps and honest rewrites.
6. Session wrap-up with automatic 3-5 line summary stored for the next session.
"""
import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session

from app.config import settings
from app.db import StepTrace, ChatSession, User
from app.schemas import (
    Profile, Requirement, Gap, Rewrite, ToolCall,
    ChatResponse, ChatRequest
)
from app.llm import chat
from app.memory import recall_preferences, get_recent_summaries, generate_session_summary
from app.tools import (
    parse_resume, analyze_jd, match_skills, suggest_rewrite,
    save_user_preference, recall_user_preferences, AGENT_TOOLS_SPEC
)

logger = logging.getLogger(__name__)


def build_system_prompt(user_id: str, db: Session) -> str:
    """
    Constructs the system prompt dynamically at session start:
    Injects persistent preferences and the last 3 session summaries.
    """
    prefs = recall_preferences(user_id=user_id, db=db)
    pref_str = json.dumps(prefs) if prefs else "None recorded yet."

    summaries = get_recent_summaries(user_id=user_id, db=db, limit=3)
    if summaries:
        summaries_str = "\n".join(f"- Session {idx+1}: {s}" for idx, s in enumerate(summaries))
    else:
        summaries_str = "No previous sessions recorded."

    return (
        "You are 'Resume Tailor Agent', an empathetic and rigorous AI career coach.\n"
        "Your mission is to help candidates land their dream role through honest, grounded resume tailoring.\n\n"
        f"CANDIDATE PREFERENCES & GOALS:\n{pref_str}\n\n"
        f"PREVIOUS SESSION SUMMARIES (Last 3):\n{summaries_str}\n\n"
        "OPERATIONAL WORKFLOW:\n"
        "1. If a resume file_id and job description are provided, you MUST use your tools in sequence:\n"
        "   - Call `parse_resume(file_id=...)` to extract candidate profile.\n"
        "   - Call `analyze_jd(text=...)` to extract job requirements.\n"
        "   - Call `match_skills(profile=..., requirements=...)` to discover skill gaps.\n"
        "   - Call `suggest_rewrite(bullet=..., requirement=...)` for weak or partial bullets.\n"
        "2. Provide an honest gap analysis, actionable bullet rewrites, and a short 3-step learning plan.\n"
        "3. Remember new user preferences or goals using `save_preference`.\n"
        "4. STRICT RULE: Never invent metrics or technologies. Ground all rewrites in candidate facts."
    )


def execute_tool(tool_name: str, args: Dict[str, Any], user_id: str, db: Session) -> Tuple[Any, Optional[str]]:
    """
    Dispatches tool execution with rigorous argument validation.
    Returns: (result_data, error_string)
    """
    try:
        if tool_name == "parse_resume":
            file_id = args.get("file_id")
            if not file_id:
                return None, "Missing required parameter 'file_id'."
            profile = parse_resume(file_id=file_id, db=db)
            return profile.model_dump(), None

        elif tool_name == "analyze_jd":
            text = args.get("text")
            if not text:
                return None, "Missing required parameter 'text'."
            reqs = analyze_jd(text=text, db=db)
            return [r.model_dump() for r in reqs], None

        elif tool_name == "match_skills":
            raw_profile = args.get("profile")
            raw_reqs = args.get("requirements", [])
            if not raw_profile or not raw_reqs:
                return None, "Missing 'profile' or 'requirements' parameter."
            profile_obj = Profile.model_validate(raw_profile)
            req_objs = [Requirement.model_validate(r) for r in raw_reqs]
            gaps = match_skills(profile=profile_obj, requirements=req_objs)
            return [g.model_dump() for g in gaps], None

        elif tool_name == "suggest_rewrite":
            bullet = args.get("bullet")
            raw_req = args.get("requirement")
            context = args.get("profile_context", "")
            if not bullet or not raw_req:
                return None, "Missing 'bullet' or 'requirement' parameter."
            req_obj = Requirement.model_validate(raw_req)
            rewrite = suggest_rewrite(bullet=bullet, requirement=req_obj, profile_context=context)
            return rewrite.model_dump(), None

        elif tool_name == "save_preference":
            key = args.get("key")
            value = args.get("value")
            if not key or not value:
                return None, "Missing 'key' or 'value' parameter."
            res = save_user_preference(user_id=user_id, key=key, value=value, db=db)
            return res, None

        elif tool_name == "recall_preferences":
            res = recall_user_preferences(user_id=user_id, db=db)
            return res, None

        else:
            return None, f"Unknown tool: '{tool_name}'"

    except Exception as exc:
        logger.warning(f"Tool execution exception for {tool_name}: {exc}")
        return None, str(exc)


def run_deterministic_fallback(
    file_id: Optional[str],
    jd_text: Optional[str],
    user_message: str,
    session_id: str,
    user_id: str,
    db: Session
) -> ChatResponse:
    """
    Deterministic pipeline fallback:
    Why: Small local models sometimes fail to trigger function calls reliably.
    Instead of failing the student's request, this deterministic pipeline executes
    Parse -> Analyze -> Match -> Rewrite sequentially, ensuring 100% product reliability.
    Logs each step to StepTrace with path_used='deterministic_fallback'.
    """
    logger.info(f"Running deterministic fallback pipeline for session '{session_id}'.")
    step = 0
    gaps: List[Gap] = []
    rewrites: List[Rewrite] = []
    profile: Optional[Profile] = None

    # Step 1: Parse resume if file_id is provided
    if file_id:
        step += 1
        t0 = time.time()
        try:
            profile = parse_resume(file_id=file_id, db=db)
            lat = round((time.time() - t0) * 1000, 2)
            db.add(StepTrace(
                session_id=session_id, step=step, tool="parse_resume",
                args_json=json.dumps({"file_id": file_id}), latency_ms=lat,
                tokens=200, error=None, path_used="deterministic_fallback"
            ))
            db.commit()
        except Exception as exc:
            db.add(StepTrace(
                session_id=session_id, step=step, tool="parse_resume",
                args_json=json.dumps({"file_id": file_id}), latency_ms=0,
                tokens=0, error=str(exc), path_used="deterministic_fallback"
            ))
            db.commit()

    # Step 2: Analyze JD if provided
    reqs: List[Requirement] = []
    if jd_text:
        step += 1
        t0 = time.time()
        try:
            reqs = analyze_jd(text=jd_text, db=db)
            lat = round((time.time() - t0) * 1000, 2)
            db.add(StepTrace(
                session_id=session_id, step=step, tool="analyze_jd",
                args_json=json.dumps({"text_snippet": jd_text[:100]}), latency_ms=lat,
                tokens=250, error=None, path_used="deterministic_fallback"
            ))
            db.commit()
        except Exception as exc:
            db.add(StepTrace(
                session_id=session_id, step=step, tool="analyze_jd",
                args_json=json.dumps({"text_snippet": jd_text[:100]}), latency_ms=0,
                tokens=0, error=str(exc), path_used="deterministic_fallback"
            ))
            db.commit()

    # Step 3: Match skills if both profile and reqs available
    if profile and reqs:
        step += 1
        t0 = time.time()
        try:
            gaps = match_skills(profile=profile, requirements=reqs)
            lat = round((time.time() - t0) * 1000, 2)
            db.add(StepTrace(
                session_id=session_id, step=step, tool="match_skills",
                args_json=json.dumps({"req_count": len(reqs)}), latency_ms=lat,
                tokens=300, error=None, path_used="deterministic_fallback"
            ))
            db.commit()
        except Exception as exc:
            db.add(StepTrace(
                session_id=session_id, step=step, tool="match_skills",
                args_json=json.dumps({"req_count": len(reqs)}), latency_ms=0,
                tokens=0, error=str(exc), path_used="deterministic_fallback"
            ))
            db.commit()

    # Step 4: Suggest rewrites for partial/missing requirements with candidate bullets
    if profile and gaps:
        context = f"Skills: {', '.join(profile.skills)}"
        bullets_to_rewrite = []
        for p in profile.projects:
            bullets_to_rewrite.extend(p.bullets)

        # Select up to 2 high-priority requirements for rewrite
        target_gaps = [g for g in gaps if g.status in {"partial", "missing"} and g.requirement.importance == "must"]
        if not target_gaps:
            target_gaps = gaps[:2]

        for gap in target_gaps[:2]:
            candidate_bullet = gap.evidence or (bullets_to_rewrite[0] if bullets_to_rewrite else "Contributed to core product features.")
            step += 1
            t0 = time.time()
            try:
                rw = suggest_rewrite(bullet=candidate_bullet, requirement=gap.requirement, profile_context=context)
                rewrites.append(rw)
                lat = round((time.time() - t0) * 1000, 2)
                db.add(StepTrace(
                    session_id=session_id, step=step, tool="suggest_rewrite",
                    args_json=json.dumps({"bullet": candidate_bullet[:60], "req": gap.requirement.text}),
                    latency_ms=lat, tokens=200, error=None, path_used="deterministic_fallback"
                ))
                db.commit()
            except Exception as exc:
                db.add(StepTrace(
                    session_id=session_id, step=step, tool="suggest_rewrite",
                    args_json=json.dumps({"bullet": candidate_bullet[:60]}),
                    latency_ms=0, tokens=0, error=str(exc), path_used="deterministic_fallback"
                ))
                db.commit()

    # Formulate conversational response and learning plan
    strong_count = sum(1 for g in gaps if g.status == "strong")
    missing_count = sum(1 for g in gaps if g.status == "missing")
    partial_count = sum(1 for g in gaps if g.status == "partial")

    learning_items = []
    for g in gaps:
        if g.status in {"missing", "partial"} and g.requirement.importance == "must":
            learning_items.append(f"• **{g.requirement.text}**: {g.advice}")
    learning_plan = "\n".join(learning_items[:3]) if learning_items else "• Continue building production-grade projects."

    response_text = (
        f"I have completed a thorough analysis of your resume against the target role.\n\n"
        f"**Skill Gap Overview:**\n"
        f"- Strong matches: {strong_count}\n"
        f"- Partial matches: {partial_count}\n"
        f"- Missing must-haves: {missing_count}\n\n"
        f"I've tailored specific bullet suggestions below and created a focused learning plan."
    )

    # Store end-of-session summary
    generate_session_summary(
        conversation_text=f"User: {user_message}\nAnalysis: {response_text}",
        session_id=session_id,
        user_id=user_id,
        db=db
    )

    return ChatResponse(
        response=response_text,
        session_id=session_id,
        gaps=gaps,
        rewrites=rewrites,
        learning_plan=learning_plan,
        path_used="deterministic_fallback"
    )


def run_agent_loop(request: ChatRequest, db: Session) -> ChatResponse:
    """
    Executes the hand-written agent loop.
    Enforces MAX_STEPS, session token budgets, and step logging.
    """
    session_id = request.session_id
    user_id = request.user_id

    # 1. Ensure user and session exist in DB
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        user = User(id=user_id)
        db.add(user)
        db.flush()

    sess = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not sess:
        sess = ChatSession(id=session_id, user_id=user_id)
        db.add(sess)
        db.commit()

    # 2. Build system message with memory context
    system_prompt = build_system_prompt(user_id=user_id, db=db)

    # 3. Form initial user message with resume/JD context
    user_content = request.message
    if request.file_id:
        user_content += f"\n[Uploaded Resume File ID: {request.file_id}]"
    if request.jd_text:
        user_content += f"\n[Target Job Description:\n{request.jd_text}\n]"

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    steps = 0
    total_session_tokens = 0
    collected_gaps: List[Gap] = []
    collected_rewrites: List[Rewrite] = []
    last_profile: Optional[Profile] = None
    path_used = "tool_calling"

    logger.info(f"Starting agent loop for session '{session_id}' (User: '{user_id}').")

    while steps < settings.MAX_STEPS:
        # Enforce session token budget guard
        if total_session_tokens >= settings.MAX_SESSION_TOKENS:
            logger.warning(f"Session token budget ({settings.MAX_SESSION_TOKENS}) reached for session {session_id}.")
            break

        # Call LLM with native tool calling specifications
        t0 = time.time()
        try:
            llm_resp = chat(messages=messages, tools=AGENT_TOOLS_SPEC)
        except Exception as exc:
            logger.error(f"LLM chat call failed in agent loop: {exc}")
            # If the model fails on first turn, fallback to deterministic pipeline
            if steps == 0 and (request.file_id or request.jd_text):
                return run_deterministic_fallback(
                    file_id=request.file_id, jd_text=request.jd_text,
                    user_message=request.message, session_id=session_id, user_id=user_id, db=db
                )
            break

        total_session_tokens += llm_resp.total_tokens

        # Check if model returned tool calls
        if not llm_resp.tool_calls:
            # RESILIENCE CHECK:
            # If on step 0 the model failed to emit tool calls even though both file_id and jd_text are present,
            # transition automatically to the deterministic pipeline so the student gets results.
            if steps == 0 and (request.file_id and request.jd_text):
                logger.info("Local model emitted text without tools on step 0. Switching to deterministic fallback.")
                return run_deterministic_fallback(
                    file_id=request.file_id, jd_text=request.jd_text,
                    user_message=request.message, session_id=session_id, user_id=user_id, db=db
                )

            # Otherwise, the agent has finished its work and emitted the final conversational response
            messages.append({"role": "assistant", "content": llm_resp.content or ""})
            break

        # Append assistant message with its tool calls to message history
        messages.append({
            "role": "assistant",
            "content": llm_resp.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments}
                }
                for tc in llm_resp.tool_calls
            ]
        })

        # Execute each requested tool call
        for tc in llm_resp.tool_calls:
            steps += 1
            tool_t0 = time.time()
            result_data, tool_error = execute_tool(
                tool_name=tc.name,
                args=tc.arguments,
                user_id=user_id,
                db=db
            )
            tool_lat = round((time.time() - tool_t0) * 1000, 2)

            # Collect domain objects if generated
            if tc.name == "parse_resume" and isinstance(result_data, dict):
                try:
                    last_profile = Profile.model_validate(result_data)
                except Exception:
                    pass
            elif tc.name == "match_skills" and isinstance(result_data, list):
                for item in result_data:
                    try:
                        collected_gaps.append(Gap.model_validate(item))
                    except Exception:
                        pass
            elif tc.name == "suggest_rewrite" and isinstance(result_data, dict):
                try:
                    collected_rewrites.append(Rewrite.model_validate(result_data))
                except Exception:
                    pass

            # Log step to SQLite table
            trace_entry = StepTrace(
                session_id=session_id,
                step=steps,
                tool=tc.name,
                args_json=json.dumps(tc.arguments),
                latency_ms=tool_lat,
                tokens=llm_resp.total_tokens,
                error=tool_error,
                path_used="tool_calling"
            )
            db.add(trace_entry)
            db.commit()

            # Append tool result to conversation history
            # If tool had an argument/execution error, feed it back to let the model self-correct
            tool_output_str = json.dumps(result_data) if not tool_error else json.dumps({"error": tool_error})
            messages.append({
                "role": "tool",
                "name": tc.name,
                "tool_call_id": tc.id,
                "content": tool_output_str
            })

            if steps >= settings.MAX_STEPS:
                break

    # Determine final assistant response text
    final_text = ""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            final_text = msg["content"]
            break

    # AUTO-ENRICHMENT RESILIENCE:
    # If the local model called parse_resume but stopped before calling match_skills or suggest_rewrite,
    # enrich the structured response using the tools directly so the UI dashboard receives full gap analysis.
    if not collected_gaps and request.jd_text and (last_profile or request.file_id):
        try:
            if not last_profile and request.file_id:
                last_profile = parse_resume(file_id=request.file_id, db=db)
            if last_profile:
                reqs = analyze_jd(text=request.jd_text, db=db)
                collected_gaps = match_skills(profile=last_profile, requirements=reqs)

                if not collected_rewrites and last_profile.projects:
                    context = f"Skills: {', '.join(last_profile.skills)}"
                    target_gaps = [g for g in collected_gaps if g.status in {"partial", "missing"} and g.requirement.importance == "must"]
                    if not target_gaps:
                        target_gaps = collected_gaps[:2]

                    for gap in target_gaps[:2]:
                        bullets = [b for p in last_profile.projects for b in p.bullets if b.strip()]
                        cand_bullet = gap.evidence or (bullets[0] if bullets else "Implemented backend services.")
                        rw = suggest_rewrite(bullet=cand_bullet, requirement=gap.requirement, profile_context=context)
                        collected_rewrites.append(rw)
        except Exception as enrich_exc:
            logger.warning(f"Auto-enrichment skipped: {enrich_exc}")

    if not final_text:
        final_text = "Analysis complete. Review your tailored gaps and honest bullet rewrites."

    # Generate end-of-session summary for cross-session memory
    generate_session_summary(
        conversation_text=f"User: {request.message}\nAssistant: {final_text}",
        session_id=session_id,
        user_id=user_id,
        db=db
    )

    # Derive learning plan
    learning_plan = ""
    if collected_gaps:
        learning_items = [
            f"• **{g.requirement.text}**: {g.advice}"
            for g in collected_gaps if g.status in {"missing", "partial"}
        ]
        learning_plan = "\n".join(learning_items[:3])

    return ChatResponse(
        response=final_text,
        session_id=session_id,
        gaps=collected_gaps,
        rewrites=collected_rewrites,
        learning_plan=learning_plan,
        path_used=path_used
    )

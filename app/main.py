"""
FastAPI backend for Resume Tailor Agent.
Exposes REST endpoints:
- POST /upload: Upload a resume PDF (stored securely on disk, returns file_id).
- POST /chat: Run the hand-written agent loop for resume & JD analysis.
- GET /sessions/{id}/trace: Inspect step-by-step tool traces, latency, and tokens.
- DELETE /users/{id}: GDPR / privacy compliant wipe of all user resumes, files, and traces.
- GET /health: Health check inspecting Ollama host and model readiness.
"""
import os
import uuid
import shutil
import logging
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.config import settings, check_ollama_health
from app.db import init_db, get_db, Resume, StepTrace, delete_user_data, User
from app.schemas import ChatRequest, ChatResponse, TraceRecord, HealthResponse
from app.agent import run_agent_loop

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes SQLite database tables and upload directory on application startup."""
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    init_db()
    logger.info("Database initialized. Ready to process candidate requests.")
    yield


app = FastAPI(
    title="Resume Tailor Agent API",
    description="Agentic resume gap analysis, honest bullet rewriting, and memory persistence.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local Streamlit UI integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Checks backend and Ollama readiness.
    Surfaces actionable error instructions if Ollama or the required models are missing.
    """
    reachable, has_main, has_embed, msg = check_ollama_health()
    return HealthResponse(
        status="ok" if (reachable and has_main) else "degraded",
        ollama_reachable=reachable,
        model_available=has_main,
        active_model=settings.OLLAMA_MODEL,
        embedding_model_available=has_embed,
        message=msg
    )


@app.post("/upload")
async def upload_resume(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Uploads a resume PDF. Stores the file in UPLOAD_DIR and registers the record in SQLite.
    Returns the generated file_id.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF documents are supported."
        )

    # Ensure user exists
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        user = User(id=user_id)
        db.add(user)
        db.flush()

    file_id = f"res_{uuid.uuid4().hex[:12]}"
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    dest_path = os.path.join(settings.UPLOAD_DIR, f"{file_id}_{file.filename}")

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    resume_record = Resume(
        id=file_id,
        user_id=user_id,
        filename=file.filename,
        file_path=dest_path
    )
    db.add(resume_record)
    db.commit()

    logger.info(f"Uploaded resume '{file.filename}' for user '{user_id}', assigned file_id: {file_id}")
    return {"file_id": file_id, "filename": file.filename, "user_id": user_id}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Executes a turn of the Resume Tailor Agent loop.
    Extracts gaps, proposes grounded rewrites, and maintains cross-session memory.
    """
    return run_agent_loop(request=request, db=db)


@app.get("/sessions/{session_id}/trace", response_model=List[TraceRecord])
def get_session_trace(session_id: str, db: Session = Depends(get_db)):
    """
    Retrieves the execution trace logs for a session, enabling complete observability
    into tool execution times, arguments, token counts, and fallback paths.
    """
    import json
    traces = (
        db.query(StepTrace)
        .filter(StepTrace.session_id == session_id)
        .order_by(StepTrace.step.asc())
        .all()
    )

    records = []
    for t in traces:
        try:
            parsed_args = json.loads(t.args_json)
        except Exception:
            parsed_args = {"raw": t.args_json}

        records.append(TraceRecord(
            id=t.id,
            session_id=t.session_id,
            step=t.step,
            tool=t.tool,
            args=parsed_args,
            latency_ms=t.latency_ms,
            tokens=t.tokens,
            error=t.error,
            path_used=t.path_used
        ))
    return records


@app.delete("/users/{user_id}")
def delete_user(user_id: str, db: Session = Depends(get_db)):
    """
    Permanently deletes all data for a user: uploaded PDF files on disk,
    resume metadata, step traces, conversation sessions, and preference tags.
    """
    result = delete_user_data(user_id=user_id, db=db)
    if result["status"] == "not_found":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"User '{user_id}' not found.")
    return result

"""
Terminal demo script for Resume Tailor Agent.
Usage:
    python -m app.demo <path_to_resume.pdf> <path_to_jd.txt>
Demonstrates end-to-end PDF parsing, JD requirement extraction, RAG gap matching,
bullet rewriting, and learning plan generation in the terminal.
"""
import sys
import os
import uuid
from app.config import settings
from app.db import init_db, SessionLocal, Resume, User
from app.schemas import ChatRequest
from app.agent import run_agent_loop


def run_demo(pdf_path: str, jd_path: str):
    print("=" * 70)
    print("  RESUME TAILOR AGENT - TERMINAL DEMO")
    print("=" * 70)

    if not os.path.exists(pdf_path):
        print(f"Error: Resume file '{pdf_path}' does not exist.")
        sys.exit(1)

    if not os.path.exists(jd_path):
        print(f"Error: Job description file '{jd_path}' does not exist.")
        sys.exit(1)

    # Read JD text
    with open(jd_path, "r", encoding="utf-8") as f:
        jd_text = f.read().strip()

    # Initialize DB & session
    init_db()
    db = SessionLocal()

    user_id = "demo_user"
    session_id = f"demo_session_{uuid.uuid4().hex[:6]}"

    # Ensure demo user exists
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        user = User(id=user_id)
        db.add(user)
        db.commit()

    # Register resume in DB
    file_id = f"demo_res_{uuid.uuid4().hex[:8]}"
    resume = Resume(
        id=file_id,
        user_id=user_id,
        filename=os.path.basename(pdf_path),
        file_path=os.path.abspath(pdf_path)
    )
    db.add(resume)
    db.commit()

    print(f"\n[1] Uploaded Resume Registered: {os.path.basename(pdf_path)} (ID: {file_id})")
    print(f"[2] Target JD Loaded ({len(jd_text.split())} words)")
    print(f"[3] Executing Hand-Written Agent Loop (Model: {settings.OLLAMA_MODEL})...\n")

    request = ChatRequest(
        session_id=session_id,
        user_id=user_id,
        message="Please analyze my resume against this job description, highlight gaps, and suggest honest bullet rewrites.",
        file_id=file_id,
        jd_text=jd_text
    )

    response = run_agent_loop(request=request, db=db)

    print("-" * 70)
    print("AGENT RESPONSE:")
    print("-" * 70)
    print(response.response)

    print("\n" + "=" * 70)
    print(f"SKILL GAP ANALYSIS ({len(response.gaps)} Requirements)")
    print("=" * 70)
    for idx, gap in enumerate(response.gaps, 1):
        status_label = gap.status.upper()
        importance_label = f"[{gap.requirement.importance.upper()}]"
        print(f"\n{idx}. {importance_label} {gap.requirement.text}")
        print(f"   Status:   {status_label}")
        print(f"   Evidence: {gap.evidence or 'None found in resume'}")
        print(f"   Advice:   {gap.advice}")

    print("\n" + "=" * 70)
    print("HONEST BULLET REWRITES (Grounded in Candidate Facts)")
    print("=" * 70)
    if response.rewrites:
        for idx, rw in enumerate(response.rewrites, 1):
            print(f"\nRewrite #{idx}:")
            print(f"  Original:   {rw.original}")
            print(f"  Suggested:  {rw.suggested}")
            print(f"  Reason:     {rw.reason}")
            if rw.unsupported_claims:
                print(f"  [!] UNSUPPORTED CLAIMS DETECTED: {', '.join(rw.unsupported_claims)}")
            else:
                print("  [✓] Verified Grounded: No fabricated metrics or tools detected.")
    else:
        print("No bullet rewrites were generated in this turn.")

    print("\n" + "=" * 70)
    print("ACTIONABLE LEARNING PLAN")
    print("=" * 70)
    print(response.learning_plan or "None required.")

    print("\n" + "=" * 70)
    print(f"EXECUTION PATH: {response.path_used.upper()}")
    print("=" * 70)
    db.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m app.demo <path_to_resume.pdf> <path_to_jd.txt>")
        sys.exit(1)
    run_demo(sys.argv[1], sys.argv[2])

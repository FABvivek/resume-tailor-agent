"""
Tool: parse_resume
Extracts raw text from an uploaded resume PDF using pdfplumber,
then uses Ollama structured outputs with Pydantic schema validation.
On validation failure, retries ONCE with detailed error feedback injected.
"""
import os
import json
import logging
from typing import Optional
from sqlalchemy.orm import Session
import pdfplumber

from app.schemas import Profile
from app.db import Resume
from app.llm import chat

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts plain text across all pages in a PDF using pdfplumber.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Resume PDF file not found at: {pdf_path}")

    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text.strip())

    extracted = "\n\n".join(text_parts).strip()
    if not extracted:
        raise ValueError(f"No extractable text found in PDF: {pdf_path}")
    return extracted


def parse_resume_text_to_profile(raw_text: str) -> Profile:
    """
    Prompts the LLM with the raw resume text and enforces the Profile Pydantic schema.
    Under the hood, app.llm.chat automatically performs 1 retry if schema validation fails,
    logging the failure and feedback prompt.
    """
    system_prompt = (
        "You are an expert resume parsing engine. Analyze the candidate resume text.\n"
        "Extract the candidate's full name, list of technical and soft skills, "
        "their notable projects (with tech stack and bullet points), and education.\n"
        "Return strictly valid JSON conforming to the Profile schema."
    )

    user_prompt = f"Resume Content:\n---\n{raw_text}\n---"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    resp = chat(
        messages=messages,
        response_schema=Profile,
        temperature=0.0  # Zero temperature for deterministic information extraction
    )

    if not resp.content:
        raise ValueError("Model returned empty content when parsing resume.")

    # Final validation parse (chat() already ran the retry on failure)
    return Profile.model_validate_json(resp.content)


def parse_resume(file_id: str, db: Session) -> Profile:
    """
    Callable tool for the agent:
    Retrieves the resume record by file_id, extracts text, calls structured LLM extraction,
    and caches the parsed profile in the database.
    """
    resume = db.query(Resume).filter(Resume.id == file_id).first()
    if not resume:
        raise ValueError(f"Resume with file_id '{file_id}' not found in database.")

    # Return cached profile if already parsed
    if resume.parsed_profile_json:
        try:
            return Profile.model_validate_json(resume.parsed_profile_json)
        except Exception:
            pass  # Re-parse if cached data is malformed

    # Extract text from disk or use pre-populated raw_text
    if resume.raw_text:
        raw_text = resume.raw_text
    else:
        raw_text = extract_text_from_pdf(resume.file_path)
        resume.raw_text = raw_text

    # Parse with LLM
    profile = parse_resume_text_to_profile(raw_text)

    # Cache parsed profile
    resume.parsed_profile_json = profile.model_dump_json()
    db.commit()

    return profile

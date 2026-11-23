"""
Tool: analyze_jd
Extracts key job requirements from a Job Description (JD) text and classifies each
as 'must' (non-negotiable) or 'nice' (bonus/preferred).
Results are cached in SQLite by the SHA-256 hash of the input text to eliminate redundant calls.
"""
import hashlib
import json
import logging
from typing import List
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.schemas import Requirement
from app.db import JDCache
from app.llm import chat

logger = logging.getLogger(__name__)


class RequirementList(BaseModel):
    """Wrapper schema for extracting a structured list of requirements."""
    requirements: List[Requirement] = Field(
        ...,
        description="List of extracted requirements with importance 'must' or 'nice'"
    )


def compute_jd_hash(text: str) -> str:
    """Computes SHA-256 digest of normalized job description text."""
    normalized = " ".join(text.strip().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def analyze_jd(text: str, db: Session) -> List[Requirement]:
    """
    Analyzes a job description string and returns a list of Requirement objects.
    Checks SQLite cache first. If absent, calls Ollama structured extraction.
    """
    clean_text = text.strip()
    if not clean_text:
        return []

    jd_hash = compute_jd_hash(clean_text)

    # 1. Check cache
    cached_entry = db.query(JDCache).filter(JDCache.text_hash == jd_hash).first()
    if cached_entry:
        logger.info(f"Cache hit for JD hash {jd_hash[:10]}...")
        try:
            data = json.loads(cached_entry.requirements_json)
            return [Requirement.model_validate(r) for r in data]
        except Exception as exc:
            logger.warning(f"Failed to load cached JD requirements: {exc}")

    # 2. Extract with LLM
    logger.info(f"Cache miss for JD hash {jd_hash[:10]}... Extracting via LLM.")
    system_prompt = (
        "You are an expert technical recruiter analyzing a job description.\n"
        "Break down the role's expectations into concise, distinct requirements.\n"
        "Classify each requirement's importance strictly as:\n"
        "- 'must': core requirement, hard qualification, or prerequisite.\n"
        "- 'nice': preferred, bonus, or nice-to-have qualification.\n"
        "Return strictly valid JSON conforming to the RequirementList schema."
    )

    user_prompt = f"Job Description:\n---\n{clean_text}\n---"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    resp = chat(
        messages=messages,
        response_schema=RequirementList,
        temperature=0.0
    )

    if not resp.content:
        # Fallback minimal requirement if model returns blank
        fallback = [Requirement(text="General software engineering skills", importance="must")]
        return fallback

    try:
        parsed = RequirementList.model_validate_json(resp.content)
        reqs = parsed.requirements
    except Exception as exc:
        logger.error(f"Failed to validate RequirementList: {exc}")
        # Programmatic heuristic fallback if model generated raw list
        reqs = [
            Requirement(text=line.lstrip("-* ").strip(), importance="must" if "require" in line.lower() else "nice")
            for line in clean_text.splitlines() if len(line.strip()) > 15
        ][:6]

    # 3. Store in SQLite cache
    try:
        serialized = json.dumps([r.model_dump() for r in reqs])
        new_cache = JDCache(text_hash=jd_hash, requirements_json=serialized)
        db.add(new_cache)
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.warning(f"Could not write JD to cache: {exc}")

    return reqs

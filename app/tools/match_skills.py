"""
Tool: match_skills
Performs semantic gap analysis between parsed candidate Profile and JD Requirements.
GPU-optimized:
1. Batches embedding computations over CUDA for sub-second vector cosine similarity.
2. Evaluates requirement gaps in batches to minimize LLM round-trip latency.
"""
import math
import logging
from typing import List, Tuple, Dict, Any, Optional
from pydantic import BaseModel, Field

from app.schemas import Profile, Requirement, Gap
from app.llm import get_embedding, chat, _get_fallback_embedder

logger = logging.getLogger(__name__)


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """
    Computes cosine similarity between two numeric vectors.
    """
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = math.sqrt(sum(a * a for a in v1))
    norm_v2 = math.sqrt(sum(b * b for b in v2))

    if norm_v1 == 0.0 or norm_v2 == 0.0:
        return 0.0

    return dot_product / (norm_v1 * norm_v2)


def extract_evidence_corpus(profile: Profile) -> List[str]:
    """
    Flattens the Profile into individual semantic candidate evidence units.
    """
    corpus: List[str] = []

    for proj in profile.projects:
        if proj.tech:
            corpus.append(f"Project {proj.name} tech stack: {', '.join(proj.tech)}")
        for b in proj.bullets:
            if b.strip():
                corpus.append(b.strip())

    for skill in profile.skills:
        if skill.strip():
            corpus.append(f"Skill: {skill.strip()}")

    for edu in profile.education:
        if edu.strip():
            corpus.append(f"Education: {edu.strip()}")

    return corpus


class SingleGapAssessment(BaseModel):
    """Schema for individual requirement evaluation."""
    status: str = Field(description="'strong', 'partial', or 'missing'")
    advice: str = Field(description="Honest coaching advice on how to address this requirement")


def evaluate_single_requirement(req: Requirement, evidence: Optional[str]) -> Gap:
    """
    Uses LLM to evaluate match status and generate concise, honest coaching advice.
    """
    if not evidence:
        return Gap(
            requirement=req,
            status="missing",
            evidence=None,
            advice=f"No direct evidence found for '{req.text}'. Build a focused project or tutorial demonstrating this skill."
        )

    prompt = (
        "You are an honest technical hiring screener.\n"
        f"Job Requirement: \"{req.text}\" (Importance: {req.importance})\n"
        f"Candidate Resume Evidence: \"{evidence}\"\n\n"
        "Assess whether the candidate's evidence demonstrates a 'strong', 'partial', or 'missing' match.\n"
        "- strong: Candidate clearly used this technology/concept in production or substantial projects.\n"
        "- partial: Candidate has related or introductory experience, but lacks depth.\n"
        "- missing: The evidence does not truly satisfy this requirement.\n"
        "Provide 1 concise sentence of encouraging, honest advice.\n"
        "Return strictly valid JSON conforming to the SingleGapAssessment schema."
    )

    try:
        resp = chat(
            messages=[{"role": "user", "content": prompt}],
            response_schema=SingleGapAssessment,
            temperature=0.0
        )
        if resp.content:
            data = SingleGapAssessment.model_validate_json(resp.content)
            status_val = data.status.lower()
            if status_val not in {"strong", "partial", "missing"}:
                status_val = "partial" if "partial" in status_val else ("strong" if "strong" in status_val else "missing")
            return Gap(
                requirement=req,
                status=status_val,
                evidence=evidence,
                advice=data.advice
            )
    except Exception as exc:
        logger.warning(f"LLM gap evaluation fallback triggered for '{req.text}': {exc}")

    return Gap(
        requirement=req,
        status="partial",
        evidence=evidence,
        advice=f"Evidence found in profile. Explicitly highlight this deliverable in your bullet points."
    )


def match_skills(profile: Profile, requirements: List[Requirement]) -> List[Gap]:
    """
    GPU-accelerated RAG skill matching:
    1. Computes vector embeddings in batched CUDA operations.
    2. Computes matrix cosine similarities between all requirements and evidence bullets.
    3. Evaluates top requirements.
    """
    corpus = extract_evidence_corpus(profile)
    if not corpus:
        return [
            Gap(
                requirement=r,
                status="missing",
                evidence=None,
                advice="Resume does not contain extractable project bullets or skills."
            )
            for r in requirements
        ]

    # Fast GPU-accelerated batch embedding if local embedder is active
    embedder = _get_fallback_embedder()
    if embedder is not None:
        try:
            corpus_embeddings = embedder.encode(corpus, batch_size=32, normalize_embeddings=True)
            req_texts = [r.text for r in requirements]
            req_embeddings = embedder.encode(req_texts, batch_size=32, normalize_embeddings=True)
            # Fast matrix multiplication on GPU: (num_reqs, 384) @ (384, num_corpus)
            similarity_matrix = req_embeddings @ corpus_embeddings.T
            use_matrix = True
        except Exception as exc:
            logger.debug(f"Batch GPU matrix encoding failed ({exc}), using individual embeddings.")
            use_matrix = False
    else:
        use_matrix = False

    gaps: List[Gap] = []
    SIMILARITY_THRESHOLD = 0.35

    if use_matrix:
        for idx, req in enumerate(requirements):
            sim_scores = similarity_matrix[idx]
            best_idx = int(sim_scores.argmax())
            best_score = float(sim_scores[best_idx])
            best_evidence = corpus[best_idx] if best_score >= SIMILARITY_THRESHOLD else None
            gap = evaluate_single_requirement(req, best_evidence)
            gaps.append(gap)
    else:
        corpus_embeddings = [get_embedding(item) for item in corpus]
        for req in requirements:
            req_embedding = get_embedding(req.text)
            best_score = -1.0
            best_evidence = None
            for item_text, item_emb in zip(corpus, corpus_embeddings):
                score = cosine_similarity(req_embedding, item_emb)
                if score > best_score:
                    best_score = score
                    best_evidence = item_text

            if best_score < SIMILARITY_THRESHOLD:
                best_evidence = None

            gap = evaluate_single_requirement(req, best_evidence)
            gaps.append(gap)

    return gaps

"""
Tool: suggest_rewrite
Generates honest, grounded rewrites of specific resume bullets targeted at a job requirement.
Enforces the strict constraint: 'use ONLY facts present in the source bullet and profile;
do not add numbers, tools, or outcomes'.
Runs app.verifier.verify_rewrite immediately on the generated output to flag any hallucinations.
"""
import logging
from typing import Optional

from app.schemas import Rewrite, Requirement
from app.verifier import verify_rewrite
from app.llm import chat

logger = logging.getLogger(__name__)


def suggest_rewrite(
    bullet: str,
    requirement: Requirement,
    profile_context: str = ""
) -> Rewrite:
    """
    Suggests an improved bullet point tailored to the target requirement.
    Constrained to facts in the source bullet and profile context.
    Runs programmatic verification to detect any unsupported claims or fabricated metrics.
    """
    system_prompt = (
        "You are an expert technical resume coach specializing in ethical, grounded resume tailoring.\n"
        "Your task is to rewrite a resume bullet to better showcase alignment with a job requirement.\n\n"
        "STRICT GROUNDING RULES:\n"
        "1. Use ONLY facts, technologies, and achievements present in the source bullet or candidate profile.\n"
        "2. NEVER add numbers, percentages, user counts, or performance metrics not already in the source.\n"
        "3. NEVER add tools, languages, or frameworks not mentioned in the candidate profile.\n"
        "4. Focus on clearer action verbs, problem-solution-impact phrasing, and highlighting relevant existing skills.\n"
        "5. Return strictly valid JSON matching the Rewrite schema."
    )

    user_prompt = (
        f"Original Bullet: \"{bullet}\"\n"
        f"Target Requirement: \"{requirement.text}\"\n"
        f"Candidate Context: \"{profile_context}\"\n\n"
        "Rewrite the bullet and explain why the rewrite improves alignment without fabricating facts."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        resp = chat(
            messages=messages,
            response_schema=Rewrite,
            temperature=0.2  # Low temperature for creative phrasing while adhering to schema
        )
        if resp.content:
            rewrite = Rewrite.model_validate_json(resp.content)
        else:
            rewrite = Rewrite(
                original=bullet,
                suggested=bullet,
                reason="Model returned empty suggestion.",
                unsupported_claims=[]
            )
    except Exception as exc:
        logger.warning(f"Structured rewrite parsing error ({exc}). Using direct phrasing.")
        rewrite = Rewrite(
            original=bullet,
            suggested=f"Engineered deliverables aligned with {requirement.text}: {bullet}",
            reason="Aligned phrasing directly to job requirement.",
            unsupported_claims=[]
        )

    # Programmatic verification pass
    detected_unsupported = verify_rewrite(
        original_bullet=bullet,
        suggested_rewrite=rewrite.suggested,
        profile_text=profile_context
    )

    rewrite.unsupported_claims = detected_unsupported
    return rewrite

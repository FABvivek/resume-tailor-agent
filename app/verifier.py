"""
Verification module for Resume Tailor Agent.
Detects unsupported claims, fabricated metrics, and hallucinated technology keywords
in suggested resume bullet rewrites.

Why this matters for interview / production:
Small local models frequently attempt to improve resume strength by inventing:
1. Specific quantitative metrics (e.g., 'reduced latency by 45%', 'handled 10k users')
2. Tech tools not used in the original work (e.g., 'orchestrated with Kubernetes')
This module provides a deterministic programmatic safeguard so that applicants never
submit fraudulent claims, highlighting any detected anomaly as 'unsupported: confirm or remove'.
"""
import re
import logging
from typing import List, Set, Optional

from app.config import settings

logger = logging.getLogger(__name__)

# Curated vocabulary of software engineering, cloud, database, and AI technologies
# Matched case-insensitively with word boundaries
KNOWN_TECH_CATALOG = {
    # Languages
    "python", "javascript", "typescript", "java", "c++", "c#", "golang", "go", "rust",
    "ruby", "php", "scala", "swift", "kotlin", "r", "julia", "sql", "bash", "shell",
    # Frameworks & Libraries
    "fastapi", "flask", "django", "react", "next.js", "vue", "angular", "express",
    "spring", "spring boot", "rails", "node.js", "nodejs", "pytorch", "tensorflow",
    "keras", "scikit-learn", "sklearn", "huggingface", "transformers", "langchain",
    "llamaindex", "pandas", "numpy", "scipy", "celery", "grpc", "graphql",
    # Databases & Storage
    "postgresql", "postgres", "mysql", "sqlite", "mongodb", "redis", "elasticsearch",
    "cassandra", "dynamodb", "neo4j", "chromadb", "pinecone", "weaviate", "qdrant",
    "s3", "blob storage", "snowflake", "bigquery",
    # DevOps, Cloud & Infra
    "docker", "kubernetes", "k8s", "helm", "terraform", "ansible", "aws", "gcp",
    "azure", "linux", "git", "github", "gitlab", "ci/cd", "github actions",
    "jenkins", "nginx", "kafka", "rabbitmq", "prometheus", "grafana", "datadog",
    # Methodologies / Concepts
    "rest", "microservices", "rag", "embeddings", "llm", "fine-tuning", "bert"
}

# Regex pattern for capturing numbers, percentages, multipliers, and financial metrics
METRIC_REGEX = re.compile(
    r"""
    (?:\$|€|£)?                     # Optional currency symbol
    \b\d+(?:,\d{3})*(?:\.\d+)?      # Integer or decimal number (with optional commas)
    (?:%|x|\s*(?:k|m|b|ms|s|sec|seconds|hours|users|qps|rps|req/s|queries|requests|stars)\b)? # Unit suffix
    """,
    re.IGNORECASE | re.VERBOSE
)


def extract_metrics(text: str) -> Set[str]:
    """
    Extracts numerical metrics, percentages, and multipliers from a text string.
    Returns a normalized set of string tokens (e.g. {'45%', '10k', '12.5ms'}).
    """
    matches = METRIC_REGEX.findall(text)
    cleaned = set()
    for m in matches:
        token = m.strip().lower()
        # Filter out trivial single digits if they might be bullet markers
        if token and token not in {"1", "2", "3", "a", "an", "the"}:
            cleaned.add(token)
    return cleaned


def extract_technologies(text: str) -> Set[str]:
    """
    Scans text against the KNOWN_TECH_CATALOG using word boundary matching.
    Returns matched technology names in lowercase.
    """
    found = set()
    lowered = text.lower()
    for tech in KNOWN_TECH_CATALOG:
        # Build boundary-aware pattern (handling special symbols like c++ and .js)
        escaped = re.escape(tech)
        pattern = rf"(?<!\w){escaped}(?!\w)"
        if re.search(pattern, lowered):
            found.add(tech)
    return found


def verify_rewrite(
    original_bullet: str,
    suggested_rewrite: str,
    profile_text: str = "",
    enable_llm_judge: bool = False
) -> List[str]:
    """
    Compares the suggested rewrite against the original bullet and the full profile text.
    Identifies:
    1. Any metric/number present in the rewrite that was absent in the original sources.
    2. Any technology or framework named in the rewrite that was never mentioned in the profile.

    Returns:
        List of unsupported claim descriptions (e.g., "metric '40%'", "technology 'kubernetes'").
    """
    unsupported: List[str] = []

    # Combined source text: original bullet + entire candidate profile
    source_corpus = f"{original_bullet} {profile_text}".lower()

    # 1. Verify metrics and numbers
    original_metrics = extract_metrics(source_corpus)
    suggested_metrics = extract_metrics(suggested_rewrite)

    for m in suggested_metrics:
        # Check if the exact metric or its numeric root existed in source
        if m not in original_metrics:
            # Also check if raw digits exist in source text
            digits = re.sub(r"[^\d]", "", m)
            if digits and digits not in source_corpus:
                unsupported.append(f"metric '{m}'")

    # 2. Verify technology tools
    original_tech = extract_technologies(source_corpus)
    suggested_tech = extract_technologies(suggested_rewrite)

    for tech in suggested_tech:
        if tech not in original_tech:
            unsupported.append(f"technology '{tech}'")

    # 3. Optional second-pass LLM Judge
    if enable_llm_judge and settings.ENABLE_LLM_VERIFIER:
        try:
            from app.llm import chat
            prompt = (
                "You are a strict resume verification auditor.\n"
                f"Candidate source text: \"{original_bullet}\"\n"
                f"Suggested rewrite: \"{suggested_rewrite}\"\n"
                "Does the suggested rewrite introduce ANY facts, achievements, or tools not present in the source? "
                "Reply with a JSON list of short strings describing any fabricated claim, or [] if completely grounded."
            )
            resp = chat(messages=[{"role": "user", "content": prompt}], temperature=0.0)
            if resp.content and resp.content.strip().startswith("["):
                import json
                llm_claims = json.loads(resp.content.strip())
                if isinstance(llm_claims, list):
                    for c in llm_claims:
                        if isinstance(c, str) and c not in unsupported:
                            unsupported.append(c)
        except Exception as exc:
            logger.debug(f"LLM verifier second-pass skipped: {exc}")

    return sorted(list(set(unsupported)))

"""
Unit tests for app.verifier.
Tests extraction of numbers, percentages, and technologies, and asserts
that fabricated metrics and unmentioned tools are flagged under unsupported_claims.
"""
from app.verifier import extract_metrics, extract_technologies, verify_rewrite


def test_extract_metrics():
    text = "Improved system throughput by 45% and reduced p99 latency to 12.5ms across 10k users."
    metrics = extract_metrics(text)
    assert "45%" in metrics
    assert "12.5ms" in metrics
    assert "10k" in metrics


def test_extract_technologies():
    text = "Developed a microservice in Python using FastAPI, Docker, and PostgreSQL."
    techs = extract_technologies(text)
    assert "python" in techs
    assert "fastapi" in techs
    assert "docker" in techs
    assert "postgresql" in techs
    assert "kubernetes" not in techs


def test_verify_rewrite_clean_grounded():
    """Rewrite introduces no new tools or numbers; should report zero unsupported claims."""
    original = "Built REST API using Python and FastAPI."
    suggested = "Architected asynchronous REST endpoints leveraging Python and FastAPI."
    profile = "Skills: Python, FastAPI, SQL. Education: BS Computer Science."

    claims = verify_rewrite(
        original_bullet=original,
        suggested_rewrite=suggested,
        profile_text=profile
    )
    assert claims == []


def test_verify_rewrite_flags_fabricated_metric():
    """Rewrite introduces 'reduced latency by 40%' when no metric was in the original bullet."""
    original = "Built REST API using Python."
    suggested = "Built REST API in Python, reducing latency by 40%."
    profile = "Skills: Python."

    claims = verify_rewrite(
        original_bullet=original,
        suggested_rewrite=suggested,
        profile_text=profile
    )
    assert any("metric '40%'" in c for c in claims)


def test_verify_rewrite_flags_fabricated_technology():
    """Rewrite introduces 'Kubernetes' which was neither in the bullet nor in the profile."""
    original = "Containerized services using Docker."
    suggested = "Containerized services using Docker and orchestrated deployments on Kubernetes."
    profile = "Skills: Docker, Python."

    claims = verify_rewrite(
        original_bullet=original,
        suggested_rewrite=suggested,
        profile_text=profile
    )
    assert any("technology 'kubernetes'" in c for c in claims)

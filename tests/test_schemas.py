"""
Unit tests for Pydantic domain schemas.
Ensures strict field validation and correct model serialization.
"""
import pytest
from pydantic import ValidationError
from app.schemas import Project, Profile, Requirement, Gap, Rewrite


def test_project_schema():
    p = Project(name="Task Engine", tech=["Python", "Redis"], bullets=["Engineered job queue."])
    assert p.name == "Task Engine"
    assert "Redis" in p.tech
    assert len(p.bullets) == 1


def test_profile_schema():
    prof = Profile(
        name="Morgan Lee",
        skills=["Python", "SQL"],
        projects=[],
        education=["B.S. in Software Engineering"]
    )
    assert prof.name == "Morgan Lee"
    assert len(prof.skills) == 2


def test_requirement_importance_validation():
    # Valid values: 'must' or 'nice'
    r1 = Requirement(text="FastAPI proficiency", importance="must")
    assert r1.importance == "must"

    r2 = Requirement(text="GraphQL bonus", importance="nice")
    assert r2.importance == "nice"

    # Invalid importance value should raise ValidationError
    with pytest.raises(ValidationError):
        Requirement(text="Invalid field", importance="critical")


def test_gap_schema():
    req = Requirement(text="Python 3.10", importance="must")
    gap = Gap(
        requirement=req,
        status="strong",
        evidence="3 years of Python experience",
        advice="Highlight specific backend frameworks."
    )
    assert gap.status == "strong"
    assert gap.evidence is not None


def test_rewrite_schema():
    rw = Rewrite(
        original="Wrote python backend code.",
        suggested="Engineered scalable asynchronous REST endpoints using Python and FastAPI.",
        reason="Demonstrates technical specificity and clarity.",
        unsupported_claims=["metric '50%'"]
    )
    assert len(rw.unsupported_claims) == 1
    assert "metric '50%'" in rw.unsupported_claims

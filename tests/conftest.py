"""
Pytest configuration and shared fixtures.
Uses in-memory SQLite so all tests run cleanly and pass without Ollama running.
"""
import os
import tempfile
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.schemas import Profile, Project, Requirement
from generate_sample_pdf import create_sample_pdf


@pytest.fixture(scope="function")
def db_session():
    """Provides an isolated in-memory SQLite database session for each test."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def sample_profile():
    """Provides a realistic candidate Profile for testing."""
    return Profile(
        name="Alex Mercer",
        skills=["Python", "FastAPI", "SQLite", "Docker", "PyTorch", "pytest"],
        projects=[
            Project(
                name="AI Career Assistant",
                tech=["Python", "FastAPI", "Docker"],
                bullets=[
                    "Designed REST API serving RAG queries over 1,500 internship postings.",
                    "Built vector embedding search using MiniLM embeddings."
                ]
            )
        ],
        education=["B.S. in Computer Science, University of Washington"]
    )


@pytest.fixture
def sample_requirements():
    """Provides sample job requirements."""
    return [
        Requirement(text="Proficiency in Python and FastAPI backend development", importance="must"),
        Requirement(text="Experience with Docker containerization", importance="must"),
        Requirement(text="Experience with Kubernetes orchestration", importance="nice")
    ]


@pytest.fixture
def temp_pdf_file():
    """Creates a temporary PDF file and removes it after test execution."""
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    create_sample_pdf(path)
    yield path
    if os.path.exists(path):
        os.remove(path)

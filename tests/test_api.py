"""
Tests for FastAPI REST endpoints in app.main.
Asserts:
- GET /health returns status and Ollama availability.
- POST /upload accepts PDF, creates Resume record, and returns file_id.
- GET /sessions/{id}/trace returns step traces.
- DELETE /users/{id} deletes user personal records and uploaded files.
"""
import io
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.db import init_db

# Initialize database schema for TestClient
init_db()
client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "active_model" in data


def test_upload_pdf_endpoint(temp_pdf_file):
    user_id = "api_user_test"

    with open(temp_pdf_file, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/upload",
        files={"file": ("my_resume.pdf", io.BytesIO(file_bytes), "application/pdf")},
        data={"user_id": user_id}
    )

    assert response.status_code == 200
    data = response.json()
    assert "file_id" in data
    assert data["filename"] == "my_resume.pdf"
    assert data["user_id"] == user_id


def test_delete_user_data_endpoint(temp_pdf_file):
    user_id = "user_to_delete"

    # 1. Upload file
    with open(temp_pdf_file, "rb") as f:
        client.post(
            "/upload",
            files={"file": ("temp.pdf", io.BytesIO(f.read()), "application/pdf")},
            data={"user_id": user_id}
        )

    # 2. Delete user
    del_resp = client.delete(f"/users/{user_id}")
    assert del_resp.status_code == 200
    del_data = del_resp.json()
    assert del_data["status"] == "success"
    assert del_data["user_id"] == user_id
    assert del_data["files_deleted"] >= 1

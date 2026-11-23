"""
Tests for resume text extraction and structured parsing in app.tools.parse_resume.
Asserts:
1. pdfplumber extracts text accurately from a PDF file.
2. Structured output parsing returns a Profile object.
3. On Pydantic validation failure, chat() retries ONCE with feedback injected.
"""
from unittest.mock import patch, MagicMock
from app.schemas import Profile, NormalizedResponse
from app.tools.parse_resume import extract_text_from_pdf, parse_resume_text_to_profile


def test_extract_text_from_pdf(temp_pdf_file):
    text = extract_text_from_pdf(temp_pdf_file)
    assert "Alex Mercer" in text
    assert "FastAPI" in text
    assert "EDUCATION" in text


def test_parse_resume_structured_output():
    valid_profile_json = """{
        "name": "Alex Mercer",
        "skills": ["Python", "Docker", "SQL"],
        "projects": [
            {
                "name": "Career Assistant",
                "tech": ["Python", "FastAPI"],
                "bullets": ["Engineered RAG backend."]
            }
        ],
        "education": ["BS in Computer Science"]
    }"""

    mock_resp = NormalizedResponse(
        content=valid_profile_json,
        prompt_tokens=100,
        completion_tokens=60,
        total_tokens=160
    )

    with patch("app.tools.parse_resume.chat", return_value=mock_resp):
        prof = parse_resume_text_to_profile("Sample resume text")
        assert prof.name == "Alex Mercer"
        assert "Docker" in prof.skills
        assert len(prof.projects) == 1
        assert prof.projects[0].name == "Career Assistant"


def test_parse_resume_validation_retry_loop():
    """
    Simulates:
    1st call returns invalid schema JSON (missing required 'name' field).
    Retry call returns corrected valid Profile JSON.
    Asserts retry was triggered and final Profile is valid.
    """
    invalid_json = '{"skills": ["Python"]}'  # Missing required 'name'
    valid_json = '{"name": "Jordan Lee", "skills": ["Python"], "projects": [], "education": []}'

    resp_fail = NormalizedResponse(content=invalid_json, prompt_tokens=40, completion_tokens=10, total_tokens=50)
    resp_success = NormalizedResponse(content=valid_json, prompt_tokens=60, completion_tokens=20, total_tokens=80)

    # Use side_effect to return fail on first raw chat, success on second
    with patch("app.llm._do_raw_chat", side_effect=[resp_fail, resp_success]) as mock_raw_chat:
        from app.llm import chat
        # Calling chat with Profile response_schema
        result = chat(
            messages=[{"role": "user", "content": "Extract resume"}],
            response_schema=Profile
        )

        # Assert exactly 2 calls occurred (original + 1 retry)
        assert mock_raw_chat.call_count == 2
        # Assert the result is now valid Profile
        parsed = Profile.model_validate_json(result.content)
        assert parsed.name == "Jordan Lee"

"""
Configuration settings for Resume Tailor Agent.
Loads configuration from environment variables or .env file using pydantic-settings.
"""
from typing import Tuple
import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings.
    Default to local Ollama endpoints and SQLite for zero-setup local development.
    """
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Ollama Host & Models
    # When running inside Docker, host.docker.internal allows reaching host Ollama
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b"
    EMBED_MODEL: str = "nomic-embed-text"

    # Database configuration (Postgres-ready: set DATABASE_URL=postgresql://user:pass@host/db)
    DATABASE_URL: str = "sqlite:///./resume_tailor.db"

    # Uploaded resume files directory
    UPLOAD_DIR: str = "./uploads"

    # Agent Loop Guardrails
    MAX_STEPS: int = 8
    MAX_SESSION_TOKENS: int = 16000

    # Verifier Settings
    ENABLE_LLM_VERIFIER: bool = False

    # Generation Temperature Controls
    # Temperature 0 for deterministic parsing/extraction, low for rewriting
    TEMPERATURE_PARSE: float = 0.0
    TEMPERATURE_REWRITE: float = 0.2


settings = Settings()


def check_ollama_health(host: str = settings.OLLAMA_HOST, model: str = settings.OLLAMA_MODEL, embed_model: str = settings.EMBED_MODEL) -> Tuple[bool, bool, bool, str]:
    """
    Performs a lightweight startup check against the Ollama REST API.
    Returns:
        (is_reachable, has_main_model, has_embed_model, message)
    Why:
        Small local models and Ollama services may not be running or have the specified
        model pulled yet. Surfacing an actionable message early prevents cryptic failures.
    """
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{host.rstrip('/')}/api/tags")
            if resp.status_code != 200:
                return False, False, False, f"Ollama returned HTTP {resp.status_code} at {host}."

            data = resp.json()
            models = [m.get("name") for m in data.get("models", [])]

            # Model names might include tags (e.g., 'llama3.2:latest' vs 'llama3.2')
            has_main = any(model in m or m in model for m in models)
            has_embed = any(embed_model in m or m in embed_model for m in models)

            if not has_main:
                available_str = ", ".join(models) if models else "None"
                msg = (
                    f"Model '{model}' is not pulled in Ollama at {host}.\n"
                    f"Available models: {available_str}\n"
                    f"Run: ollama pull {model}"
                )
                return True, False, has_embed, msg

            return True, True, has_embed, "Ollama is healthy and ready."

    except Exception as exc:
        return False, False, False, f"Could not connect to Ollama at {host}: {exc}"

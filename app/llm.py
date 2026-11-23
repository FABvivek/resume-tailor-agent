"""
LLM abstraction layer for Resume Tailor Agent.
Directly interfaces with the Ollama REST API via httpx (no heavy SDK dependencies).
Handles:
1. Native tool calling via Ollama's POST /api/chat with full GPU offloading (num_gpu: 99).
2. Structured output generation with Pydantic JSON schemas via the 'format' field.
3. 1-time retry loop on Pydantic validation failures with detailed feedback injected.
4. Resilience for local models (temperature control, fallback parsing, timeout safeguards).
5. GPU-accelerated embeddings (Ollama nomic-embed-text / sentence-transformers on CUDA).
"""
import json
import logging
import re
import time
from typing import List, Dict, Any, Optional, Type
import httpx
from pydantic import BaseModel, ValidationError

from app.config import settings
from app.schemas import NormalizedResponse, ToolCall

# Configure module-level logger
logger = logging.getLogger(__name__)

# Cached fallback embedding model instance to avoid re-initializing weights on every call
_FALLBACK_EMBEDDER = None


def _get_fallback_embedder():
    """
    Lazy-loads sentence-transformers 'all-MiniLM-L6-v2' on CUDA GPU if available.
    """
    global _FALLBACK_EMBEDDER
    if _FALLBACK_EMBEDDER is None:
        try:
            import torch
            from sentence_transformers import SentenceTransformer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Initializing SentenceTransformer ('all-MiniLM-L6-v2') on device: {device}.")
            _FALLBACK_EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        except Exception as exc:
            logger.warning(f"Could not load SentenceTransformer: {exc}")
    return _FALLBACK_EMBEDDER


def get_embedding(text: str, host: str = settings.OLLAMA_HOST, model: str = settings.EMBED_MODEL) -> List[float]:
    """
    Generates vector embeddings for a given text string.
    First tries Ollama's /api/embed or /api/embeddings endpoint.
    If the model is missing or Ollama is offline, falls back to sentence-transformers on CUDA.
    """
    clean_text = text.strip()
    if not clean_text:
        return [0.0] * 384  # Standard MiniLM dimension

    # Attempt 1: Ollama native embedding REST API
    try:
        with httpx.Client(timeout=10.0) as client:
            url = f"{host.rstrip('/')}/api/embed"
            payload = {"model": model, "input": clean_text}
            resp = client.post(url, json=payload)

            if resp.status_code == 404:
                url = f"{host.rstrip('/')}/api/embeddings"
                payload = {"model": model, "prompt": clean_text}
                resp = client.post(url, json=payload)

            if resp.status_code == 200:
                data = resp.json()
                if "embeddings" in data and len(data["embeddings"]) > 0:
                    return data["embeddings"][0]
                elif "embedding" in data:
                    return data["embedding"]
    except Exception as exc:
        logger.debug(f"Ollama embedding lookup failed ({exc}). Proceeding to sentence-transformers fallback.")

    # Attempt 2: Local SentenceTransformer on GPU
    embedder = _get_fallback_embedder()
    if embedder is not None:
        try:
            vector = embedder.encode(clean_text, normalize_embeddings=True)
            return vector.tolist()
        except Exception as exc:
            logger.error(f"Fallback SentenceTransformer failed: {exc}")

    return [0.0] * 384


def _clean_json_markdown(text: str) -> str:
    """Removes ```json ... ``` code fence wrappers often emitted by local LLMs."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _do_raw_chat(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    response_schema: Optional[Type[BaseModel]] = None,
    temperature: Optional[float] = None,
    host: str = settings.OLLAMA_HOST,
    model: str = settings.OLLAMA_MODEL,
    timeout: float = 120.0
) -> NormalizedResponse:
    """
    Executes a single HTTP POST request to Ollama's /api/chat endpoint with GPU acceleration.
    """
    start_time = time.time()
    url = f"{host.rstrip('/')}/api/chat"

    # Default temperatures: 0.0 for structured output parsing, 0.2 for conversational rewrite
    if temperature is None:
        temperature = settings.TEMPERATURE_PARSE if response_schema else settings.TEMPERATURE_REWRITE

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_gpu": 99  # Instruct Ollama to offload all model layers to NVIDIA GPU VRAM
        }
    }

    if tools:
        payload["tools"] = tools

    if response_schema:
        try:
            payload["format"] = response_schema.model_json_schema()
        except Exception:
            payload["format"] = "json"

    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, json=payload)

    latency_ms = (time.time() - start_time) * 1000.0

    if resp.status_code != 200:
        err_msg = f"Ollama API error {resp.status_code}: {resp.text}"
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    data = resp.json()
    msg_data = data.get("message", {})
    content = msg_data.get("content", "")

    # Token counts provided natively by Ollama
    prompt_tokens = data.get("prompt_eval_count", 0)
    completion_tokens = data.get("eval_count", 0)
    total_tokens = prompt_tokens + completion_tokens

    # Parse tool calls
    parsed_tool_calls: List[ToolCall] = []
    raw_tool_calls = msg_data.get("tool_calls", [])

    for idx, tc in enumerate(raw_tool_calls):
        func_dict = tc.get("function", {})
        call_name = func_dict.get("name", "")
        call_args = func_dict.get("arguments", {})

        if isinstance(call_args, str):
            try:
                call_args = json.loads(call_args)
            except Exception:
                call_args = {"raw_input": call_args}

        parsed_tool_calls.append(
            ToolCall(
                id=f"call_{idx}_{int(time.time()*1000)}",
                name=call_name,
                arguments=call_args
            )
        )

    return NormalizedResponse(
        content=content,
        tool_calls=parsed_tool_calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_ms=round(latency_ms, 2),
        raw=data
    )


def chat(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    response_schema: Optional[Type[BaseModel]] = None,
    temperature: Optional[float] = None,
    host: str = settings.OLLAMA_HOST,
    model: str = settings.OLLAMA_MODEL,
    timeout: float = 120.0
) -> NormalizedResponse:
    """
    Primary interface for all agent and tool interactions.
    Calls Ollama REST API. If a response_schema is specified, it validates the result.
    On validation failure, retries ONCE feeding the validation error back to the model.
    """
    initial_resp = _do_raw_chat(
        messages=messages,
        tools=tools,
        response_schema=response_schema,
        temperature=temperature,
        host=host,
        model=model,
        timeout=timeout
    )

    if response_schema is None:
        return initial_resp

    content_to_parse = _clean_json_markdown(initial_resp.content or "")
    try:
        response_schema.model_validate_json(content_to_parse)
        return initial_resp
    except (ValidationError, Exception) as val_err:
        logger.warning(
            f"[Structured Output Retry] Schema validation failed for '{response_schema.__name__}': {val_err}. "
            f"Raw content was: {initial_resp.content!r}. Retrying ONCE with feedback injected."
        )

        feedback_messages = list(messages)
        feedback_messages.append({"role": "assistant", "content": initial_resp.content or "{}"})
        feedback_messages.append({
            "role": "user",
            "content": (
                f"Your previous response failed Pydantic schema validation for '{response_schema.__name__}':\n"
                f"{str(val_err)}\n"
                f"Please correct the JSON output. Return ONLY the valid JSON object matching the schema."
            )
        })

        retry_resp = _do_raw_chat(
            messages=feedback_messages,
            tools=tools,
            response_schema=response_schema,
            temperature=0.0,
            host=host,
            model=model,
            timeout=timeout
        )

        retry_resp.latency_ms += initial_resp.latency_ms
        retry_resp.prompt_tokens += initial_resp.prompt_tokens
        retry_resp.completion_tokens += initial_resp.completion_tokens
        retry_resp.total_tokens += initial_resp.total_tokens

        retry_content = _clean_json_markdown(retry_resp.content or "")
        try:
            response_schema.model_validate_json(retry_content)
            logger.info(f"[Structured Output Retry] Schema validation succeeded on retry for '{response_schema.__name__}'.")
        except Exception as second_err:
            logger.error(f"[Structured Output Retry] Schema validation failed again on retry: {second_err}.")

        return retry_resp

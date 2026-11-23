"""
Pydantic v2 domain schemas and data transfer objects for Resume Tailor Agent.
"""
from typing import Literal, Optional, Any, List, Dict
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. Domain Entities specified in product requirements
# ---------------------------------------------------------------------------

class Project(BaseModel):
    """Represents a project or experience extracted from a candidate's resume."""
    name: str = Field(description="Name or title of the project or initiative")
    tech: List[str] = Field(..., description="Technologies, frameworks, or tools used")
    bullets: List[str] = Field(..., description="Descriptive accomplishment bullets")


class Profile(BaseModel):
    """Structured representation of a parsed resume."""
    name: str = Field(description="Candidate's full name")
    skills: List[str] = Field(..., description="Explicit or inferred candidate skills")
    projects: List[Project] = Field(..., description="List of notable projects or roles")
    education: List[str] = Field(..., description="Educational background items")


class Requirement(BaseModel):
    """A single requirement extracted from a Job Description."""
    text: str = Field(description="Normalized requirement statement")
    importance: Literal["must", "nice"] = Field(
        description="Whether this is a non-negotiable must-have or a nice-to-have bonus"
    )


class Gap(BaseModel):
    """Analysis of candidate qualification against a job requirement."""
    requirement: Requirement = Field(description="The target JD requirement")
    status: Literal["strong", "partial", "missing"] = Field(
        description="Candidate's match status based on resume evidence"
    )
    evidence: Optional[str] = Field(
        default=None,
        description="Best matching evidence sentence/bullet from the resume, or None if missing"
    )
    advice: str = Field(
        description="Actionable guidance on how to address or frame this gap honestly"
    )


class Rewrite(BaseModel):
    """An honest bullet rewrite grounded strictly in resume facts."""
    original: str = Field(description="Original bullet from the resume")
    suggested: str = Field(description="Rewritten bullet aligned to the target requirement")
    reason: str = Field(description="Explanation of why this rewrite is stronger")
    unsupported_claims: List[str] = Field(
        default_factory=list,
        description="Any newly introduced metrics, tools, or facts not present in original profile"
    )


# ---------------------------------------------------------------------------
# 2. Tool Calling and LLM Normalized Interfaces
# ---------------------------------------------------------------------------

class ToolCall(BaseModel):
    """Normalized tool call emitted by an LLM."""
    id: str = Field(description="Unique tool call identifier")
    name: str = Field(description="Function name to execute")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Parsed JSON arguments")


class NormalizedResponse(BaseModel):
    """
    Standard response format returned by app.llm.chat across all model providers.
    Why: Decouples the hand-written agent loop from provider-specific response shapes.
    """
    content: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    raw: Optional[Any] = None


# ---------------------------------------------------------------------------
# 3. API Request / Response schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str = Field(description="Conversation session identifier")
    user_id: str = Field(description="User identifier for persistence")
    message: str = Field(description="User prompt or instructions")
    file_id: Optional[str] = Field(default=None, description="Optional uploaded resume ID")
    jd_text: Optional[str] = Field(default=None, description="Optional target job description text")


class ChatResponse(BaseModel):
    response: str
    session_id: str
    gaps: List[Gap] = Field(default_factory=list)
    rewrites: List[Rewrite] = Field(default_factory=list)
    learning_plan: str = ""
    path_used: str = Field(default="tool_calling", description="'tool_calling' or 'deterministic_fallback'")


class TraceRecord(BaseModel):
    id: int
    session_id: str
    step: int
    tool: str
    args: Dict[str, Any]
    latency_ms: float
    tokens: int
    error: Optional[str] = None
    path_used: str


class HealthResponse(BaseModel):
    status: str
    ollama_reachable: bool
    model_available: bool
    active_model: str
    embedding_model_available: bool
    message: str

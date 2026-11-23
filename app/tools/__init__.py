"""
Tools registry and schema definitions for the hand-written agent loop.
Exposes native Ollama tool definitions conforming to the JSON schema format.
"""
from typing import Dict, Any, List

from app.tools.parse_resume import parse_resume
from app.tools.analyze_jd import analyze_jd
from app.tools.match_skills import match_skills
from app.tools.suggest_rewrite import suggest_rewrite
from app.tools.memory_tools import save_user_preference, recall_user_preferences

# Native tool calling definitions in Ollama / OpenAI function format
AGENT_TOOLS_SPEC: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "parse_resume",
            "description": "Extracts structured candidate Profile (skills, projects, education) from an uploaded resume file ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "string",
                        "description": "The unique identifier of the uploaded resume PDF."
                    }
                },
                "required": ["file_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_jd",
            "description": "Analyzes job description text and extracts list of must-have and nice-to-have requirements.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Full job description text."
                    }
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "match_skills",
            "description": "Performs embedding-based cosine similarity and gap analysis between parsed candidate Profile and JD requirements.",
            "parameters": {
                "type": "object",
                "properties": {
                    "profile": {
                        "type": "object",
                        "description": "Parsed candidate Profile dictionary."
                    },
                    "requirements": {
                        "type": "array",
                        "description": "List of Requirement dictionaries.",
                        "items": {"type": "object"}
                    }
                },
                "required": ["profile", "requirements"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_rewrite",
            "description": "Suggests an honest rewrite for a specific resume bullet aligned with a target requirement.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bullet": {
                        "type": "string",
                        "description": "Original accomplishment bullet from the resume."
                    },
                    "requirement": {
                        "type": "object",
                        "description": "Target Requirement dictionary."
                    },
                    "profile_context": {
                        "type": "string",
                        "description": "Optional background skills or project context from the profile."
                    }
                },
                "required": ["bullet", "requirement"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_preference",
            "description": "Saves a user preference or career goal across sessions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Preference key (e.g. 'target_role', 'preferred_tone')."
                    },
                    "value": {
                        "type": "string",
                        "description": "Value to store."
                    }
                },
                "required": ["key", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recall_preferences",
            "description": "Recalls all previously saved user goals and preferences.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]

__all__ = [
    "parse_resume",
    "analyze_jd",
    "match_skills",
    "suggest_rewrite",
    "save_user_preference",
    "recall_user_preferences",
    "AGENT_TOOLS_SPEC"
]

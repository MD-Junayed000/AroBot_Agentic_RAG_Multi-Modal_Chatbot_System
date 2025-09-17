# core/prompts/__init__.py
"""System prompts and templates"""

from .system import get_system_prompt, format_medical_prompt, DEFAULT_SYSTEM

__all__ = ["get_system_prompt", "format_medical_prompt", "DEFAULT_SYSTEM"]

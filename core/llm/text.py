# core/llm/text.py
"""Text generation module - focused on core LLM interactions"""

import logging
import ollama
from typing import Optional
from config.env_config import OLLAMA_BASE_URL, OLLAMA_TEXT_MODEL

logger = logging.getLogger(__name__)

class TextGenerator:
    """Handles text-only LLM interactions"""
    
    def __init__(self):
        self.client = ollama.Client(host=OLLAMA_BASE_URL)
        self.text_model = OLLAMA_TEXT_MODEL
    
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text response from LLM"""
        try:
            system = system_prompt or "You are a helpful medical assistant. Be concise and accurate."
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ]
            r = self.client.chat(
                model=self.text_model,
                messages=messages,
                stream=False,
                options={"temperature": 0.2, "num_ctx": 4096, "num_predict": 250, "top_p": 0.9},
                keep_alive="30m"
            )
            return r.get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Text generation error: {e}")
            return f"Error generating response: {str(e)}"
    
    def fast_format(self, prompt: str) -> str:
        """Fast formatting for quick responses"""
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Be concise."},
                {"role": "user", "content": prompt}
            ]
            r = self.client.chat(
                model=self.text_model,
                messages=messages,
                stream=False,
                options={"temperature": 0.1, "num_ctx": 1024, "num_predict": 200, "top_p": 0.9},
                keep_alive="30m"
            )
            return r.get("message", {}).get("content", "")
        except Exception:
            return self.generate_response(prompt)
    
    def greeting_response(self, text: str = "") -> str:
        """Generate contextual greeting"""
        greetings = text.lower().strip() if text else ""
        
        if "good morning" in greetings:
            start = "Good morning!"
        elif "good evening" in greetings:
            start = "Good evening!"
        elif "good afternoon" in greetings:
            start = "Good afternoon!"
        else:
            start = "Hello!"
            
        return (
            f"{start} I'm your medical assistant. I can help with medicine information, "
            "prescription analysis, and general health questions. What can I help you with?"
        )
    
    def about_response(self) -> str:
        """About/capabilities response"""
        return (
            "I'm an AI medical assistant providing evidence-based guidance. "
            "I can answer medicine questions, analyze prescriptions, review documents, "
            "and provide Bangladesh-specific medicine information. "
            "Always consult healthcare professionals for medical decisions."
        )
    
    @staticmethod
    def safety_footer(text: str) -> str:
        """Add safety warnings for urgent symptoms"""
        urgent_terms = ["chest pain", "stroke", "shortness of breath", "emergency"]
        if any(term in text.lower() for term in urgent_terms):
            return text + "\n\n**⚠️ For urgent symptoms, seek immediate medical care.**"
        return text

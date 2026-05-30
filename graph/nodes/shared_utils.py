from dotenv import load_dotenv
load_dotenv()
import logging
from langchain_groq import ChatGroq
import os
import json
import re 
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Model fallback chain — best quality first, smallest/fastest last
_FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.1-8b-instant",
]
_GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def _invoke_with_fallback(prompt: str, max_tokens: int = 1024) -> str:
    """
    Try each model in _FALLBACK_MODELS in order.
    Falls back on 413 (too large) or 429 (rate limit).
    Raises RuntimeError if all models fail.
    """
    last_error = None
    for model in _FALLBACK_MODELS:
        try:
            llm = ChatGroq(model=model, api_key=_GROQ_API_KEY, max_tokens=max_tokens)
            response = llm.invoke(prompt)
            if model != _FALLBACK_MODELS[0]:
                logger.warning(f"_invoke_with_fallback: used fallback model {model}")
            return response.content
        except Exception as e:
            err_str = str(e)
            if "413" in err_str or "429" in err_str or "rate_limit" in err_str or "too large" in err_str.lower():
                logger.warning(f"_invoke_with_fallback: {model} failed ({type(e).__name__}), trying next...")
                last_error = e
                continue
            raise  # non-rate-limit errors bubble up immediately
    raise RuntimeError(f"All models exhausted. Last error: {last_error}")

# ════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ════════════════════════════════════════════════════════════════

def _extract_urls(text: str) -> list[str]:
    return re.findall(r'https?://[^\s\)\"\']+', text)


def _parse_llm_json(raw: str) -> dict:
    """Strip markdown fences then parse JSON — tolerant of LLM formatting habits."""
    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    return json.loads(cleaned)


def _extract_exa_score(text: str) -> float:
    """Pull the Exa-Score we embedded in the formatted result string."""
    match = re.search(r'Exa-Score:\s*([0-9.]+)', text)
    return float(match.group(1)) if match else 0.5


def _is_homepage(url: str) -> bool:
    """Catch cases where Exa returns a site homepage instead of a specific article."""
    try:
        path = urlparse(url).path.rstrip("/")
        homepage_patterns = {"", "/home", "/index", "/en/home", "/ext/en/home", "/en"}
        return path in homepage_patterns
    except Exception:
        return False
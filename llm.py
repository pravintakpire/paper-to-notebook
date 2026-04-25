"""Unified LLM client supporting OpenAI, Gemini, and local (Ollama-compatible) providers."""
from __future__ import annotations

import json
import os
import time
from io import BytesIO
from typing import Callable, Optional

from pypdf import PdfReader

from config import (
    DEFAULT_LOCAL_BASE_URL,
    MAX_RETRIES,
    PROVIDER_GEMINI,
    PROVIDER_LOCAL,
    PROVIDER_OPENAI,
    RETRY_DELAYS,
)


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

def load_pdf_text(pdf_path: str) -> str:
    """Read a PDF file and return its text content."""
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_pdf_text_from_bytes(pdf_bytes: bytes) -> str:
    """Read PDF bytes and return text content."""
    reader = PdfReader(BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def detect_provider(model: str = "", api_key: str = "") -> str:
    """Infer provider from model name prefix or environment variables."""
    m = (model or "").lower()
    if m.startswith("gemini"):
        return PROVIDER_GEMINI
    if m.startswith(("gpt-", "o1", "o3", "o4")):
        return PROVIDER_OPENAI
    if os.environ.get("GEMINI_API_KEY"):
        return PROVIDER_GEMINI
    if os.environ.get("OPENAI_API_KEY"):
        return PROVIDER_OPENAI
    return PROVIDER_LOCAL


# ---------------------------------------------------------------------------
# Core LLM call (single attempt)
# ---------------------------------------------------------------------------

def call_llm(
    system_prompt: str,
    user_content: str,
    max_tokens: int = 4096,
    model: str = "gpt-4o",
    provider: str = PROVIDER_OPENAI,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    on_thinking: Optional[Callable[[str], None]] = None,
) -> str:
    """Dispatch a single LLM call to the chosen provider."""
    if provider == PROVIDER_GEMINI:
        return _call_gemini(system_prompt, user_content, max_tokens, model, api_key, on_thinking)
    if provider == PROVIDER_LOCAL:
        return _call_openai_compat(
            system_prompt, user_content, max_tokens, model,
            api_key or "ollama",
            base_url or DEFAULT_LOCAL_BASE_URL,
            on_thinking,
        )
    return _call_openai_compat(
        system_prompt, user_content, max_tokens, model,
        api_key or os.environ.get("OPENAI_API_KEY", ""),
        None,
        on_thinking,
    )


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

def _call_openai_compat(
    system_prompt: str,
    user_content: str,
    max_tokens: int,
    model: str,
    api_key: str,
    base_url: Optional[str],
    on_thinking: Optional[Callable[[str], None]],
) -> str:
    import openai

    kwargs: dict = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = openai.OpenAI(**kwargs)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    if on_thinking:
        full_text = ""
        stream = client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=0.7, stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                on_thinking(delta)
                full_text += delta
        return full_text

    response = client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=0.7,
    )
    return response.choices[0].message.content


def _call_gemini(
    system_prompt: str,
    user_content: str,
    max_tokens: int,
    model: str,
    api_key: Optional[str],
    on_thinking: Optional[Callable[[str], None]],
) -> str:
    from google import genai
    from google.genai import types

    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    client = genai.Client(api_key=key)

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=max_tokens,
        temperature=0.7,
    )

    if on_thinking:
        full_text = ""
        for chunk in client.models.generate_content_stream(
            model=model, contents=user_content, config=config
        ):
            if chunk.text:
                on_thinking(chunk.text)
                full_text += chunk.text
        return full_text

    response = client.models.generate_content(
        model=model, contents=user_content, config=config
    )
    return response.text


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------

def call_llm_with_retry(
    system_prompt: str,
    user_content: str,
    max_tokens: int = 4096,
    model: str = "gpt-4o",
    provider: str = PROVIDER_OPENAI,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    on_thinking: Optional[Callable[[str], None]] = None,
) -> str:
    """Call LLM with retry logic for transient errors."""
    last_error: Optional[Exception] = None

    for attempt in range(MAX_RETRIES):
        try:
            return call_llm(
                system_prompt, user_content, max_tokens,
                model, provider, api_key, base_url, on_thinking,
            )
        except Exception as e:
            error_str = str(e).lower()
            if any(kw in error_str for kw in ["429", "rate", "500", "503", "overloaded", "unavailable", "resource_exhausted"]):
                last_error = e
                wait = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                print(f"  Transient error. Waiting {wait}s before retry {attempt + 1}/{MAX_RETRIES}...")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries. Last error: {last_error}")


# ---------------------------------------------------------------------------
# JSON parsing with repair
# ---------------------------------------------------------------------------

def parse_llm_json(
    raw_text: str,
    step_name: str,
    model: str,
    provider: str = PROVIDER_OPENAI,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> dict | list:
    """Parse JSON from LLM response, with one repair attempt on failure."""
    text = raw_text.strip()

    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  Warning: JSON parse failed in {step_name}. Attempting repair...")
        repair_prompt = (
            f"The following text was supposed to be valid JSON but has a syntax error:\n\n"
            f"{text[:4000]}\n\nError: {e}\n\n"
            f"Return ONLY the corrected valid JSON, nothing else."
        )
        repaired = call_llm_with_retry(
            system_prompt="You are a JSON repair tool. Return only valid JSON.",
            user_content=repair_prompt,
            max_tokens=max(len(text) // 2, 4096),
            model=model,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
        )
        repaired = repaired.strip()
        if repaired.startswith("```"):
            repaired = repaired.split("\n", 1)[1]
        if repaired.endswith("```"):
            repaired = repaired[:-3]
        return json.loads(repaired.strip())

"""Gemini API wrapper with native PDF support and retry logic."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable, Optional

from google import genai
from google.genai import types

from config import MAX_RETRIES, RETRY_DELAYS

_FALLBACK_KEY = "AIzaSyAoT3bvLuhHqnWJ52Nftz6ABidvd1ZQ_nI"


def _get_api_key(api_key: str | None = None) -> str:
    return api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or _FALLBACK_KEY


def load_pdf_as_part(pdf_path: str) -> types.Part:
    """Read a PDF file and return a Gemini API Part for inline data."""
    pdf_bytes = Path(pdf_path).read_bytes()
    return types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")


def call_gemini(
    system_prompt: str,
    user_content: list,
    max_tokens: int = 8192,
    model: str = "gemini-2.5-pro",
    api_key: str | None = None,
    on_thinking: Optional[Callable[[str], None]] = None,
) -> str:
    """Make a Gemini API call and return the text response.

    If on_thinking is provided, uses streaming to capture thinking tokens.
    """
    client = genai.Client(api_key=_get_api_key(api_key))
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
            try:
                if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                    for part in chunk.candidates[0].content.parts:
                        if getattr(part, 'thought', False):
                            if part.text:
                                on_thinking(part.text)
                        else:
                            if part.text:
                                full_text += part.text
            except (AttributeError, IndexError):
                if hasattr(chunk, 'text') and chunk.text:
                    full_text += chunk.text
        return full_text
    else:
        response = client.models.generate_content(
            model=model, contents=user_content, config=config
        )
        return response.text


def call_gemini_with_retry(
    system_prompt: str,
    user_content: list,
    max_tokens: int = 8192,
    model: str = "gemini-2.5-pro",
    api_key: str | None = None,
    on_thinking: Optional[Callable[[str], None]] = None,
) -> str:
    """Call Gemini API with retry logic for transient errors."""
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            return call_gemini(system_prompt, user_content, max_tokens, model, api_key, on_thinking)

        except Exception as e:
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ["429", "rate", "500", "503", "overloaded", "unavailable"]):
                last_error = e
                wait = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                print(f"  Transient error. Waiting {wait}s before retry {attempt + 1}/{MAX_RETRIES}...")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries. Last error: {last_error}")


def parse_llm_json(raw_text: str, step_name: str, model: str, api_key: str | None = None) -> dict | list:
    """Parse JSON from LLM response, with cleanup and one repair attempt."""
    text = raw_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        first_newline = text.index("\n")
        text = text[first_newline + 1:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  Warning: JSON parse failed in {step_name}. Attempting repair...")
        repair_prompt = (
            f"The following text was supposed to be valid JSON but has a syntax error:\n\n"
            f"{text[:4000]}\n\n"
            f"Error: {e}\n\n"
            f"Return ONLY the corrected valid JSON, nothing else."
        )
        repaired = call_gemini_with_retry(
            system_prompt="You are a JSON repair tool. Return only valid JSON.",
            user_content=[repair_prompt],
            max_tokens=max(len(text) // 2, 4096),
            model=model,
            api_key=api_key,
        )
        repaired = repaired.strip()
        if repaired.startswith("```"):
            repaired = repaired.split("\n", 1)[1]
        if repaired.endswith("```"):
            repaired = repaired[:-3]
        return json.loads(repaired.strip())

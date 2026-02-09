"""OpenAI API wrapper with local PDF text extraction and retry logic."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable, Optional

import json
import os
import time
from pathlib import Path
from typing import Callable, Optional

import openai
from pypdf import PdfReader

from config import MAX_RETRIES, RETRY_DELAYS




def _get_api_key(api_key: str | None = None) -> str:
    return api_key or os.environ.get("OPENAI_API_KEY")


def load_pdf_text(pdf_path: str) -> str:
    """Read a PDF file and return its text content."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def load_pdf_text_from_bytes(pdf_bytes: bytes) -> str:
    """Read PDF bytes and return text content."""
    from io import BytesIO
    reader = PdfReader(BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def call_llm(
    system_prompt: str,
    user_content: str,
    max_tokens: int = 4096,
    model: str = "gpt-4o",
    api_key: str | None = None,
    on_thinking: Optional[Callable[[str], None]] = None,
) -> str:
    """Make an OpenAI API call and return the text response.

    If on_thinking is provided, uses streaming to capture chunks (simulating thinking).
    """
    client = openai.OpenAI(api_key=_get_api_key(api_key))
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    if on_thinking:
        full_text = ""
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                text_chunk = chunk.choices[0].delta.content
                on_thinking(text_chunk)
                full_text += text_chunk
        return full_text
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content


def call_llm_with_retry(
    system_prompt: str,
    user_content: str,
    max_tokens: int = 4096,
    model: str = "gpt-4o",
    api_key: str | None = None,
    on_thinking: Optional[Callable[[str], None]] = None,
) -> str:
    """Call OpenAI API with retry logic for transient errors."""
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            return call_llm(system_prompt, user_content, max_tokens, model, api_key, on_thinking)

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
        repaired = call_llm_with_retry(
            system_prompt="You are a JSON repair tool. Return only valid JSON.",
            user_content=repair_prompt,
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

"""Web-compatible pipeline wrapper with progress callbacks."""
from __future__ import annotations

import io
import json
from typing import Callable, Optional

import nbformat

from config import (
    DEFAULT_MODEL,
    PROVIDER_OPENAI,
    MAX_TOKENS_ANALYSIS,
    MAX_TOKENS_DESIGN,
    MAX_TOKENS_GENERATE,
    MAX_TOKENS_VALIDATE,
)
from llm import call_llm_with_retry, load_pdf_text_from_bytes, parse_llm_json
from notebook_builder import build_notebook
from prompts import (
    ANALYSIS_PROMPT,
    DESIGN_PROMPT_TEMPLATE,
    GENERATE_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    VALIDATE_PROMPT_TEMPLATE,
)

ProgressCallback = Callable[[int, str, str, Optional[dict]], None]
ThinkingCallback = Callable[[str], None]


def _nb_to_bytes(nb: nbformat.NotebookNode) -> bytes:
    buffer = io.StringIO()
    nbformat.write(nb, buffer)
    return buffer.getvalue().encode("utf-8")


def _cells_to_bytes(cells: list) -> bytes:
    return _nb_to_bytes(build_notebook(cells))


def run_web_pipeline(
    pdf_bytes: bytes,
    model: str = DEFAULT_MODEL,
    on_progress: Optional[ProgressCallback] = None,
    provider: str = PROVIDER_OPENAI,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    on_thinking: Optional[ThinkingCallback] = None,
) -> bytes:
    """Run the full pipeline on PDF bytes, returning .ipynb bytes."""

    def _notify(step: int, name: str, detail: str = "", extra: Optional[dict] = None):
        if on_progress:
            on_progress(step, name, detail, extra)

    def _llm(system_prompt: str, user_content: str, max_tokens: int) -> str:
        return call_llm_with_retry(
            system_prompt=system_prompt,
            user_content=user_content,
            max_tokens=max_tokens,
            model=model,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            on_thinking=on_thinking,
        )

    def _parse(raw: str, step: str) -> dict | list:
        return parse_llm_json(raw, step, model, provider=provider, api_key=api_key, base_url=base_url)

    pdf_text = load_pdf_text_from_bytes(pdf_bytes)

    # Step 1: Paper Analysis
    _notify(1, "Analyzing paper", "Reading PDF and extracting structure...")
    analysis = _parse(
        _llm(SYSTEM_PROMPT, f"Here is the research paper content:\n\n{pdf_text}\n\nInstructions:\n{ANALYSIS_PROMPT}", MAX_TOKENS_ANALYSIS),
        "paper_analysis",
    )
    title = analysis.get("title", "Unknown Paper")
    num_algos = len(analysis.get("algorithms", []))
    _notify(1, "Analyzing paper", f"Found: {title}", {
        "type": "analysis",
        "title": title,
        "algorithms": num_algos,
        "insight": analysis.get("key_insight", ""),
        "problem": analysis.get("problem_statement", ""),
    })

    # Step 2: Design Plan
    _notify(2, "Designing implementation", "Planning model architecture and training...")
    design_prompt = DESIGN_PROMPT_TEMPLATE.format(analysis_json=json.dumps(analysis, indent=2))
    design = _parse(
        _llm(SYSTEM_PROMPT, f"Here is the research paper content:\n\n{pdf_text}\n\nInstructions:\n{design_prompt}", MAX_TOKENS_DESIGN),
        "toy_design",
    )
    arch = design.get("model_architecture", {})
    _notify(2, "Designing implementation", "Architecture designed", {
        "type": "design",
        "notebook_title": design.get("notebook_title", ""),
        "model_type": arch.get("type", ""),
        "embed_dim": arch.get("embed_dim", ""),
        "num_layers": arch.get("num_layers", ""),
        "num_heads": arch.get("num_heads", ""),
    })

    # Step 3: Generate Notebook Cells
    _notify(3, "Generating notebook", "Writing PyTorch code and explanations...")
    generate_prompt = GENERATE_PROMPT_TEMPLATE.format(
        analysis_json=json.dumps(analysis, indent=2),
        design_json=json.dumps(design, indent=2),
    )
    cells = _parse(
        _llm(SYSTEM_PROMPT, f"Here is the research paper content:\n\n{pdf_text}\n\nInstructions:\n{generate_prompt}", MAX_TOKENS_GENERATE),
        "generate_cells",
    )
    num_cells = len(cells)
    code_cells = sum(1 for c in cells if c.get("cell_type") == "code")
    previews = [{"type": c.get("cell_type", "code"), "preview": c.get("source", "")[:300]} for c in cells]

    draft_bytes = _cells_to_bytes(cells)
    _notify(3, "Generating notebook", f"Generated {num_cells} cells ({code_cells} code)", {
        "type": "cells_generated",
        "num_cells": num_cells,
        "code_cells": code_cells,
        "previews": previews,
        "draft_bytes": draft_bytes,
    })

    # Step 4: Validate & Repair
    _notify(4, "Validating code", "LLM reviewing for errors...")
    validate_prompt = VALIDATE_PROMPT_TEMPLATE.format(cells_json=json.dumps(cells, indent=2))
    validated_cells = _parse(
        _llm(SYSTEM_PROMPT, validate_prompt, MAX_TOKENS_VALIDATE),
        "validate",
    )
    _notify(4, "Validating code", "Validation complete")

    nb = build_notebook(validated_cells)
    return _nb_to_bytes(nb)

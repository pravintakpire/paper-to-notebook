"""Web-compatible pipeline wrapper with progress callbacks."""
from __future__ import annotations

import io
import json
from typing import Callable, Optional

import nbformat


from config import (
    DEFAULT_MODEL,
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

# callback(step_number, step_name, detail_message, extra_data)
ProgressCallback = Callable[[int, str, str, Optional[dict]], None]
ThinkingCallback = Callable[[str], None]


def _nb_to_bytes(nb: nbformat.NotebookNode) -> bytes:
    buffer = io.StringIO()
    nbformat.write(nb, buffer)
    return buffer.getvalue().encode("utf-8")


def _cells_to_bytes(cells: list) -> bytes:
    nb = build_notebook(cells)
    return _nb_to_bytes(nb)


def run_web_pipeline(
    pdf_bytes: bytes,
    model: str = DEFAULT_MODEL,
    on_progress: Optional[ProgressCallback] = None,
    api_key: Optional[str] = None,
    on_thinking: Optional[ThinkingCallback] = None,
) -> bytes:
    """Run the full pipeline on PDF bytes, returning .ipynb bytes."""

    def _notify(step: int, name: str, detail: str = "", extra: Optional[dict] = None):
        if on_progress:
            on_progress(step, name, detail, extra)

    pdf_text = load_pdf_text_from_bytes(pdf_bytes)

    # Step 1: Paper Analysis
    _notify(1, "Analyzing paper", "Reading PDF and extracting structure...")
    analysis_raw = call_llm_with_retry(
        system_prompt=SYSTEM_PROMPT,
        user_content=f"Here is the research paper content:\n\n{pdf_text}\n\nInstructions:\n{ANALYSIS_PROMPT}",
        max_tokens=MAX_TOKENS_ANALYSIS,
        model=model,
        api_key=api_key,
        on_thinking=on_thinking,
    )
    analysis = parse_llm_json(analysis_raw, "paper_analysis", model, api_key=api_key)
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
    design_prompt = DESIGN_PROMPT_TEMPLATE.format(
        analysis_json=json.dumps(analysis, indent=2)
    )
    design_raw = call_llm_with_retry(
        system_prompt=SYSTEM_PROMPT,
        user_content=f"Here is the research paper content:\n\n{pdf_text}\n\nInstructions:\n{design_prompt}",
        max_tokens=MAX_TOKENS_DESIGN,
        model=model,
        api_key=api_key,
        on_thinking=on_thinking,
    )
    design = parse_llm_json(design_raw, "toy_design", model, api_key=api_key)
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
    cells_raw = call_llm_with_retry(
        system_prompt=SYSTEM_PROMPT,
        user_content=f"Here is the research paper content:\n\n{pdf_text}\n\nInstructions:\n{generate_prompt}",
        max_tokens=MAX_TOKENS_GENERATE,
        model=model,
        api_key=api_key,
        on_thinking=on_thinking,
    )
    cells = parse_llm_json(cells_raw, "generate_cells", model, api_key=api_key)
    num_cells = len(cells)
    code_cells = sum(1 for c in cells if c.get("cell_type") == "code")
    previews = []
    for c in cells:
        previews.append({
            "type": c.get("cell_type", "code"),
            "preview": c.get("source", "")[:300],
        })

    # Build draft notebook bytes and send as draft_ready
    draft_bytes = _cells_to_bytes(cells)
    _notify(3, "Generating notebook", f"Generated {num_cells} cells ({code_cells} code)", {
        "type": "cells_generated",
        "num_cells": num_cells,
        "code_cells": code_cells,
        "previews": previews,
        "draft_bytes": draft_bytes,
    })

    # Step 4: Validate & Repair (LLM review)
    _notify(4, "Validating code", "LLM reviewing for errors...")
    validate_prompt = VALIDATE_PROMPT_TEMPLATE.format(
        cells_json=json.dumps(cells, indent=2)
    )
    validated_raw = call_llm_with_retry(
        system_prompt=SYSTEM_PROMPT,
        user_content=validate_prompt,
        max_tokens=MAX_TOKENS_VALIDATE,
        model=model,
        api_key=api_key,
        on_thinking=on_thinking,
    )
    validated_cells = parse_llm_json(validated_raw, "validate", model, api_key=api_key)
    _notify(4, "Validating code", "Validation complete")

    # Build and return validated notebook
    nb = build_notebook(validated_cells)
    return _nb_to_bytes(nb)

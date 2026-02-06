"""Web-compatible pipeline wrapper with progress callbacks and notebook execution."""
from __future__ import annotations

import io
import json
import traceback
from typing import Callable, Optional

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
from google.genai import types

from config import (
    DEFAULT_MODEL,
    EXECUTE_TIMEOUT,
    MAX_FIX_ATTEMPTS,
    MAX_TOKENS_ANALYSIS,
    MAX_TOKENS_DESIGN,
    MAX_TOKENS_FIX,
    MAX_TOKENS_GENERATE,
    MAX_TOKENS_VALIDATE,
)
from llm import call_gemini_with_retry, parse_llm_json
from notebook_builder import build_notebook
from prompts import (
    ANALYSIS_PROMPT,
    DESIGN_PROMPT_TEMPLATE,
    FIX_ERRORS_PROMPT_TEMPLATE,
    GENERATE_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    VALIDATE_PROMPT_TEMPLATE,
)

# callback(step_number, step_name, detail_message, extra_data)
# extra_data is optional dict with keys like "content", "type" for the UI
ProgressCallback = Callable[[int, str, str, Optional[dict]], None]


def _cells_to_json(cells_list: list) -> str:
    return json.dumps(cells_list, indent=2)


def _nb_to_bytes(nb: nbformat.NotebookNode) -> bytes:
    buffer = io.StringIO()
    nbformat.write(nb, buffer)
    return buffer.getvalue().encode("utf-8")


def _execute_notebook(nb: nbformat.NotebookNode) -> tuple[bool, int, str]:
    """Execute a notebook and return (success, failed_cell_index, error_traceback)."""
    ep = ExecutePreprocessor(timeout=EXECUTE_TIMEOUT, kernel_name="python3")
    try:
        ep.preprocess(nb, {"metadata": {"path": "/tmp"}})
        return True, -1, ""
    except CellExecutionError as e:
        # Find which cell failed
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == "code" and cell.get("outputs"):
                for output in cell.outputs:
                    if output.get("output_type") == "error":
                        tb = "\n".join(output.get("traceback", []))
                        return False, i, tb
        return False, -1, str(e)
    except Exception as e:
        return False, -1, str(e)


def run_web_pipeline(
    pdf_bytes: bytes,
    model: str = DEFAULT_MODEL,
    on_progress: Optional[ProgressCallback] = None,
) -> bytes:
    """Run the full pipeline on PDF bytes, returning .ipynb bytes.

    Args:
        pdf_bytes: Raw PDF file content.
        model: Gemini model ID.
        on_progress: Called with (step_num, step_name, detail_msg, extra_data).

    Returns:
        The generated .ipynb file as UTF-8 bytes.
    """

    def _notify(step: int, name: str, detail: str = "", extra: Optional[dict] = None):
        if on_progress:
            on_progress(step, name, detail, extra)

    # Build Gemini Part from raw bytes
    pdf_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")

    # ------------------------------------------------------------------
    # Step 1: Paper Analysis
    # ------------------------------------------------------------------
    _notify(1, "Analyzing paper", "Reading PDF and extracting structure...")
    analysis_raw = call_gemini_with_retry(
        system_prompt=SYSTEM_PROMPT,
        user_content=[pdf_part, ANALYSIS_PROMPT],
        max_tokens=MAX_TOKENS_ANALYSIS,
        model=model,
    )
    analysis = parse_llm_json(analysis_raw, "paper_analysis", model)
    title = analysis.get("title", "Unknown Paper")
    num_algos = len(analysis.get("algorithms", []))
    _notify(1, "Analyzing paper", f"Found: {title}", {
        "type": "analysis",
        "title": title,
        "algorithms": num_algos,
        "insight": analysis.get("key_insight", ""),
        "problem": analysis.get("problem_statement", ""),
    })

    # ------------------------------------------------------------------
    # Step 2: Design Plan
    # ------------------------------------------------------------------
    _notify(2, "Designing implementation", "Planning model architecture and training...")
    design_prompt = DESIGN_PROMPT_TEMPLATE.format(
        analysis_json=json.dumps(analysis, indent=2)
    )
    design_raw = call_gemini_with_retry(
        system_prompt=SYSTEM_PROMPT,
        user_content=[pdf_part, design_prompt],
        max_tokens=MAX_TOKENS_DESIGN,
        model=model,
    )
    design = parse_llm_json(design_raw, "toy_design", model)
    arch = design.get("model_architecture", {})
    _notify(2, "Designing implementation", "Architecture designed", {
        "type": "design",
        "notebook_title": design.get("notebook_title", ""),
        "model_type": arch.get("type", ""),
        "embed_dim": arch.get("embed_dim", ""),
        "num_layers": arch.get("num_layers", ""),
        "num_heads": arch.get("num_heads", ""),
    })

    # ------------------------------------------------------------------
    # Step 3: Generate Notebook Cells
    # ------------------------------------------------------------------
    _notify(3, "Generating notebook", "Writing PyTorch code and explanations...")
    generate_prompt = GENERATE_PROMPT_TEMPLATE.format(
        analysis_json=json.dumps(analysis, indent=2),
        design_json=json.dumps(design, indent=2),
    )
    cells_raw = call_gemini_with_retry(
        system_prompt=SYSTEM_PROMPT,
        user_content=[pdf_part, generate_prompt],
        max_tokens=MAX_TOKENS_GENERATE,
        model=model,
    )
    cells = parse_llm_json(cells_raw, "generate_cells", model)
    num_cells = len(cells)
    code_cells = sum(1 for c in cells if c.get("cell_type") == "code")
    # Send cell previews to the UI
    previews = []
    for c in cells:
        previews.append({
            "type": c.get("cell_type", "code"),
            "preview": c.get("source", "")[:300],
        })
    _notify(3, "Generating notebook", f"Generated {num_cells} cells ({code_cells} code)", {
        "type": "cells_generated",
        "num_cells": num_cells,
        "code_cells": code_cells,
        "previews": previews,
    })

    # ------------------------------------------------------------------
    # Step 4: Validate & Repair (LLM review)
    # ------------------------------------------------------------------
    _notify(4, "Validating code", "LLM reviewing for errors...")
    validate_prompt = VALIDATE_PROMPT_TEMPLATE.format(
        cells_json=json.dumps(cells, indent=2)
    )
    validated_raw = call_gemini_with_retry(
        system_prompt=SYSTEM_PROMPT,
        user_content=[validate_prompt],
        max_tokens=MAX_TOKENS_VALIDATE,
        model=model,
    )
    validated_cells = parse_llm_json(validated_raw, "validate", model)
    _notify(4, "Validating code", "LLM review complete")

    # ------------------------------------------------------------------
    # Step 5: Execute notebook & fix runtime errors
    # ------------------------------------------------------------------
    _notify(5, "Executing notebook", "Running all cells to verify...")

    current_cells = validated_cells
    for attempt in range(MAX_FIX_ATTEMPTS + 1):
        nb = build_notebook(current_cells)
        success, failed_idx, error_tb = _execute_notebook(nb)

        if success:
            _notify(5, "Executing notebook", "All cells executed successfully!", {
                "type": "execution_success",
            })
            return _nb_to_bytes(nb)

        if attempt < MAX_FIX_ATTEMPTS:
            # Report the error and attempt a fix
            failed_source = current_cells[failed_idx]["source"] if 0 <= failed_idx < len(current_cells) else "unknown"
            _notify(5, "Executing notebook",
                    f"Cell {failed_idx} failed (attempt {attempt + 1}/{MAX_FIX_ATTEMPTS}). Fixing...", {
                        "type": "execution_error",
                        "cell_index": failed_idx,
                        "error": error_tb[:500],
                    })

            fix_prompt = FIX_ERRORS_PROMPT_TEMPLATE.format(
                cell_index=failed_idx,
                cell_source=failed_source,
                error_traceback=error_tb[:3000],
                cells_json=_cells_to_json(current_cells),
            )
            fixed_raw = call_gemini_with_retry(
                system_prompt=SYSTEM_PROMPT,
                user_content=[fix_prompt],
                max_tokens=MAX_TOKENS_FIX,
                model=model,
            )
            current_cells = parse_llm_json(fixed_raw, "fix_errors", model)
        else:
            # Out of fix attempts — return notebook as-is with a warning
            _notify(5, "Executing notebook",
                    f"Could not fix all errors after {MAX_FIX_ATTEMPTS} attempts. Notebook may have issues.", {
                        "type": "execution_warning",
                        "error": error_tb[:500],
                    })
            nb = build_notebook(current_cells)
            return _nb_to_bytes(nb)

    # Fallback (should not reach here)
    nb = build_notebook(current_cells)
    return _nb_to_bytes(nb)

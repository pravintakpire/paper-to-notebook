"""Multi-step pipeline: PDF → analysis → design → cells → validated notebook."""

import json
import sys
from typing import Optional

from config import (
    MAX_TOKENS_ANALYSIS,
    MAX_TOKENS_DESIGN,
    MAX_TOKENS_GENERATE,
    MAX_TOKENS_VALIDATE,
    PROVIDER_OPENAI,
)
from llm import call_llm_with_retry, load_pdf_text, parse_llm_json
from notebook_builder import build_notebook, save_notebook
from prompts import (
    ANALYSIS_PROMPT,
    DESIGN_PROMPT_TEMPLATE,
    GENERATE_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    VALIDATE_PROMPT_TEMPLATE,
)


def _print_step(step_num: int, name: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Step {step_num}/4: {name}")
    print(f"{'='*60}")


def run_pipeline(
    pdf_path: str,
    output_path: str,
    model: str,
    provider: str = PROVIDER_OPENAI,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """Run the full paper-to-code pipeline.

    Args:
        pdf_path: Path to the research paper PDF.
        output_path: Where to save the generated .ipynb.
        model: LLM model ID to use.
        provider: LLM provider (openai, gemini, local).
        api_key: API key for the provider.
        base_url: Base URL for local provider.
        verbose: If True, print intermediate JSON outputs.
    """
    def _llm(system_prompt: str, user_content: str, max_tokens: int) -> str:
        return call_llm_with_retry(
            system_prompt=system_prompt,
            user_content=user_content,
            max_tokens=max_tokens,
            model=model,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
        )

    def _parse(raw: str, step: str) -> dict | list:
        return parse_llm_json(raw, step, model, provider=provider, api_key=api_key, base_url=base_url)

    print(f"Loading PDF: {pdf_path}")
    pdf_text = load_pdf_text(pdf_path)
    print(f"  Loaded {len(pdf_text)} characters of text.")

    # ------------------------------------------------------------------
    # Step 1: Paper Analysis
    # ------------------------------------------------------------------
    _print_step(1, "Analyzing paper")
    print("  Extracting title, algorithms, baselines, metrics...")

    analysis = _parse(
        _llm(SYSTEM_PROMPT, f"Here is the research paper content:\n\n{pdf_text}\n\nInstructions:\n{ANALYSIS_PROMPT}", MAX_TOKENS_ANALYSIS),
        "paper_analysis",
    )
    title = analysis.get("title", "Unknown Paper")
    num_algos = len(analysis.get("algorithms", []))
    print(f"  Paper: {title}")
    print(f"  Found {num_algos} algorithm(s)")

    if verbose:
        print("\n  --- Analysis JSON ---")
        print(json.dumps(analysis, indent=2))

    # ------------------------------------------------------------------
    # Step 2: Toy Design Plan
    # ------------------------------------------------------------------
    _print_step(2, "Designing toy implementation")
    print("  Planning synthetic data, mock models, experiment loop...")

    design_prompt = DESIGN_PROMPT_TEMPLATE.format(analysis_json=json.dumps(analysis, indent=2))
    design = _parse(
        _llm(SYSTEM_PROMPT, f"Here is the research paper content:\n\n{pdf_text}\n\nInstructions:\n{design_prompt}", MAX_TOKENS_DESIGN),
        "toy_design",
    )
    num_mocks = len(design.get("mock_models", []))
    num_viz = len(design.get("visualizations", []))
    print(f"  Mock components: {num_mocks}")
    print(f"  Planned visualizations: {num_viz}")

    if verbose:
        print("\n  --- Design JSON ---")
        print(json.dumps(design, indent=2))

    # ------------------------------------------------------------------
    # Step 3: Generate Notebook Cells
    # ------------------------------------------------------------------
    _print_step(3, "Generating notebook cells")
    print("  Writing code and markdown for all 12 sections...")

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
    md_cells = sum(1 for c in cells if c.get("cell_type") == "markdown")
    print(f"  Generated {num_cells} cells ({code_cells} code, {md_cells} markdown)")

    if verbose:
        print("\n  --- First 3 cells ---")
        for c in cells[:3]:
            ctype = c.get("cell_type", "?")
            src = c.get("source", "")[:200]
            print(f"  [{ctype}] {src}...")

    # ------------------------------------------------------------------
    # Step 4: Validate & Repair
    # ------------------------------------------------------------------
    _print_step(4, "Validating notebook")
    print("  Checking for undefined variables, missing imports, placeholders...")

    validate_prompt = VALIDATE_PROMPT_TEMPLATE.format(cells_json=json.dumps(cells, indent=2))
    validated_cells = _parse(
        _llm(SYSTEM_PROMPT, validate_prompt, MAX_TOKENS_VALIDATE),
        "validate",
    )
    print(f"  Validated: {len(validated_cells)} cells")

    # ------------------------------------------------------------------
    # Build and save notebook
    # ------------------------------------------------------------------
    print("\nBuilding notebook...")
    nb = build_notebook(validated_cells)
    save_notebook(nb, output_path)
    print(f"Saved to: {output_path}")

#!/usr/bin/env python3
"""
Generate an educational Jupyter notebook from a research paper PDF.

Usage:
    # OpenAI (default)
    export OPENAI_API_KEY="sk-..."
    python generate_notebook.py paper.pdf

    # Gemini
    export GEMINI_API_KEY="AIza..."
    python generate_notebook.py paper.pdf --provider gemini --model gemini-2.0-flash

    # Local (Ollama)
    python generate_notebook.py paper.pdf --provider local --model llama3 --base-url http://localhost:11434/v1
"""

import argparse
import os
import sys
from pathlib import Path

from config import (
    DEFAULT_OPENAI_MODEL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_LOCAL_MODEL,
    DEFAULT_LOCAL_BASE_URL,
    MAX_PDF_SIZE_MB,
    PROVIDER_OPENAI,
    PROVIDER_GEMINI,
    PROVIDER_LOCAL,
)
from llm import detect_provider


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a toy-implementation Jupyter notebook from a research paper PDF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python generate_notebook.py paper.pdf
  python generate_notebook.py paper.pdf --provider gemini --model gemini-2.0-flash
  python generate_notebook.py paper.pdf --provider local --base-url http://localhost:11434/v1
  python generate_notebook.py paper.pdf -o output.ipynb --verbose
        """,
    )
    parser.add_argument("pdf_path", type=str, help="Path to the research paper PDF file")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output notebook path (default: <pdf_stem>_notebook.ipynb)")
    parser.add_argument("--provider", type=str, default=None,
                        choices=[PROVIDER_OPENAI, PROVIDER_GEMINI, PROVIDER_LOCAL],
                        help="LLM provider (default: auto-detect from env)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model ID (default: provider default)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key (overrides env variable)")
    parser.add_argument("--base-url", type=str, default=DEFAULT_LOCAL_BASE_URL,
                        help=f"Base URL for local provider (default: {DEFAULT_LOCAL_BASE_URL})")
    parser.add_argument("--verbose", action="store_true",
                        help="Print intermediate pipeline outputs")
    return parser.parse_args()


def main():
    args = parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"ERROR: File not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)
    if pdf_path.suffix.lower() != ".pdf":
        print(f"ERROR: File must be a PDF: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
    if file_size_mb > MAX_PDF_SIZE_MB:
        print(f"ERROR: PDF is {file_size_mb:.1f}MB. Max is {MAX_PDF_SIZE_MB}MB.", file=sys.stderr)
        sys.exit(1)

    # Resolve provider and model
    provider = args.provider or detect_provider()
    _model_defaults = {
        PROVIDER_OPENAI: DEFAULT_OPENAI_MODEL,
        PROVIDER_GEMINI: DEFAULT_GEMINI_MODEL,
        PROVIDER_LOCAL:  DEFAULT_LOCAL_MODEL,
    }
    model = args.model or _model_defaults[provider]

    # Resolve API key
    api_key = args.api_key
    if api_key is None:
        if provider == PROVIDER_OPENAI:
            api_key = os.environ.get("OPENAI_API_KEY")
        elif provider == PROVIDER_GEMINI:
            api_key = os.environ.get("GEMINI_API_KEY")

    if provider in (PROVIDER_OPENAI, PROVIDER_GEMINI) and not api_key:
        env_var = "OPENAI_API_KEY" if provider == PROVIDER_OPENAI else "GEMINI_API_KEY"
        print(f"ERROR: {env_var} not set and --api-key not provided.", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or f"{pdf_path.stem}_notebook.ipynb"
    base_url = args.base_url if provider == PROVIDER_LOCAL else None

    print("Research Paper → Code Notebook")
    print(f"  Input:    {pdf_path}")
    print(f"  Output:   {output_path}")
    print(f"  Provider: {provider}")
    print(f"  Model:    {model}")

    from pipeline import run_pipeline

    run_pipeline(
        pdf_path=str(pdf_path),
        output_path=output_path,
        model=model,
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        verbose=args.verbose,
    )

    print(f"\nDone! Open with: jupyter notebook {output_path}")


if __name__ == "__main__":
    main()

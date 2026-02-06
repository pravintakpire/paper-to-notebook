#!/usr/bin/env python3
"""
Generate an educational Jupyter notebook from a research paper PDF.

Usage:
    export GOOGLE_API_KEY="AIza..."
    python generate_notebook.py paper.pdf
    python generate_notebook.py paper.pdf -o my_notebook.ipynb
    python generate_notebook.py paper.pdf --model gemini-2.5-pro
"""

import argparse
import os
import sys
from pathlib import Path

from config import DEFAULT_MODEL, MAX_PDF_SIZE_MB


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a toy-implementation Jupyter notebook from a research paper PDF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python generate_notebook.py paper.pdf
  python generate_notebook.py paper.pdf -o output.ipynb
  python generate_notebook.py paper.pdf --model gemini-2.5-pro --verbose
        """,
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to the research paper PDF file",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output notebook path (default: <pdf_stem>_notebook.ipynb)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Gemini model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print intermediate pipeline outputs to console",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate PDF exists
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"ERROR: File not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)
    if pdf_path.suffix.lower() != ".pdf":
        print(f"ERROR: File must be a PDF: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    # Check file size
    file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
    if file_size_mb > MAX_PDF_SIZE_MB:
        print(
            f"ERROR: PDF is {file_size_mb:.1f}MB. Max supported is {MAX_PDF_SIZE_MB}MB.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate API key
    if not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY environment variable not set.", file=sys.stderr)
        print("  export GOOGLE_API_KEY='AIza...'", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    output_path = args.output or f"{pdf_path.stem}_notebook.ipynb"

    print(f"Research Paper -> Toy Implementation Notebook")
    print(f"  Input:  {pdf_path}")
    print(f"  Output: {output_path}")
    print(f"  Model:  {args.model}")

    # Run pipeline
    from pipeline import run_pipeline

    run_pipeline(
        pdf_path=str(pdf_path),
        output_path=output_path,
        model=args.model,
        verbose=args.verbose,
    )

    print(f"\nDone! Open with: jupyter notebook {output_path}")


if __name__ == "__main__":
    main()

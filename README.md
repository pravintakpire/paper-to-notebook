# Paper to Code

**Turn Research Papers into Runnable PyTorch Code.**

`paper-to-code` is an AI-powered tool that converts academic research papers (PDFs) into educational, fully-runnable Jupyter Notebooks. It supports **OpenAI**, **Google Gemini**, and **local LLMs** (via Ollama).

Upload a PDF → get a 12-section notebook with real PyTorch implementations that run on CPU in under 15 minutes.

---

## Features

- **Multi-Provider**: OpenAI (GPT-4o), Google Gemini (gemini-2.0-flash), or local models via Ollama
- **Deep Analysis**: Extracts algorithms, architectures, equations, and training details from raw PDFs
- **Faithful Implementation**: Scaled-down toy experiments using *real* PyTorch layers — no mock-ups
- **Self-Correcting Pipeline**: AI reviews its own generated code for shape mismatches, undefined variables, and missing imports
- **Streaming UI**: Real-time progress with thinking display, draft download mid-generation, and activity cards
- **CLI + Web**: Use from the browser or the command line

---

## Architecture

```
PDF Upload
    │
    ▼
┌─────────────────────────────────────────────┐
│              4-Step Pipeline                │
│                                             │
│  1. Analyze   → Extract algorithms,         │
│                 equations, architecture     │
│                                             │
│  2. Design    → Plan scaled-down PyTorch    │
│                 implementation              │
│                                             │
│  3. Generate  → Write 12-section notebook   │
│                 with real training loops    │
│                                             │
│  4. Validate  → LLM reviews code, fixes     │
│                 errors, ensures runability  │
└─────────────────────────────────────────────┘
    │
    ▼
Generated .ipynb  ←  Download (draft + validated)
```

---

## Installation

```bash
git clone https://github.com/pravintakpire/paper-to-code.git
cd paper-to-code
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=python3
```

---

## Usage

### Web Interface

```bash
./start.sh
```

Open [http://localhost:8000](http://localhost:8000), select your provider, enter your API key, and upload a PDF.

Or start manually:

```bash
uvicorn app:app --reload --port 8000
```

### CLI

**OpenAI** (default):
```bash
export OPENAI_API_KEY="sk-..."
python generate_notebook.py paper.pdf
python generate_notebook.py paper.pdf -o output.ipynb --model gpt-4o
```

**Gemini**:
```bash
export GEMINI_API_KEY="AIza..."
python generate_notebook.py paper.pdf --provider gemini --model gemini-2.0-flash
```

**Local (Ollama)**:
```bash
# Requires Ollama running at localhost:11434
python generate_notebook.py paper.pdf --provider local --model llama3
python generate_notebook.py paper.pdf --provider local --base-url http://localhost:11434/v1
```

---

## LLM Providers

| Provider | Default Model | API Key Env Var | Notes |
|----------|--------------|-----------------|-------|
| OpenAI | `gpt-4o` | `OPENAI_API_KEY` | Best quality |
| Gemini | `gemini-2.0-flash` | `GEMINI_API_KEY` | Fast & free tier |
| Local | `llama3` | *(none)* | Requires [Ollama](https://ollama.com) |

---

## Project Structure

```
paper-to-code/
├── app.py              # FastAPI server (streaming SSE, multi-provider)
├── web_pipeline.py     # Async pipeline wrapper with progress callbacks
├── pipeline.py         # CLI pipeline (same 4 steps)
├── llm.py              # Unified LLM client (OpenAI / Gemini / Local)
├── config.py           # Provider constants, model defaults, token limits
├── prompts.py          # System + step-specific prompt templates
├── notebook_builder.py # nbformat notebook construction
├── generate_notebook.py# CLI entry point with --provider flag
├── static/index.html   # Dark-themed web UI with provider selector
├── requirements.txt    # All dependencies (openai + google-genai + torch)
├── Dockerfile          # Container build
└── start.sh / stop.sh  # Server lifecycle scripts
```

---

## Generated Notebook Sections

Each generated notebook contains 12 sections:

1. Title & Paper Overview
2. Problem Intuition
3. Imports & Setup
4. Dataset & Tokenization
5. Model Architecture
6. Loss Function & Training Utilities
7. Baseline Implementation
8. Main Algorithm — Training
9. Inference / Generation
10. Full Experiment & Evaluation
11. Visualizations
12. Summary & Next Steps

---

## Requirements

- Python 3.9+
- PyTorch (CPU is sufficient)
- One of: OpenAI API key, Gemini API key, or Ollama running locally

"""Prompt templates for each pipeline step."""

SYSTEM_PROMPT = (
    "You are an expert research engineer and educator who faithfully implements "
    "academic papers as runnable, educational Python code. You use real ML components "
    "(PyTorch, Transformer layers, actual training loops) at a reduced scale that "
    "runs on CPU. You prioritize faithful replication of the paper's architecture "
    "and algorithms while making the code deeply educational with clear explanations, "
    "verbose logging, and insightful visualizations."
)

# ---------------------------------------------------------------------------
# Step 1: Paper Analysis
# ---------------------------------------------------------------------------
ANALYSIS_PROMPT = """\
Read this research paper carefully and extract a thorough structured analysis.

Return a JSON object with EXACTLY these fields:

{
  "title": "Full paper title",
  "authors": "Author list as a single string",
  "abstract_summary": "2-3 sentence plain English summary of the paper",
  "problem_statement": "What problem does the paper solve? (2-3 sentences, no jargon)",
  "key_insight": "The core idea or innovation in one sentence",
  "algorithms": [
    {
      "name": "Algorithm name (e.g., 'GRPO', 'DPO', 'LLaDA Pre-training', etc.)",
      "description": "What this algorithm does in plain English",
      "inputs": ["list of inputs with types and shapes where applicable"],
      "outputs": ["list of outputs with types and shapes where applicable"],
      "steps": ["ordered list of DETAILED algorithmic steps — include math operations, loss functions, gradient updates"],
      "is_core": true,
      "equations": ["key equations used in this algorithm, in LaTeX or descriptive form"],
      "architecture_details": "Describe the neural network architecture used (layers, dimensions, attention type, etc.)"
    }
  ],
  "baselines": [
    {
      "name": "Baseline method name",
      "description": "Detailed description of how it works, including its loss function and training procedure"
    }
  ],
  "evaluation_metrics": ["list of metrics used to evaluate, with formulas if available"],
  "key_equations": ["ALL important equations from the paper described precisely, e.g., 'L_ELBO = E_t[1/t * CE(p_theta(x_t), x_0)]'"],
  "model_architecture": {
    "type": "Transformer/CNN/RNN/etc.",
    "key_layers": ["list of layer types used"],
    "dimensions": "hidden dim, num heads, num layers mentioned in paper",
    "special_features": "any non-standard architectural choices (e.g., no causal mask, bidirectional attention)"
  },
  "training_details": {
    "optimizer": "optimizer used",
    "learning_rate": "learning rate or schedule",
    "batch_size": "batch size",
    "epochs_or_steps": "training duration",
    "loss_function": "detailed description of the loss function"
  },
  "data_description": "What kind of data the paper uses, its format, and scale"
}

IMPORTANT:
- Be EXTREMELY detailed about algorithms — every mathematical operation matters
- Extract ALL equations, not just the main one
- Describe the model architecture precisely (layer types, dimensions, attention patterns)
- Note training hyperparameters — we will scale them down but keep the same structure
- Capture the FULL pipeline: data → model → training → inference → evaluation
- Return ONLY valid JSON, no markdown fencing, no explanation before or after
"""

# ---------------------------------------------------------------------------
# Step 2: Implementation Design Plan
# ---------------------------------------------------------------------------
DESIGN_PROMPT_TEMPLATE = """\
You are designing a FAITHFUL implementation of a research paper for educational purposes.

Here is the structured analysis of the paper:
{analysis_json}

Design a complete implementation plan. The core philosophy is:
"Replicate the paper as faithfully as possible at a smaller scale that runs on CPU."

We use REAL components:
- Real PyTorch Transformer encoder/decoder layers (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)
- Real embedding layers (nn.Embedding)
- Real training loops with actual backpropagation (optimizer.zero_grad(), loss.backward(), optimizer.step())
- Real loss functions (CrossEntropyLoss, etc.)
- Real tokenization (character-level or simple word-level — no need for BPE but must be actual tokenization)

We scale DOWN (not replace):
- Instead of 7B parameters → use a small Transformer (2-4 layers, 64-128 dim, 2-4 heads)
- Instead of millions of samples → use hundreds to thousands of synthetic samples
- Instead of 100 epochs on GPU → train for enough steps on CPU to show convergence (should complete in under 15 minutes)
- Instead of 32k vocabulary → use a small character-level or word-level vocabulary

We NEVER:
- Replace a Transformer with a lookup table or heuristic
- Replace training with memorization
- Skip the actual forward/backward pass
- Mock out the core model — it must be a real neural network
- Remove any algorithmic step from the paper

Return a JSON object with EXACTLY these fields:

{{
  "notebook_title": "A clear, descriptive title for the notebook",
  "synthetic_dataset": {{
    "description": "What the synthetic data represents and why it's a good testbed",
    "num_samples": "<int, 500-2000 for training, 100-200 for test>",
    "format": "Exact format of each sample (e.g., 'string of characters', 'token sequence', etc.)",
    "vocabulary": "Description of the vocabulary/token set",
    "generation_strategy": "How to generate structured data that tests the algorithm's capabilities",
    "code_hint": "Detailed description of the data generation logic"
  }},
  "model_architecture": {{
    "description": "Faithful scaled-down version of the paper's architecture",
    "type": "The architecture type (e.g., Transformer encoder, decoder, etc.)",
    "vocab_size": "<int>",
    "embed_dim": "<int, 64-128>",
    "num_heads": "<int, 2-4>",
    "num_layers": "<int, 2-4>",
    "max_seq_len": "<int>",
    "special_features": "Any non-standard features from the paper (e.g., no causal mask, bidirectional)",
    "pytorch_components": ["list of nn.Module components to use, e.g., 'nn.TransformerEncoderLayer', 'nn.Embedding'"]
  }},
  "training_plan": {{
    "loss_function": "Exact loss function from the paper",
    "optimizer": "Optimizer to use (e.g., AdamW)",
    "learning_rate": "<float>",
    "num_epochs": "<int>",
    "batch_size": "<int>",
    "training_steps_description": "Step-by-step description of one training iteration, faithful to the paper"
  }},
  "evaluation_function": {{
    "description": "How to evaluate the model, matching the paper's metrics",
    "metrics": ["list of metrics to compute"],
    "interface": "function signature"
  }},
  "baseline_algorithm": {{
    "name": "Baseline name from the paper",
    "description": "What it does and how it differs from the main approach",
    "implementation_approach": "How to implement it — also with real components where applicable",
    "steps": ["ordered implementation steps"]
  }},
  "main_algorithms": [
    {{
      "name": "Algorithm name from the paper",
      "description": "What it does and its key innovation",
      "steps": ["ordered implementation steps with mathematical detail"],
      "key_operations_to_log": ["specific tensor shapes, loss values, gradient norms to print"],
      "intermediate_states_to_show": ["what intermediate results to visualize"]
    }}
  ],
  "inference_procedure": {{
    "description": "How the trained model generates outputs at inference time",
    "steps": ["ordered steps for the inference/generation procedure"],
    "num_steps": "<int, if iterative>"
  }},
  "experiment_loop": {{
    "num_iterations": "<int>",
    "what_varies": "What changes across iterations/epochs",
    "what_to_measure": ["metrics to track per iteration"],
    "comparison": "How baseline vs main algorithm are compared"
  }},
  "visualizations": [
    {{
      "title": "Plot title",
      "type": "line|bar|scatter|heatmap|histogram",
      "x_axis": "What the x-axis represents",
      "y_axis": "What the y-axis represents",
      "purpose": "What insight this visualization provides"
    }}
  ]
}}

Return ONLY valid JSON, no markdown fencing.
"""

# ---------------------------------------------------------------------------
# Step 3: Generate Notebook Cells
# ---------------------------------------------------------------------------
GENERATE_PROMPT_TEMPLATE = """\
You are creating a complete educational Jupyter notebook that FAITHFULLY implements a research paper using real PyTorch components at a reduced scale.

Paper analysis:
{analysis_json}

Implementation plan:
{design_json}

Generate the COMPLETE notebook as a JSON array of cell objects. Each cell is:
{{"cell_type": "markdown" | "code", "source": "cell content as a string"}}

REQUIRED SECTIONS (in this EXACT order):

1. **Title & Paper Overview** (markdown)
   - Paper title, authors, arxiv link
   - One-paragraph summary of what the paper does
   - "What we'll implement" — explain that this is a faithful scaled-down implementation
   - Architecture diagram in text/ASCII if helpful

2. **Problem Intuition** (markdown)
   - Plain English explanation of the problem
   - Why it matters
   - Key insight of the paper explained with an analogy or example
   - How this paper differs from prior work

3. **Imports & Setup** (code)
   - import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
   - import numpy as np, matplotlib.pyplot as plt
   - Set seeds: torch.manual_seed(42), np.random.seed(42)
   - Set device = torch.device('cpu')
   - Print PyTorch version

4. **Dataset & Tokenization** (code + markdown)
   - Generate synthetic data that meaningfully tests the algorithm
   - Implement actual tokenization (character-level or word-level)
   - Create proper PyTorch Dataset and DataLoader
   - Print dataset statistics, vocabulary size, sample sequences
   - Show tokenized examples

5. **Model Architecture** (code + markdown)
   - Implement the paper's model using REAL PyTorch layers:
     * nn.Embedding for token embeddings
     * nn.TransformerEncoderLayer / nn.TransformerDecoderLayer for the core architecture
     * Positional encoding (sinusoidal or learned)
     * Proper output projection head
   - The model class should mirror the paper's architecture as closely as possible
   - Print model summary: parameter count, layer structure
   - Explain each component and why it's there, referencing the paper

6. **Loss Function & Training Utilities** (code + markdown)
   - Implement the paper's EXACT loss function
   - Implement any special training procedures (e.g., masking strategies, noise schedules)
   - Demonstrate the loss computation on a single batch
   - Print shapes at each step to make the data flow clear

7. **Baseline Implementation** (code + markdown)
   - Implement the baseline approach from the paper
   - This should also use real components where applicable
   - Train it on the same data
   - Log training loss, show convergence

8. **Paper's Main Algorithm — Training** (code + markdown)
   - Implement the full training loop faithful to the paper
   - Print per-epoch: loss, learning rate, gradient norm, sample predictions
   - Show the model improving over time
   - Highlight what makes this different from the baseline
   - Include the ACTUAL algorithm steps from the paper as comments in the code

9. **Inference / Generation** (code + markdown)
   - Implement the paper's inference procedure exactly
   - If iterative (e.g., diffusion), show each step
   - Print intermediate states at every step
   - Generate multiple examples and display them
   - Compare outputs before and after training

10. **Full Experiment & Evaluation** (code + markdown)
    - Run systematic evaluation on held-out test data
    - Compute the paper's metrics
    - Compare baseline vs main algorithm quantitatively
    - Print a results table
    - Run statistical comparisons if applicable

11. **Visualizations** (code + markdown)
    - Training loss curves (baseline vs main)
    - At least 4 matplotlib plots total
    - Attention weight visualization if applicable
    - Generation quality over training
    - Distribution plots, heatmaps for iterative processes
    - Use clear labels, titles, legends, and professional styling

12. **Summary & Next Steps** (markdown)
    - What we observed (reference specific numbers)
    - How our results qualitatively match the paper's claims
    - What would change at full scale
    - Concrete ideas for extending this implementation

CRITICAL RULES:
- USE REAL PYTORCH: nn.TransformerEncoderLayer, nn.Embedding, actual backpropagation
- Every code cell MUST be complete and runnable — NO placeholders (TODO, pass, ..., NotImplementedError)
- Variables must be defined before use (cells run top-to-bottom)
- Algorithm functions MUST print intermediate states: tensor shapes, loss values, sample outputs
- torch.manual_seed(42) and np.random.seed(42) for reproducibility
- Total notebook runtime must be under 15 minutes on CPU
- Allowed libraries: torch, numpy, matplotlib, collections, math, random, itertools, functools
- DO NOT use: tensorflow, sklearn, transformers (huggingface), scipy, pandas
- Add type hints to all function signatures
- Add docstrings explaining what each function does and how it relates to the paper
- Reference specific sections/equations from the paper in comments and markdown
- Each code cell should be focused — separate model definition, training, inference, evaluation
- Include markdown cells between code cells explaining the "why" behind each step
- The code should be EDUCATIONAL: someone reading it should understand the paper better than from reading the paper alone

Return ONLY a valid JSON array of cell objects. No markdown fencing around the JSON.
"""

# ---------------------------------------------------------------------------
# Step 4: Validate & Repair
# ---------------------------------------------------------------------------
VALIDATE_PROMPT_TEMPLATE = """\
You are a meticulous code reviewer checking a Jupyter notebook that implements a research paper using PyTorch.

Here is the notebook as a JSON array of cells:
{cells_json}

Check for ALL of the following issues and fix any you find:

1. **Undefined variables**: Any variable used before it's defined (across cells, top-to-bottom order)
2. **Missing imports**: Any module or function used but not imported (especially torch submodules)
3. **Placeholder code**: Any TODO, pass, "...", NotImplementedError, or incomplete implementations
4. **Syntax errors**: Any Python syntax issues
5. **Tensor shape mismatches**: Check that tensor operations have compatible shapes
6. **Missing sections**: All 12 required sections must be present:
   - Title & Paper Overview, Problem Intuition, Imports & Setup, Dataset & Tokenization,
   - Model Architecture, Loss Function & Training Utilities, Baseline Implementation,
   - Main Algorithm Training, Inference/Generation, Full Experiment & Evaluation,
   - Visualizations, Summary & Next Steps
7. **Real PyTorch usage**: The notebook MUST use real nn.TransformerEncoderLayer or nn.TransformerDecoderLayer (not mock/fake models)
8. **Real training**: There must be actual optimizer.zero_grad(), loss.backward(), optimizer.step() calls
9. **Missing observability**: Training loops must print loss, model functions must log tensor shapes
10. **Missing seeds**: torch.manual_seed(42) and np.random.seed(42) must be set
11. **Runtime concerns**: Training loops should complete within ~15 minutes on CPU
12. **Forbidden imports**: No tensorflow, sklearn, transformers (huggingface), scipy, pandas

For each issue found, fix it directly in the cells.

If the notebook is already correct, return it unchanged.

Return ONLY the complete JSON array of cells (whether modified or not). No explanation, no markdown fencing.
"""

# ---------------------------------------------------------------------------
# Step 5: Fix Runtime Errors
# ---------------------------------------------------------------------------
FIX_ERRORS_PROMPT_TEMPLATE = """\
You are debugging a Jupyter notebook that implements a research paper using PyTorch.

The notebook was executed and the following cell produced a runtime error.

**Cell number (0-indexed):** {cell_index}
**Cell source code:**
```python
{cell_source}
```

**Error traceback:**
```
{error_traceback}
```

**Full notebook (as JSON array of cells):**
{cells_json}

Fix the error. Common issues include:
- Tensor shape mismatches (reshape, view, matmul dimension errors)
- Wrong variable names or typos
- Missing function arguments
- Incorrect indexing
- Type errors (e.g., passing int where float expected)
- Device mismatches (should all be CPU)
- Incorrect use of PyTorch APIs

IMPORTANT:
- Fix ONLY the broken cell(s) — do not rewrite the entire notebook
- The fix may require changes in earlier cells if the root cause is upstream
- Make sure the fix is consistent with the rest of the notebook
- Do NOT add placeholder code — the fix must be complete

Return the COMPLETE JSON array of ALL cells (the full notebook with fixes applied). No explanation, no markdown fencing.
"""

"""Configuration constants for the paper-to-notebook tool."""

# Default Gemini model
DEFAULT_MODEL = "gemini-2.5-pro"

# Token limits per pipeline step (larger for real PyTorch implementations)
MAX_TOKENS_ANALYSIS = 8192
MAX_TOKENS_DESIGN = 8192
MAX_TOKENS_GENERATE = 65536
MAX_TOKENS_VALIDATE = 65536
MAX_TOKENS_FIX = 65536

# Notebook execution
EXECUTE_TIMEOUT = 300  # seconds per cell
MAX_FIX_ATTEMPTS = 2   # max times to retry fixing errors

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAYS = [5, 15, 30]  # seconds

# PDF constraints
MAX_PDF_SIZE_MB = 30
MAX_PDF_PAGES = 100

# Required notebook sections (in order)
REQUIRED_SECTIONS = [
    "Title & Paper Overview",
    "Problem Intuition",
    "Imports & Setup",
    "Dataset & Tokenization",
    "Model Architecture",
    "Loss Function & Training Utilities",
    "Baseline Implementation",
    "Paper's Main Algorithm — Training",
    "Inference / Generation",
    "Full Experiment & Evaluation",
    "Visualizations",
    "Summary & Next Steps",
]

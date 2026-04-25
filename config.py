"""Configuration constants for the paper-to-code tool."""

# LLM Providers
PROVIDER_OPENAI = "openai"
PROVIDER_GEMINI = "gemini"
PROVIDER_LOCAL  = "local"

# Default models per provider
DEFAULT_OPENAI_MODEL   = "gpt-4o"
DEFAULT_GEMINI_MODEL   = "gemini-2.0-flash"
DEFAULT_LOCAL_MODEL    = "llama3"
DEFAULT_LOCAL_BASE_URL = "http://localhost:11434/v1"

# Fallback default (used by CLI when no provider flag given)
DEFAULT_MODEL = DEFAULT_OPENAI_MODEL

# Token limits per pipeline step
MAX_TOKENS_ANALYSIS = 8192
MAX_TOKENS_DESIGN   = 8192
MAX_TOKENS_GENERATE = 16384
MAX_TOKENS_VALIDATE = 16384
MAX_TOKENS_FIX      = 16384

# Notebook execution
EXECUTE_TIMEOUT  = 300  # seconds per cell
MAX_FIX_ATTEMPTS = 2

# Retry configuration
MAX_RETRIES  = 3
RETRY_DELAYS = [5, 15, 30]  # seconds

# PDF constraints
MAX_PDF_SIZE_MB = 30
MAX_PDF_PAGES   = 100

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

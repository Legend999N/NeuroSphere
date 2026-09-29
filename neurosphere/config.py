from pathlib import Path


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Local storage
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "index"

# Semantic model
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# Retrieval
DEFAULT_TOP_K = 10
RRF_K = 60

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# Supported files
SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".py",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".js",
    ".ts",
    ".html",
    ".css",
    ".json",
    ".xml",
    ".sql",
    ".pdf",
    ".docx",
}


def ensure_directories() -> None:
    """Create required project directories."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
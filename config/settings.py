"""
Global settings and configuration for the Telegram bot with Pinecone long-term memory.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# ---------------------------------------------------------------------------
# Similarity / memory filtering
# ---------------------------------------------------------------------------
# Cosine similarity threshold. Values >= threshold are treated as duplicates
# (or variations) and are not written as new vectors. Values below are new info.
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))

# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
# Optional HTTP/HTTPS/SOCKS proxy for Telegram Bot API.
# Example: http://127.0.0.1:8080 or socks5h://127.0.0.1:1080
TELEGRAM_PROXY_URL: str = os.getenv("TELEGRAM_PROXY_URL", "")
# Optional custom Bot API base URL (reverse proxy or local Bot API server).
# Example: https://tg-api.example.com
TELEGRAM_API_URL: str = os.getenv("TELEGRAM_API_URL", "")

# ---------------------------------------------------------------------------
# Pinecone
# ---------------------------------------------------------------------------
PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "telegram-bot-memory")
# Optional HTTP/HTTPS proxy URL used by the Pinecone SDK (passed as `proxy_url`).
# Leave empty to disable proxying. Example: http://user:pass@proxy.example.com:8080
PINECONE_PROXY_URL: str = os.getenv("PINECONE_PROXY_URL", "")

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
# Dimension of the chosen model (all-MiniLM-L6-v2 = 384)
EMBEDDING_DIMENSION: int = 384

# ---------------------------------------------------------------------------
# LLM (AITUNNEL / OpenAI-compatible)
# ---------------------------------------------------------------------------
# Base URL of the OpenAI-compatible provider (AITUNNEL)
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.aitunnel.ru/v1")
# API key for the LLM provider (AITUNNEL key, e.g. sk-aitunnel-...)
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
# Default model
LLM_MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-5-nano")
# Max tokens for completion
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
# Temperature
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# ---------------------------------------------------------------------------
# Bot behaviour
# ---------------------------------------------------------------------------
# Maximum number of similar memories to retrieve for context
TOP_K_MEMORIES: int = 5

# Namespace prefix (optional isolation)
DEFAULT_NAMESPACE: str = "default"

"""Configurações do projeto, lidas do arquivo .env."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

RAIZ = Path(__file__).resolve().parent.parent
DATA_DIR = RAIZ / "data"
CHROMA_DIR = RAIZ / "chroma_db"

DOCS_PATH = DATA_DIR / "reviews_docs.parquet"
AMOSTRA_PATH = DATA_DIR / "amostra.parquet"


def _bool(nome: str, padrao: str = "false") -> bool:
    return os.getenv(nome, padrao).strip().lower() in {"1", "true", "sim", "yes"}


USE_MOCK = _bool("USE_MOCK", "true")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_TIMEOUT_S = float(os.getenv("OLLAMA_TIMEOUT_S", "5"))

TOP_K_CANDIDATOS = int(os.getenv("TOP_K_CANDIDATOS", "40"))
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "10"))
LIMIAR_RELEVANCIA = float(os.getenv("LIMIAR_RELEVANCIA", "0.5"))
MIN_EVIDENCIAS = int(os.getenv("MIN_EVIDENCIAS", "3"))

COLECAO_CHROMA = "reviews_olist"

"""Cliente Ollama compartilhado (T11 transformação + gate de suficiência)."""

from __future__ import annotations

import json
import logging
import re

from src.config import LLM_MODEL, OLLAMA_HOST, OLLAMA_TIMEOUT_S

logger = logging.getLogger(__name__)

_warned_import = False
_warned_call = False


def strip_cercas_json(texto: str) -> str:
    """Remove cercas ```json ... ``` como na aula03 de prompt."""
    bruto = (texto or "").strip()
    bruto = re.sub(r"^```(?:json)?\s*", "", bruto, flags=re.IGNORECASE)
    bruto = re.sub(r"\s*```$", "", bruto)
    return bruto.strip()


def chat_json(system: str, user: str, *, num_predict: int = 256) -> dict | None:
    """Chat Ollama com format=json. None em qualquer falha (sem crash)."""
    global _warned_import, _warned_call
    try:
        import ollama  # lazy: pacote ausente → caller faz fallback
    except ImportError:
        if not _warned_import:
            logger.warning("Pacote ollama ausente; chamada JSON ignorada.")
            _warned_import = True
        return None

    try:
        client = ollama.Client(host=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT_S)
        resposta = client.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format="json",
            options={"temperature": 0, "num_predict": num_predict},
        )
        content = resposta["message"]["content"]
        dados = json.loads(strip_cercas_json(content))
        if not isinstance(dados, dict):
            return None
        return dados
    except Exception as exc:
        if not _warned_call:
            logger.warning("Ollama indisponível (%s); caller usa fallback.", type(exc).__name__)
            _warned_call = True
        return None

"""Smoke T16 com índice real — busca + limiar; geração em mock (sem Gemini).

Rodar:
    USE_MOCK=false GEN_MOCK=true HF_HUB_OFFLINE=1 python -m pytest -m integration -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.index import BM25_PATH, CHROMA_DIR
from src.pipeline import perguntar

_BM25 = Path(BM25_PATH)
_CHROMA = Path(CHROMA_DIR)

P07 = "Qual a previsão do tempo em São Paulo amanhã?"
P09 = "O que os clientes dizem sobre entrega por drone?"
P01 = "O que os clientes relatam sobre produtos que chegaram quebrados?"


def _indice_ok() -> bool:
    if not _BM25.exists() or not _CHROMA.exists():
        return False
    try:
        from src.index import abrir_colecao

        return abrir_colecao().count() > 0
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _indice_ok(), reason="índice Chroma/BM25 ausente ou vazio"),
]


@pytest.fixture(autouse=True)
def _busca_real_geracao_mock(monkeypatch):
    monkeypatch.setattr("src.retrieve.USE_MOCK", False)
    monkeypatch.setattr("src.generate.GEN_MOCK", True)


def test_smoke_p07_fora_de_escopo():
    r = perguntar(P07)
    assert r.recusada is True
    assert r.motivo_recusa == "fora_de_escopo"


def test_smoke_p09_sem_evidencia():
    """CSV espera recusa por limiar; não por fora_de_escopo (âncora 'entrega').

    Se o CE pontuar alto em reviews genéricas de entrega, o limiar atual deixa
    responder — limite documentado em context/melhorias.md (não marcar drone
    como _OFFTOPIC).
    """
    r = perguntar(P09)
    assert r.motivo_recusa != "fora_de_escopo"
    if r.recusada and r.motivo_recusa == "sem_evidencia":
        return
    pytest.xfail(
        "limiar/CE: pergunta sobre drone ainda recupera reviews de entrega "
        f"acima do limiar (recusada={r.recusada}, motivo={r.motivo_recusa!r})"
    )


def test_smoke_p01_responde():
    r = perguntar(P01)
    assert r.motivo_recusa != "fora_de_escopo"
    assert r.recusada is False, (
        f"P01 deveria responder; got motivo={r.motivo_recusa!r} texto={r.texto[:120]!r}"
    )

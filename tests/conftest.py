"""Fixtures compartilhadas e trava anti-Ollama (T11 determinístico)."""

from __future__ import annotations

import pytest

from src.schema import ConsultaProcessada, Evidencia


@pytest.fixture(autouse=True)
def _forcar_t11_regras(monkeypatch):
    """Evita flaky/lento se .env tiver LLM_PROVIDER=ollama."""
    monkeypatch.setattr("src.retrieve.LLM_PROVIDER", "regras")


@pytest.fixture
def evidencia():
    def _fazer(
        review_id: str = "rev1",
        texto: str = "produto chegou atrasado",
        score: float = 0.8,
        **kwargs,
    ) -> Evidencia:
        return Evidencia(review_id=review_id, texto=texto, score=score, **kwargs)

    return _fazer


@pytest.fixture
def consulta():
    def _fazer(
        pergunta: str = "atraso na entrega",
        tipo: str = "pontual",
        filtros: dict | None = None,
        consulta_txt: str | None = None,
    ) -> ConsultaProcessada:
        return ConsultaProcessada(
            pergunta_original=pergunta,
            consulta=consulta_txt if consulta_txt is not None else pergunta,
            filtros=filtros or {},
            tipo=tipo,
        )

    return _fazer

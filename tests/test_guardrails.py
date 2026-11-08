"""T16 — três recusas + ok + filtrar_relevantes (sem índice)."""

from __future__ import annotations

import pytest

from src.guardrails import (
    MSG_FORA_DE_ESCOPO,
    MSG_INSUFICIENTE,
    MSG_SEM_EVIDENCIA,
    avaliar_evidencias,
    filtrar_relevantes,
)
from src.schema import ConsultaProcessada, Evidencia


@pytest.fixture(autouse=True)
def _limiar_fix(monkeypatch):
    monkeypatch.setattr("src.guardrails.LIMIAR_RELEVANCIA", 0.5)
    monkeypatch.setattr("src.guardrails.MIN_EVIDENCIAS", 3)


def _c(tipo: str = "pontual") -> ConsultaProcessada:
    return ConsultaProcessada(pergunta_original="q", consulta="q", tipo=tipo)


def _e(score: float, rid: str = "r") -> Evidencia:
    return Evidencia(review_id=rid, texto="t", score=score)


def test_fora_de_escopo():
    v = avaliar_evidencias(_c("fora_de_escopo"), [_e(0.99)])
    assert v.pode_responder is False
    assert v.motivo == "fora_de_escopo"
    assert v.mensagem == MSG_FORA_DE_ESCOPO


def test_sem_evidencia():
    v = avaliar_evidencias(_c(), [_e(0.1), _e(0.49)])
    assert v.pode_responder is False
    assert v.motivo == "sem_evidencia"
    assert v.mensagem == MSG_SEM_EVIDENCIA


def test_evidencia_insuficiente():
    v = avaliar_evidencias(_c(), [_e(0.6), _e(0.2)])
    assert v.pode_responder is True
    assert v.motivo == "evidencia_insuficiente"
    assert v.mensagem == MSG_INSUFICIENTE


def test_ok():
    v = avaliar_evidencias(_c(), [_e(0.6, "a"), _e(0.7, "b"), _e(0.8, "c")])
    assert v.pode_responder is True
    assert v.motivo == "ok"
    assert v.mensagem == ""


def test_filtrar_relevantes():
    evs = [_e(0.4, "baixo"), _e(0.5, "limiar"), _e(0.9, "alto")]
    ok = filtrar_relevantes(evs)
    assert [e.review_id for e in ok] == ["limiar", "alto"]

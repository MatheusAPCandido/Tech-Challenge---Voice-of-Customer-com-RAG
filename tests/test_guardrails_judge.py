"""Gate de suficiência Ollama (JUDGE_ENABLED) — só mocks, sem daemon."""

from __future__ import annotations

import pytest

from src.guardrails import (
    MSG_INSUFICIENTE,
    MSG_SEM_EVIDENCIA,
    avaliar_evidencias,
)
from src.schema import ConsultaProcessada, Evidencia


@pytest.fixture(autouse=True)
def _limiar_fix(monkeypatch):
    monkeypatch.setattr("src.guardrails.LIMIAR_RELEVANCIA", 0.5)
    monkeypatch.setattr("src.guardrails.MIN_EVIDENCIAS", 3)


def _c(tipo: str = "pontual", pergunta: str = "entrega por drone") -> ConsultaProcessada:
    return ConsultaProcessada(pergunta_original=pergunta, consulta=pergunta, tipo=tipo)


def _e(score: float, rid: str = "r", texto: str = "atraso na entrega") -> Evidencia:
    return Evidencia(review_id=rid, texto=texto, score=score)


def _altas() -> list[Evidencia]:
    return [_e(0.8, "a"), _e(0.7, "b"), _e(0.9, "c")]


def test_judge_off_path_igual_t16(monkeypatch):
    monkeypatch.setattr("src.guardrails.JUDGE_ENABLED", False)
    chamado = {"n": 0}

    def _boom(*_a, **_k):
        chamado["n"] += 1
        raise AssertionError("chat_json não deve ser chamado com judge off")

    monkeypatch.setattr("src.guardrails.chat_json", _boom)
    v = avaliar_evidencias(_c(), _altas())
    assert v.motivo == "ok"
    assert chamado["n"] == 0


def test_judge_on_suficiente_false(monkeypatch):
    monkeypatch.setattr("src.guardrails.JUDGE_ENABLED", True)
    monkeypatch.setattr(
        "src.guardrails.chat_json",
        lambda *_a, **_k: {"suficiente": False, "motivo": "sem drone"},
    )
    v = avaliar_evidencias(_c(), _altas())
    assert v.pode_responder is False
    assert v.motivo == "sem_evidencia"
    assert v.mensagem == MSG_SEM_EVIDENCIA


def test_judge_on_suficiente_true_ok(monkeypatch):
    monkeypatch.setattr("src.guardrails.JUDGE_ENABLED", True)
    monkeypatch.setattr(
        "src.guardrails.chat_json",
        lambda *_a, **_k: {"suficiente": True, "motivo": "ok"},
    )
    v = avaliar_evidencias(_c(), _altas())
    assert v.pode_responder is True
    assert v.motivo == "ok"


def test_judge_on_suficiente_true_insuficiente(monkeypatch):
    monkeypatch.setattr("src.guardrails.JUDGE_ENABLED", True)
    monkeypatch.setattr(
        "src.guardrails.chat_json",
        lambda *_a, **_k: {"suficiente": "sim"},
    )
    v = avaliar_evidencias(_c(), [_e(0.9)])
    assert v.pode_responder is True
    assert v.motivo == "evidencia_insuficiente"
    assert v.mensagem == MSG_INSUFICIENTE


def test_judge_on_falha_fallback_limiar(monkeypatch):
    monkeypatch.setattr("src.guardrails.JUDGE_ENABLED", True)
    monkeypatch.setattr("src.guardrails.chat_json", lambda *_a, **_k: None)
    v = avaliar_evidencias(_c(), _altas())
    assert v.pode_responder is True
    assert v.motivo == "ok"


def test_judge_on_fora_de_escopo_nao_chama(monkeypatch):
    monkeypatch.setattr("src.guardrails.JUDGE_ENABLED", True)
    chamado = {"n": 0}

    def _spy(*_a, **_k):
        chamado["n"] += 1
        return {"suficiente": False}

    monkeypatch.setattr("src.guardrails.chat_json", _spy)
    v = avaliar_evidencias(_c("fora_de_escopo"), _altas())
    assert v.motivo == "fora_de_escopo"
    assert chamado["n"] == 0


def test_judge_on_sem_relevantes_nao_chama(monkeypatch):
    monkeypatch.setattr("src.guardrails.JUDGE_ENABLED", True)
    chamado = {"n": 0}

    def _spy(*_a, **_k):
        chamado["n"] += 1
        return {"suficiente": True}

    monkeypatch.setattr("src.guardrails.chat_json", _spy)
    v = avaliar_evidencias(_c(), [_e(0.1), _e(0.2)])
    assert v.motivo == "sem_evidencia"
    assert chamado["n"] == 0

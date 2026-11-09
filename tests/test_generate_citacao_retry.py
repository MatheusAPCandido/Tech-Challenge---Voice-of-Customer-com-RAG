"""T21 — retry de citação [R#] quando a 1ª geração omite (mock, sem rede)."""

from __future__ import annotations

import pytest

import src.generate as g
from src.retrieve import _MOCK

IDS = [e.review_id for e in _MOCK]


@pytest.fixture
def sem_mock(monkeypatch):
    monkeypatch.setattr(g, "GEN_MOCK", False)
    monkeypatch.setattr(g, "MIN_EVIDENCIAS", 3)


def test_retry_quando_primeira_sem_citacao(sem_mock, monkeypatch):
    chamadas = {"n": 0}

    def _fake(_sistema, _usuario):
        chamadas["n"] += 1
        if chamadas["n"] == 1:
            return g.SaidaLLM(texto="Clientes relatam atraso na entrega.", pode_responder=True)
        return g.SaidaLLM(texto="Atraso na entrega [R1][R2].", pode_responder=True)

    monkeypatch.setattr(g, "chamar_llm", _fake)
    r = g.responder("atrasos?", _MOCK)
    assert chamadas["n"] == 2
    assert r.recusada is False
    assert r.citacoes == [IDS[0], IDS[1]]


def test_retry_falha_ainda_sem_citacao(sem_mock, monkeypatch):
    chamadas = {"n": 0}

    def _fake(_sistema, _usuario):
        chamadas["n"] += 1
        return g.SaidaLLM(texto="Resposta sem citação válida.", pode_responder=True)

    monkeypatch.setattr(g, "chamar_llm", _fake)
    r = g.responder("atrasos?", _MOCK)
    assert chamadas["n"] == 2
    assert r.recusada is True
    assert r.motivo_recusa == "sem_evidencia"
    assert r.texto == g.FRASE_RECUSA


def test_modelo_recusou_nao_faz_retry(sem_mock, monkeypatch):
    chamadas = {"n": 0}

    def _fake(_sistema, _usuario):
        chamadas["n"] += 1
        return g.SaidaLLM(texto="qualquer", pode_responder=False)

    monkeypatch.setattr(g, "chamar_llm", _fake)
    r = g.responder("telefone?", _MOCK)
    assert chamadas["n"] == 1
    assert r.recusada is True
    assert r.motivo_recusa == "sem_evidencia"


def test_primeira_ja_com_citacao_nao_retry(sem_mock, monkeypatch):
    chamadas = {"n": 0}

    def _fake(_sistema, _usuario):
        chamadas["n"] += 1
        return g.SaidaLLM(texto="Atraso [R1].", pode_responder=True)

    monkeypatch.setattr(g, "chamar_llm", _fake)
    r = g.responder("atrasos?", _MOCK)
    assert chamadas["n"] == 1
    assert r.recusada is False
    assert r.citacoes == [IDS[0]]

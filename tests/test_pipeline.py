"""T16 leve — perguntar() com mocks (sem índice/LLM)."""

from __future__ import annotations

import pytest

from src.pipeline import perguntar
from src.schema import ConsultaProcessada, Evidencia, Resposta


@pytest.fixture(autouse=True)
def _limiar_fix(monkeypatch):
    monkeypatch.setattr("src.guardrails.LIMIAR_RELEVANCIA", 0.5)
    monkeypatch.setattr("src.guardrails.MIN_EVIDENCIAS", 3)


def _consulta(tipo: str = "pontual", filtros: dict | None = None) -> ConsultaProcessada:
    return ConsultaProcessada(
        pergunta_original="q",
        consulta="q",
        tipo=tipo,
        filtros=filtros or {},
    )


def _ev(score: float, rid: str) -> Evidencia:
    return Evidencia(review_id=rid, texto=f"texto {rid}", score=score)


def test_fora_de_escopo_nao_chama_buscar(monkeypatch):
    monkeypatch.setattr(
        "src.pipeline.processar_pergunta",
        lambda q: _consulta("fora_de_escopo"),
    )
    chamado = {"buscar": False}

    def _buscar(*_a, **_k):
        chamado["buscar"] = True
        return []

    monkeypatch.setattr("src.pipeline.buscar", _buscar)
    r = perguntar("previsão do tempo")
    assert chamado["buscar"] is False
    assert r.recusada is True
    assert r.motivo_recusa == "fora_de_escopo"


def test_sem_evidencia_recusa(monkeypatch):
    monkeypatch.setattr("src.pipeline.processar_pergunta", lambda q: _consulta())
    monkeypatch.setattr(
        "src.pipeline.buscar",
        lambda *a, **k: [_ev(0.1, "a"), _ev(0.2, "b")],
    )
    r = perguntar("drone")
    assert r.recusada is True
    assert r.motivo_recusa == "sem_evidencia"


def test_evidencia_insuficiente_prefixa_aviso(monkeypatch):
    monkeypatch.setattr("src.pipeline.processar_pergunta", lambda q: _consulta())
    monkeypatch.setattr(
        "src.pipeline.buscar",
        lambda *a, **k: [_ev(0.9, "a"), _ev(0.1, "b")],
    )

    def _responder(pergunta, evidencias):
        assert len(evidencias) == 1
        assert evidencias[0].review_id == "a"
        return Resposta(texto="resposta parcial mock", citacoes=["a"], evidencias=evidencias)

    monkeypatch.setattr("src.pipeline.responder", _responder)
    r = perguntar("tema raro")
    assert r.recusada is False
    assert r.confianca == "baixa"
    assert r.texto.startswith("Encontrei poucas avaliações")
    assert "resposta parcial mock" in r.texto


def test_ok_filtra_antes_de_responder(monkeypatch):
    monkeypatch.setattr("src.pipeline.processar_pergunta", lambda q: _consulta())
    monkeypatch.setattr(
        "src.pipeline.buscar",
        lambda *a, **k: [_ev(0.9, "a"), _ev(0.8, "b"), _ev(0.7, "c"), _ev(0.1, "lixo")],
    )

    def _responder(pergunta, evidencias):
        assert [e.review_id for e in evidencias] == ["a", "b", "c"]
        return Resposta(texto="ok mock", citacoes=["a", "b", "c"], evidencias=evidencias)

    monkeypatch.setattr("src.pipeline.responder", _responder)
    r = perguntar("entrega")
    assert r.recusada is False
    assert r.texto == "ok mock"
    assert r.confianca == "media"


def test_pipeline_judge_recusa_sem_evidencia(monkeypatch):
    """P09-like: limiar passa, judge diz insuficiente → sem_evidencia."""
    monkeypatch.setattr("src.guardrails.JUDGE_ENABLED", True)
    monkeypatch.setattr(
        "src.guardrails.chat_json",
        lambda *_a, **_k: {"suficiente": False, "motivo": "sem drone"},
    )
    monkeypatch.setattr("src.pipeline.processar_pergunta", lambda q: _consulta())
    monkeypatch.setattr(
        "src.pipeline.buscar",
        lambda *a, **k: [_ev(0.9, "a"), _ev(0.8, "b"), _ev(0.7, "c")],
    )

    def _nao_gerar(*_a, **_k):
        raise AssertionError("responder não deve rodar após recusa do judge")

    monkeypatch.setattr("src.pipeline.responder", _nao_gerar)
    r = perguntar("entrega por drone")
    assert r.recusada is True
    assert r.motivo_recusa == "sem_evidencia"

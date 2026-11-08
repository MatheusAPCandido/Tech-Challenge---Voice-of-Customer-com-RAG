"""T22 — validação de citações e recusa decidida pelo código (LLM falso, sem rede)."""

from __future__ import annotations

import pytest

import src.generate as g
from src.retrieve import _MOCK

IDS = [e.review_id for e in _MOCK]  # R1 = IDS[0], R2 = IDS[1], ...


@pytest.fixture
def com_llm_falso(monkeypatch):
    """Troca chamar_llm por uma resposta fixa e roda responder() nas evidências do mock."""
    monkeypatch.setattr(g, "GEN_MOCK", False)
    monkeypatch.setattr(g, "MIN_EVIDENCIAS", 3)

    def _rodar(texto: str, pode_responder: bool = True):
        saida = g.SaidaLLM(texto=texto, pode_responder=pode_responder)
        monkeypatch.setattr(g, "chamar_llm", lambda s, u: saida)
        return g.responder("pergunta de teste", _MOCK)

    return _rodar


def _assert_recusa(r):
    assert r.recusada is True
    assert r.texto == g.FRASE_RECUSA
    assert r.motivo_recusa == "sem_evidencia"
    assert r.confianca == "baixa"


def test_a_citacao_invalida_removida(com_llm_falso):
    r = com_llm_falso("Atrasos relatados [R1][R9].")
    assert "[R9]" not in r.texto and "[R1]" in r.texto
    assert r.citacoes == [IDS[0]]
    assert r.recusada is False
    assert r.confianca == "media"


def test_b_modelo_recusou(com_llm_falso):
    _assert_recusa(com_llm_falso("Qualquer texto.", pode_responder=False))


def test_c_texto_e_a_frase_de_recusa(com_llm_falso):
    _assert_recusa(com_llm_falso(g.FRASE_RECUSA))


def test_d_sem_citacao_valida(com_llm_falso):
    _assert_recusa(com_llm_falso("Os clientes reclamam de atrasos [R9]."))


def test_e_tres_citacoes_confianca_alta(com_llm_falso):
    r = com_llm_falso("Atrasos [R1][R2][R3].")
    assert r.citacoes == IDS[:3]
    assert r.confianca == "alta"


def test_f_parcial_nao_vira_recusa(com_llm_falso):
    r = com_llm_falso(
        "Atrasos [R1][R3]. Não há informações sobre o preço dos produtos nas avaliações recuperadas."
    )
    assert r.recusada is False
    assert r.citacoes == [IDS[0], IDS[2]]
    assert "preço" in r.texto

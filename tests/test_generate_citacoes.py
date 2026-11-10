"""Contexto e citações (T19/T22) — sem LLM."""

from src.generate import montar_contexto, remover_citacoes_invalidas, validar_citacoes
from src.schema import Evidencia


def _evs():
    return [
        Evidencia(review_id="id-a", texto="atraso <grave>", score=0.9, review_score=1),
        Evidencia(review_id="id-b", texto="ok", score=0.8, review_score=5),
    ]


def test_montar_contexto_mapa_e_escape():
    ctx, mapa = montar_contexto(_evs())
    assert mapa == {"R1": "id-a", "R2": "id-b"}
    assert "<avaliacoes>" in ctx
    assert "‹grave›" in ctx
    assert "<grave>" not in ctx
    assert 'confiavel="nao"' in ctx


def test_validar_citacoes_so_do_mapa():
    _, mapa = montar_contexto(_evs())
    ids = validar_citacoes("Atraso [R1] e algo [R9] e de novo [R1]", mapa)
    assert ids == ["id-a"]


def test_remover_citacoes_invalidas():
    _, mapa = montar_contexto(_evs())
    limpo = remover_citacoes_invalidas("Relato [R1] e ruído [R99].", mapa)
    assert "[R1]" in limpo
    assert "[R99]" not in limpo

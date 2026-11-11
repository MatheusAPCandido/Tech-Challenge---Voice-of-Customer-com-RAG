"""T07 — métricas da comparação de embeddings (funções puras, sem modelo)."""

from __future__ import annotations

import numpy as np

from eval.comparar_embeddings import (
    bootstrap_diferenca,
    eh_relevante,
    precisao_at_k,
    prefixos,
    rr_at_k,
)


def test_precisao_at_k():
    rel = [True, False, True, True, False, False, False, False, False, True]
    assert precisao_at_k(rel, 5) == 0.6
    assert precisao_at_k(rel, 10) == 0.4


def test_precisao_lista_menor_que_k_conta_como_irrelevante():
    assert precisao_at_k([True, True], 5) == 0.4


def test_rr_at_k():
    assert rr_at_k([False, False, True], 10) == 1 / 3
    assert rr_at_k([False] * 10 + [True], 10) == 0.0


def test_relevancia_por_intersecao_de_temas():
    assert eh_relevante({"produto_defeito", "atendimento_loja"}, {"produto_defeito"})
    assert not eh_relevante({"entrega_rapida"}, {"atraso_entrega", "nao_recebido"})
    assert not eh_relevante(set(), {"produto_errado"})


def test_bootstrap_ic_contem_a_media_e_zero_quando_iguais():
    a = np.array([0.5, 0.7, 0.9, 0.3])
    media, lo, hi = bootstrap_diferenca(a, a, n=500)
    assert media == 0 and lo == 0 and hi == 0
    media, lo, hi = bootstrap_diferenca(a + 0.1, a, n=500)
    assert abs(media - 0.1) < 1e-9 and lo <= media <= hi


def test_prefixos_e5_e_bge():
    assert prefixos("intfloat/multilingual-e5-base") == ("query: ", "passage: ")
    assert prefixos("BAAI/bge-m3") == ("", "")

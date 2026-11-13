"""T28 — métricas e reconstrução das variações (funções puras, sem índice nem modelos)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from eval.avaliacao_final import bootstrap_diferenca, metricas_ranking, rankings_do_pool


def test_precisao_e_rr():
    rot = {"a": True, "b": False, "c": True}
    m = metricas_ranking(["b", "a", "c", "x"], rot, k=10)
    assert m["p@5"] == 2 / 5 and m["p@10"] == 2 / 10
    assert m["rr@10"] == 0.5
    assert m["nao_rotulados"] == 1 and m["n_itens"] == 4


def test_final_mede_sobre_entregues():
    rot = {"a": True, "b": False}
    m = metricas_ranking(["a", "b"], rot, k=10, por_entregues=True)
    assert m["p@10"] == 0.5 and m["p@5"] == 0.5
    vazio = metricas_ranking([], rot, k=10, por_entregues=True)
    assert math.isnan(vazio["p@10"])


def test_rankings_do_pool():
    pool = pd.DataFrame({
        "review_id": ["a", "b", "c", "d"],
        "score_ce": [0.9, 0.2, 0.7, np.nan],          # d veio só da vetorial pura (fora do pool do RRF)
        "score_rrf": [0.5, 0.9, 0.7, np.nan],
        "rank_vetorial": [2, None, 3, 1],
        "rank_bm25": [1, 2, None, None],
    })
    r = rankings_do_pool(pool, limiar=0.5, k=10)
    assert r["vetorial"] == ["d", "a", "c"]
    assert r["bm25"] == ["a", "b"]
    assert r["hibrida"] == ["b", "c", "a"]
    assert r["hibrida+reranker"] == ["a", "c", "b"]
    assert r["final"] == ["a", "c"]                    # b fica abaixo do limiar


def test_bootstrap():
    a = np.array([0.5, 0.6, 0.7])
    assert bootstrap_diferenca(a, a, n=200) == (0.0, 0.0, 0.0)
    media, lo, hi = bootstrap_diferenca(a + 0.2, a, n=200)
    assert abs(media - 0.2) < 1e-9 and lo <= media <= hi

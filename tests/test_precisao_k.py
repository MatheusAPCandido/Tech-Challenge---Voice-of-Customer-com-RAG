"""T17a — precisão@k (funções puras, sem XLSX/índice)."""

from __future__ import annotations

import pandas as pd

from eval.precisao_k import calcular, precisao_por_pergunta
from eval.rotulos_io import norm_relevante


def test_norm_relevante_1_0():
    assert norm_relevante("1") == "sim"
    assert norm_relevante(1) == "sim"
    assert norm_relevante("0") == "nao"
    assert norm_relevante(0) == "nao"
    assert norm_relevante("sim") == "sim"
    assert norm_relevante(None) is None


def _df_ranks(pid: str, relevantes: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [pid] * len(relevantes),
            "rank": list(range(1, len(relevantes) + 1)),
            "relevante": relevantes,
        }
    )


def test_precisao_at_10_sete_sim():
    df = _df_ranks("P01", ["sim"] * 7 + ["nao"] * 3)
    out = precisao_por_pergunta(df, 10)
    assert len(out) == 1
    assert out.iloc[0]["precisao@10"] == 0.7
    assert out.iloc[0]["n_sim@10"] == 7


def test_omite_pergunta_com_menos_de_k_rotulos():
    df = _df_ranks("P28", ["sim"] * 3)
    out = precisao_por_pergunta(df, 5)
    assert out.empty


def test_macro_media_duas_perguntas():
    df = pd.concat(
        [
            _df_ranks("P01", ["sim"] * 5),  # 1.0 @5
            _df_ranks("P02", ["sim", "nao", "nao", "nao", "nao"]),  # 0.2 @5
        ],
        ignore_index=True,
    )
    metricas = calcular(df, ks=(5,))
    assert len(metricas) == 2
    assert abs(metricas["precisao@5"].mean() - 0.6) < 1e-9

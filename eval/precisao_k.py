"""T17a — precisão@k a partir da planilha de rótulos (sem geração Gemini).

Macro-average: média das precisões por pergunta com ≥k rótulos no ranking.

Uso:
    python -m eval.precisao_k
    python -m eval.precisao_k --xlsx eval/rotulos_t15_manual.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eval.rotulos_io import ler_linhas_rotuladas

PASTA = Path(__file__).resolve().parent
XLSX_PADRAO = PASTA / "rotulos_t15_manual.xlsx"
KS_PADRAO = (5, 10)


def precisao_por_pergunta(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Uma linha por pergunta incluída: id, precisao@k, n_sim.

    Omite perguntas com menos de k linhas rotuladas (após ordenar por rank).
    """
    if k < 1:
        raise ValueError("k deve ser >= 1")
    linhas = []
    for pid, g in df.groupby("id", sort=True):
        g = g.sort_values("rank", kind="mergesort")
        if len(g) < k:
            continue
        top = g.head(k)
        n_sim = int((top["relevante"] == "sim").sum())
        linhas.append({
            "id": pid,
            f"precisao@{k}": n_sim / k,
            f"n_sim@{k}": n_sim,
        })
    return pd.DataFrame(linhas)


def calcular(df: pd.DataFrame, ks: tuple[int, ...] = KS_PADRAO) -> pd.DataFrame:
    """Junta precisão@k para vários k (inner join pelos ids elegíveis em todos os k).

    Na prática: começa pelo menor k e faz outer merge; médias usam só linhas
    não-nulas de cada coluna. Preferimos reportar por k independentemente:
    cada k tem seu próprio conjunto de perguntas com ≥k rótulos.
    """
    if df.empty:
        return pd.DataFrame(columns=["id"])

    base: pd.DataFrame | None = None
    for k in sorted(ks):
        parte = precisao_por_pergunta(df, k)
        if base is None:
            base = parte
        else:
            base = base.merge(parte, on="id", how="outer")
    return base if base is not None else pd.DataFrame(columns=["id"])


def rodar(caminho: Path = XLSX_PADRAO, ks: tuple[int, ...] = KS_PADRAO) -> pd.DataFrame:
    df = ler_linhas_rotuladas(caminho)
    metricas = calcular(df, ks)
    out = PASTA / "metricas_precisao.csv"
    metricas.to_csv(out, index=False)

    if metricas.empty:
        print("Nenhuma pergunta com rótulos suficientes para os k pedidos.")
        return metricas

    cols_show = ["id"]
    for k in sorted(ks):
        cols_show.extend([f"precisao@{k}", f"n_sim@{k}"])
    cols_show = [c for c in cols_show if c in metricas.columns]
    print(metricas[cols_show].to_string(index=False))

    print()
    for k in sorted(ks):
        col = f"precisao@{k}"
        if col not in metricas.columns:
            continue
        serie = metricas[col].dropna()
        if serie.empty:
            print(f"Macro precisão@{k}: n/a (0 perguntas)")
        else:
            print(f"Macro precisão@{k}: {serie.mean():.3f}  (n={len(serie)} perguntas)")
    print(f"\nGravado: {out}")
    return metricas


def main() -> None:
    parser = argparse.ArgumentParser(description="T17a: precisão@k a partir dos rótulos T15")
    parser.add_argument("--xlsx", type=Path, default=XLSX_PADRAO)
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=list(KS_PADRAO),
        help="Valores de k (default: 5 10)",
    )
    args = parser.parse_args()
    ks = tuple(sorted(set(args.k)))
    rodar(args.xlsx, ks)


if __name__ == "__main__":
    main()

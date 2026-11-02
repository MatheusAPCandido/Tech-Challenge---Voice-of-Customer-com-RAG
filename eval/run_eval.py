"""Avaliação da solução (Pessoa B — T17, T28).

Uso:
    python -m eval.run_eval

Gera eval/resultados.csv com a resposta de cada pergunta para rotulagem e métricas.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.pipeline import perguntar

PASTA = Path(__file__).resolve().parent


def rodar() -> pd.DataFrame:
    perguntas = pd.read_csv(PASTA / "perguntas.csv")
    linhas = []
    for _, p in perguntas.iterrows():
        r = perguntar(p["pergunta"])
        linhas.append({
            "id": p["id"],
            "tipo": p["tipo"],
            "esperado": p["esperado"],
            "recusou": r.recusada,
            "recusa_correta": r.recusada == (p["esperado"] == "recusar"),
            "n_evidencias": len(r.evidencias),
            "n_citacoes": len(r.citacoes),
            "resposta": r.texto,
            # TODO (T17): rotular manualmente a relevância de cada evidência → precisão@k
            # TODO (T17): fidelidade (afirmações com suporte / total), manual ou RAGAS
        })
    df = pd.DataFrame(linhas)
    df.to_csv(PASTA / "resultados.csv", index=False)
    print(df[["id", "tipo", "recusou", "recusa_correta", "n_citacoes"]].to_string(index=False))
    print(f"\nTaxa de recusa correta: {df['recusa_correta'].mean():.0%}")
    return df


if __name__ == "__main__":
    rodar()

"""Avaliação T17a — recusa correta (pipeline). Precisão@k: eval.precisao_k. Fidelidade: T17b.

Uso (busca real, geração mock — sem Gemini):
    USE_MOCK=false GEN_MOCK=true HF_HUB_OFFLINE=1 python -m eval.run_eval

JUDGE_ENABLED default false (reprodutível). true só se quiser medir o gate Ollama (ex. P09).

Gera eval/resultados.csv.
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
        esperado = str(p["esperado"]).strip()
        linhas.append({
            "id": p["id"],
            "tipo": p["tipo"],
            "esperado": esperado,
            "recusou": r.recusada,
            "motivo_recusa": r.motivo_recusa or "",
            "recusa_correta": r.recusada == (esperado == "recusar"),
            "n_evidencias": len(r.evidencias),
            "n_citacoes": len(r.citacoes),
            "resposta": r.texto,
            # T17a: precisão@k → python -m eval.precisao_k
            # T17b: fidelidade (afirmações com suporte), manual ou RAGAS — precisa geração real
        })
    df = pd.DataFrame(linhas)
    df.to_csv(PASTA / "resultados.csv", index=False)

    cols = ["id", "tipo", "esperado", "recusou", "motivo_recusa", "recusa_correta", "n_citacoes"]
    print(df[cols].to_string(index=False))
    print(f"\nTaxa de recusa correta: {df['recusa_correta'].mean():.0%}")

    falhas = df.loc[~df["recusa_correta"], "id"].tolist()
    if falhas:
        print(f"Falhas ({len(falhas)}): {', '.join(str(i) for i in falhas)}")
    else:
        print("Falhas: nenhuma")

    # Diagnóstico curto: esperado × motivo (útil P07/P09 vs over-refusal)
    diag = (
        df.assign(motivo=df["motivo_recusa"].replace("", "(respondeu)"))
        .groupby(["esperado", "motivo"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    print("\nEsperado × motivo_recusa:")
    print(diag.to_string(index=False))
    return df


if __name__ == "__main__":
    rodar()

"""Lê rotulos_t15.xlsx (coluna relevante preenchida) e sugere LIMIAR_RELEVANCIA.

Uso:
    python -m eval.calibrar_limiar
    python -m eval.calibrar_limiar --aplicar   # grava sugestão em .env.example e config

Calibra apenas score_ce (score do cross-encoder). Não altera o .env local.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from src.config import MIN_EVIDENCIAS

PASTA = Path(__file__).resolve().parent
RAIZ = PASTA.parent
XLSX = PASTA / "rotulos_t15.xlsx"
ENV_EXAMPLE = RAIZ / ".env.example"
CONFIG = RAIZ / "src" / "config.py"

HEADER_ROW = 4  # linha 1-indexada do openpyxl com nomes das colunas


def _norm_relevante(val) -> str | None:
    if val is None:
        return None
    s = str(val).strip().casefold()
    if not s or s == "none":
        return None
    if s in {"sim", "s", "yes", "y", "1", "true", "relevante"}:
        return "sim"
    if s in {"nao", "não", "n", "no", "0", "false", "irrelevante"}:
        return "nao"
    return None


def _ler_rotulos(caminho: Path = XLSX) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not caminho.exists():
        raise SystemExit(f"Arquivo não encontrado: {caminho}. Rode python -m eval.gerar_rotulos_t15")

    wb = load_workbook(caminho, read_only=True, data_only=True)
    indice = pd.read_excel(caminho, sheet_name="indice")
    linhas = []
    for nome in wb.sheetnames:
        if nome == "indice":
            continue
        ws = wb[nome]
        rows = list(ws.iter_rows(min_row=HEADER_ROW, values_only=True))
        if not rows:
            continue
        header = [str(c).strip() if c is not None else "" for c in rows[0]]
        if "relevante" not in header or "score_ce" not in header:
            continue
        idx = {h: i for i, h in enumerate(header)}
        for row in rows[1:]:
            if row is None or all(c is None for c in row):
                continue
            rel = _norm_relevante(row[idx["relevante"]] if idx["relevante"] < len(row) else None)
            if rel is None:
                continue
            score = row[idx["score_ce"]] if idx["score_ce"] < len(row) else None
            if score is None:
                continue
            linhas.append(
                {
                    "id": nome,
                    "score_ce": float(score),
                    "relevante": rel,
                    "review_id": row[idx["review_id"]] if "review_id" in idx else None,
                }
            )
    wb.close()
    if not linhas:
        raise SystemExit(
            "Nenhuma linha com `relevante` preenchido. "
            "Marque sim/nao nas abas P0x e rode de novo."
        )
    return indice, pd.DataFrame(linhas)


def _avaliar_limiar(df: pd.DataFrame, indice: pd.DataFrame, limiar: float) -> dict:
    uteis = df[df["relevante"] == "sim"]
    inuteis = df[df["relevante"] == "nao"]
    uteis_ret = int((uteis["score_ce"] >= limiar).sum())
    inuteis_ret = int((inuteis["score_ce"] >= limiar).sum())
    n_uteis = len(uteis)
    n_inuteis = len(inuteis)

    # perguntas que deveriam responder: quantas ficam sem evidência / abaixo de MIN
    meta = indice.set_index("id") if "id" in indice.columns else None
    zero_ev = 0
    abaixo_min = 0
    if meta is not None and "esperado" in meta.columns:
        for pid, g in df.groupby("id"):
            if pid not in meta.index:
                continue
            if str(meta.loc[pid, "esperado"]).strip().casefold() != "responder":
                continue
            n_ok = int((g["score_ce"] >= limiar).sum())
            if n_ok == 0:
                zero_ev += 1
            elif n_ok < MIN_EVIDENCIAS:
                abaixo_min += 1

    return {
        "limiar": limiar,
        "uteis_retidos": uteis_ret,
        "uteis_total": n_uteis,
        "inuteis_retidos": inuteis_ret,
        "inuteis_total": n_inuteis,
        "score": uteis_ret - inuteis_ret,
        "perguntas_zero_ev": zero_ev,
        "perguntas_abaixo_min": abaixo_min,
    }


def sugerir(caminho: Path = XLSX) -> float:
    indice, df = _ler_rotulos(caminho)
    scores = df["score_ce"]
    lo = max(0.05, float(scores.min()) - 0.05)
    hi = min(0.95, float(scores.max()) + 0.05)
    # grid 0.05
    candidatos = []
    x = round(lo, 2)
    while x <= hi + 1e-9:
        candidatos.append(round(x, 2))
        x += 0.05
    if not candidatos:
        candidatos = [0.5]

    resultados = [_avaliar_limiar(df, indice, lim) for lim in candidatos]
    tab = pd.DataFrame(resultados)
    print(tab.to_string(index=False))
    print(
        f"\nDiagnóstico: perguntas 'responder' com 0 evidências ≥ limiar / "
        f"< MIN_EVIDENCIAS={MIN_EVIDENCIAS} (não altera MIN neste PR)."
    )

    # maximizar (úteis − inúteis); empate → limiar mais alto
    melhor_score = tab["score"].max()
    finalistas = tab[tab["score"] == melhor_score]
    escolhido = float(finalistas["limiar"].max())
    print(f"\nLimiar sugerido (score_ce): {escolhido}")
    return escolhido


def _aplicar(limiar: float) -> None:
    if ENV_EXAMPLE.exists():
        texto = ENV_EXAMPLE.read_text(encoding="utf-8")
        novo, n = re.subn(
            r"^LIMIAR_RELEVANCIA=.*$",
            f"LIMIAR_RELEVANCIA={limiar}",
            texto,
            flags=re.M,
        )
        if n:
            ENV_EXAMPLE.write_text(novo, encoding="utf-8")
            print(f"Atualizado {ENV_EXAMPLE}")
        else:
            print(f"Aviso: LIMIAR_RELEVANCIA não encontrado em {ENV_EXAMPLE}")

    if CONFIG.exists():
        texto = CONFIG.read_text(encoding="utf-8")
        novo, n = re.subn(
            r'(LIMIAR_RELEVANCIA = float\(os\.getenv\("LIMIAR_RELEVANCIA", ")[^"]+("\)\))',
            rf"\g<1>{limiar}\2",
            texto,
        )
        if n:
            CONFIG.write_text(novo, encoding="utf-8")
            print(f"Atualizado default em {CONFIG}")
        else:
            print(f"Aviso: default LIMIAR_RELEVANCIA não encontrado em {CONFIG}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibra LIMIAR_RELEVANCIA a partir da planilha T15")
    parser.add_argument(
        "--aplicar",
        action="store_true",
        help="Grava o limiar sugerido em .env.example e no default de src/config.py",
    )
    parser.add_argument("--xlsx", type=Path, default=XLSX)
    args = parser.parse_args()
    limiar = sugerir(args.xlsx)
    if args.aplicar:
        _aplicar(limiar)
        print("Revise o README (seção limiar / T15) com o valor escolhido.")
    else:
        print("Para gravar a sugestão: python -m eval.calibrar_limiar --aplicar")


if __name__ == "__main__":
    main()

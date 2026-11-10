"""Leitura compartilhada das planilhas de rótulos T15 (sim/nao por evidência)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

HEADER_ROW = 4  # linha 1-indexada do openpyxl com nomes das colunas


def norm_relevante(val) -> str | None:
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


def ler_linhas_rotuladas(caminho: Path) -> pd.DataFrame:
    """Linhas com `relevante` preenchido: id, rank, relevante, score_ce?, review_id?."""
    if not caminho.exists():
        raise SystemExit(
            f"Arquivo não encontrado: {caminho}. "
            "Rode python -m eval.gerar_rotulos_t15 e preencha relevante, "
            "ou passe --xlsx com rotulos_t15_manual.xlsx"
        )

    wb = load_workbook(caminho, read_only=True, data_only=True)
    linhas = []
    for nome in wb.sheetnames:
        if nome == "indice":
            continue
        ws = wb[nome]
        rows = list(ws.iter_rows(min_row=HEADER_ROW, values_only=True))
        if not rows:
            continue
        header = [str(c).strip() if c is not None else "" for c in rows[0]]
        if "relevante" not in header:
            continue
        idx = {h: i for i, h in enumerate(header)}
        for row in rows[1:]:
            if row is None or all(c is None for c in row):
                continue
            rel = norm_relevante(row[idx["relevante"]] if idx["relevante"] < len(row) else None)
            if rel is None:
                continue

            item: dict = {"id": nome, "relevante": rel}
            if "rank" in idx and idx["rank"] < len(row) and row[idx["rank"]] is not None:
                try:
                    item["rank"] = int(row[idx["rank"]])
                except (TypeError, ValueError):
                    continue
            else:
                continue

            if "score_ce" in idx and idx["score_ce"] < len(row) and row[idx["score_ce"]] is not None:
                try:
                    item["score_ce"] = float(row[idx["score_ce"]])
                except (TypeError, ValueError):
                    pass

            if "review_id" in idx and idx["review_id"] < len(row):
                item["review_id"] = row[idx["review_id"]]

            linhas.append(item)
    wb.close()

    if not linhas:
        raise SystemExit(
            "Nenhuma linha com `relevante` preenchido. "
            "Marque sim/nao nas abas P0x e rode de novo."
        )
    return pd.DataFrame(linhas)

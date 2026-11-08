"""Gera planilha XLS para rotulagem humana do limiar (T15).

Uso:
    USE_MOCK=false python -m eval.gerar_rotulos_t15

Saída: eval/rotulos_t15.xlsx (aba indice + uma aba por pergunta).
Preencha a coluna `relevante` com sim/nao (ou 1/0), depois rode:
    python -m eval.calibrar_limiar
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font

from src.config import TOP_K_CANDIDATOS, USE_MOCK
from src.retrieve import buscar, processar_pergunta

PASTA = Path(__file__).resolve().parent
SAIDA = PASTA / "rotulos_t15.xlsx"

COLS = [
    "rank",
    "review_id",
    "score_ce",
    "score_rrf",
    "score_bm25",
    "score_vetorial",
    "rank_bm25",
    "rank_vetorial",
    "review_score",
    "estado",
    "categoria",
    "texto",
    "relevante",
]


def _escrever_cabecalho(ws, pergunta: str, tipo_csv: str, tipo_proc: str, esperado: str) -> None:
    ws["A1"] = "pergunta"
    ws["B1"] = pergunta
    ws["A2"] = "tipo_csv / tipo_processado / esperado"
    ws["B2"] = f"{tipo_csv} | {tipo_proc} | {esperado}"
    ws["A1"].font = Font(bold=True)
    ws["A2"].font = Font(bold=True)
    ws["B1"].alignment = Alignment(wrap_text=True)
    ws.merge_cells("B1:G1")


def _escrever_tabela(ws, evidencias, inicio_linha: int = 4) -> None:
    for col, nome in enumerate(COLS, start=1):
        cell = ws.cell(inicio_linha, col, nome)
        cell.font = Font(bold=True)

    for i, e in enumerate(evidencias, start=1):
        linha = inicio_linha + i
        valores = [
            i,
            e.review_id,
            round(e.score, 6) if e.score is not None else None,
            round(e.score_rrf, 6) if e.score_rrf is not None else None,
            round(e.score_bm25, 6) if e.score_bm25 is not None else None,
            round(e.score_vetorial, 6) if e.score_vetorial is not None else None,
            e.rank_bm25,
            e.rank_vetorial,
            e.review_score,
            e.estado,
            e.categoria,
            e.texto,
            "",  # relevante — preencher à mão
        ]
        for col, val in enumerate(valores, start=1):
            ws.cell(linha, col, val)
        ws.cell(linha, 12).alignment = Alignment(wrap_text=True)

    ws.column_dimensions["A"].width = 8
    ws.column_dimensions["B"].width = 36
    ws.column_dimensions["L"].width = 60
    ws.column_dimensions["M"].width = 12


def gerar(caminho: Path = SAIDA) -> Path:
    if USE_MOCK:
        raise SystemExit(
            "USE_MOCK=true: a planilha de calibração precisa do índice real. "
            "Rode com USE_MOCK=false."
        )

    perguntas = pd.read_csv(PASTA / "perguntas.csv")
    wb = Workbook()
    indice = wb.active
    indice.title = "indice"
    indice.append(["id", "tipo_csv", "tipo_processado", "esperado", "pergunta", "n_evidencias"])
    vazias: list[str] = []

    for _, p in perguntas.iterrows():
        pid = str(p["id"])
        pergunta = str(p["pergunta"])
        tipo_csv = str(p["tipo"])
        esperado = str(p["esperado"])
        consulta = processar_pergunta(pergunta)

        ws = wb.create_sheet(title=pid[:31])
        _escrever_cabecalho(ws, pergunta, tipo_csv, consulta.tipo, esperado)

        if consulta.tipo == "fora_de_escopo":
            ws["A4"] = (
                "fora_de_escopo — busca não executada (limiar não se aplica; "
                "guardrails recusam pelo tipo)."
            )
            indice.append([pid, tipo_csv, consulta.tipo, esperado, pergunta, 0])
            print(f"{pid}: fora_de_escopo (sem busca)")
            continue

        evidencias = buscar(
            consulta.consulta,
            consulta.filtros,
            k=TOP_K_CANDIDATOS,
            detalhado=True,
        )
        n = len(evidencias)
        indice.append([pid, tipo_csv, consulta.tipo, esperado, pergunta, n])
        print(f"{pid}: {n} evidências (tipo={consulta.tipo}, filtros={consulta.filtros})")
        if n == 0:
            vazias.append(pid)
            ws["A4"] = (
                "AVISO: busca retornou 0 evidências — não é fora_de_escopo. "
                "Regenere com USE_MOCK=false e índice carregado."
            )
            continue
        _escrever_tabela(ws, evidencias)

    if vazias:
        raise SystemExit(
            "Abortado: busca vazia em "
            + ", ".join(vazias)
            + ". Planilha NÃO foi salva. Verifique o índice (USE_MOCK=false) e rode de novo."
        )

    caminho.parent.mkdir(parents=True, exist_ok=True)
    tmp = caminho.with_suffix(caminho.suffix + ".tmp")
    wb.save(tmp)
    tmp.replace(caminho)
    print(f"Planilha salva em {caminho}")
    print("Preencha a coluna `relevante` (sim/nao ou 1/0) e rode: python -m eval.calibrar_limiar")
    return caminho


if __name__ == "__main__":
    gerar()

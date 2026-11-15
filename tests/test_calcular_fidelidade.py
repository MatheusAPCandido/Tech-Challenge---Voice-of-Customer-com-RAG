"""T17b — fidelidade (funções puras + leitura XLSX leve, sem Ollama/índice)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from openpyxl import Workbook

from eval.calcular_fidelidade import (
    calcular_metricas,
    ler_rotulos_fidelidade,
    texto_para_avaliar,
)
from src.agregar import MARCADOR_EXEMPLOS


def test_texto_para_avaliar_com_marcador():
    resposta = f"contagens fake\n\n{MARCADOR_EXEMPLOS}\n\nAtraso em 2 de 3 [R1]."
    assert texto_para_avaliar(resposta) == "Atraso em 2 de 3 [R1]."


def test_texto_para_avaliar_sem_marcador():
    resposta = "Clientes relatam atraso [R1]."
    assert texto_para_avaliar(resposta) == resposta


def test_macro_micro_uma_elegivel_e_uma_recusa():
    df = pd.DataFrame(
        [
            {
                "id": "P01",
                "esperado": "responder",
                "recusou": False,
                "n_afirmacoes": 4,
                "n_suportadas": 3,
            },
            {
                "id": "P07",
                "esperado": "recusar",
                "recusou": True,
                "n_afirmacoes": "",
                "n_suportadas": "",
            },
        ]
    )
    out = calcular_metricas(df)
    assert out["n"] == 1
    assert out["omitidos"] == ["P07"]
    assert abs(out["macro"] - 0.75) < 1e-9
    assert abs(out["micro"] - 0.75) < 1e-9
    assert list(out["por_pergunta"]["id"]) == ["P01"]


def test_micro_diferente_da_macro_com_duas_perguntas():
    # P01: 1/1 = 1.0; P02: 1/3 ≈ 0.333 → macro ≈ 0.667; micro = 2/4 = 0.5
    df = pd.DataFrame(
        [
            {
                "id": "P01",
                "esperado": "responder",
                "recusou": False,
                "n_afirmacoes": 1,
                "n_suportadas": 1,
            },
            {
                "id": "P02",
                "esperado": "responder",
                "recusou": False,
                "n_afirmacoes": 3,
                "n_suportadas": 1,
            },
        ]
    )
    out = calcular_metricas(df)
    assert out["n"] == 2
    assert abs(out["macro"] - (1.0 + 1 / 3) / 2) < 1e-9
    assert abs(out["micro"] - 0.5) < 1e-9


def test_contagem_invalida_em_elegivel_aborta():
    df = pd.DataFrame(
        [
            {
                "id": "P01",
                "esperado": "responder",
                "recusou": False,
                "n_afirmacoes": "",
                "n_suportadas": "",
            },
        ]
    )
    with pytest.raises(SystemExit, match="P01"):
        calcular_metricas(df)


def test_n_suportadas_maior_que_afirmacoes_aborta():
    df = pd.DataFrame(
        [
            {
                "id": "P02",
                "esperado": "responder",
                "recusou": "false",
                "n_afirmacoes": 2,
                "n_suportadas": 3,
            },
        ]
    )
    with pytest.raises(SystemExit, match="P02"):
        calcular_metricas(df)


def _aba_minima(
    ws,
    *,
    tipo: str,
    esperado: str,
    recusou: bool,
    motivo: str,
    n_afirmacoes,
    n_suportadas,
) -> None:
    pares = [
        (1, "pergunta", "pergunta teste"),
        (2, "tipo / esperado", f"{tipo} | {esperado}"),
        (3, "recusou / motivo_recusa", f"{recusou} | {motivo}"),
        (4, "citacoes", ""),
        (5, "resposta_avaliada", "claim [R1]"),
        (6, "n_afirmacoes", n_afirmacoes),
        (7, "n_suportadas", n_suportadas),
        (8, "observacoes", ""),
    ]
    for linha, rotulo, valor in pares:
        ws.cell(linha, 1, rotulo)
        ws.cell(linha, 2, valor)


def test_ler_xlsx_e_calcular(tmp_path: Path):
    caminho = tmp_path / "fidelidade_rotulos.xlsx"
    wb = Workbook()
    indice = wb.active
    indice.title = "indice"
    indice.append(["id"])

    p01 = wb.create_sheet("P01")
    _aba_minima(
        p01,
        tipo="especifico",
        esperado="responder",
        recusou=False,
        motivo="",
        n_afirmacoes=4,
        n_suportadas=3,
    )
    p07 = wb.create_sheet("P07")
    _aba_minima(
        p07,
        tipo="fora_de_escopo",
        esperado="recusar",
        recusou=True,
        motivo="fora_de_escopo",
        n_afirmacoes=None,
        n_suportadas=None,
    )
    wb.save(caminho)

    df = ler_rotulos_fidelidade(caminho)
    assert set(df["id"]) == {"P01", "P07"}
    out = calcular_metricas(df)
    assert out["n"] == 1
    assert out["omitidos"] == ["P07"]
    assert abs(out["macro"] - 0.75) < 1e-9

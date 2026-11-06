"""Etapa 1 — Construção da base de conhecimento (Pessoa A).

Uso:
    python -m src.prepare            # base completa
    python -m src.prepare --amostra  # também gera data/amostra.parquet (~1.000 reviews)

Tarefas: T03 (amostra), T05 (joins, limpeza, ausentes, metadados).

A preparação segue a arquitetura medallion (src/medallion/):
    CSVs Olist → bronze (cru) → silver (limpo e tipado) → gold (documentos do RAG)
A gold é gravada em data/reviews_docs.parquet com as colunas de schema.COLUNAS_DOCUMENTO.
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.config import AMOSTRA_PATH, DOCS_PATH
from src.medallion import bronze, gold, silver
from src.medallion.silver import limpar_texto  # noqa: F401  (reexportado para os outros módulos)
from src.schema import COLUNAS_DOCUMENTO


def montar_documentos() -> pd.DataFrame:
    """Executa bronze → silver → gold e devolve os documentos (regras em src/medallion/gold.py)."""
    bronze.executar()
    silver.executar()
    return gold.executar()


def gerar_amostra(docs: pd.DataFrame, n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """T03: amostra estratificada por nota para B e C trabalharem sem a base completa."""
    por_nota = max(1, n // docs["review_score"].nunique())
    return (
        docs.sample(frac=1, random_state=seed)
        .groupby("review_score")
        .head(por_nota)
        .sort_values("review_score")
        .reset_index(drop=True)
    )


def preparar_base(salvar_amostra: bool = False) -> pd.DataFrame:
    docs = montar_documentos()
    faltando = set(COLUNAS_DOCUMENTO) - set(docs.columns)
    if faltando:
        raise ValueError(f"Colunas do contrato ausentes: {faltando}")
    docs[COLUNAS_DOCUMENTO].to_parquet(DOCS_PATH, index=False)
    print(f"{len(docs)} documentos salvos em {DOCS_PATH}")
    if salvar_amostra:
        gerar_amostra(docs)[COLUNAS_DOCUMENTO].to_parquet(AMOSTRA_PATH, index=False)
        print(f"Amostra salva em {AMOSTRA_PATH}")
    return docs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--amostra", action="store_true")
    args = parser.parse_args()
    preparar_base(salvar_amostra=args.amostra)

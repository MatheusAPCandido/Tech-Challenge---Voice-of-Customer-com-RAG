"""Etapa 1 — Construção da base de conhecimento (Pessoa A).

Uso:
    python -m src.prepare            # base completa
    python -m src.prepare --amostra  # também gera data/amostra.parquet (~1.000 reviews)

Tarefas: T03 (amostra), T05 (joins, limpeza, ausentes, metadados).
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.config import AMOSTRA_PATH, DATA_DIR, DOCS_PATH
from src.schema import COLUNAS_DOCUMENTO


def carregar_csvs() -> dict[str, pd.DataFrame]:
    """Lê os CSVs do Kaggle que estão em data/."""
    arquivos = {
        "reviews": "olist_order_reviews_dataset.csv",
        "orders": "olist_orders_dataset.csv",
        "items": "olist_order_items_dataset.csv",
        "products": "olist_products_dataset.csv",
        "customers": "olist_customers_dataset.csv",
    }
    return {nome: pd.read_csv(DATA_DIR / arq) for nome, arq in arquivos.items()}


def limpar_texto(texto: str) -> str:
    """Limpeza leve: mantém acentos, caixa e stopwords (importantes para embeddings).

    TODO (T05):
      - normalizar quebras de linha (\\r\\n) e espaços repetidos
      - remover caracteres de controle
    """
    if not isinstance(texto, str):
        return ""
    return " ".join(texto.split())


def montar_documentos(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Junta as tabelas e devolve um DataFrame com as colunas de COLUNAS_DOCUMENTO.

    TODO (T05):
      1. Juntar título + mensagem em `texto` e aplicar limpar_texto.
      2. Remover reviews sem texto (documentar quantas foram removidas).
      3. Remover textos sem conteúdo (ex.: ".", "ok") com regra explícita.
      4. Join com orders → data_compra, dias_atraso, entregue.
      5. Join com items + products → categoria (agregar pedidos com vários itens).
      6. Join com customers → estado.
      7. Deduplicar textos idênticos.
      8. Criar coluna `tema` vazia (preenchida na T08).
    """
    raise NotImplementedError("Implementar na tarefa T05")


def gerar_amostra(docs: pd.DataFrame, n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """T03: amostra estratificada por nota para B e C trabalharem sem a base completa."""
    por_nota = max(1, n // docs["review_score"].nunique())
    return (
        docs.groupby("review_score", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), por_nota), random_state=seed))
        .reset_index(drop=True)
    )


def preparar_base(salvar_amostra: bool = False) -> pd.DataFrame:
    dfs = carregar_csvs()
    docs = montar_documentos(dfs)
    faltando = set(COLUNAS_DOCUMENTO) - set(docs.columns)
    if faltando:
        raise ValueError(f"Colunas do contrato ausentes: {faltando}")
    docs[COLUNAS_DOCUMENTO].to_parquet(DOCS_PATH, index=False)
    print(f"{len(docs)} documentos salvos em {DOCS_PATH}")
    if salvar_amostra:
        gerar_amostra(docs).to_parquet(AMOSTRA_PATH, index=False)
        print(f"Amostra salva em {AMOSTRA_PATH}")
    return docs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--amostra", action="store_true")
    args = parser.parse_args()
    preparar_base(salvar_amostra=args.amostra)

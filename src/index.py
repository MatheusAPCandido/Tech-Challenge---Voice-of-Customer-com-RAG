"""Embeddings, banco vetorial e índice BM25 (Pessoa A).

Uso:
    python -m src.index            # indexa data/reviews_docs.parquet
    python -m src.index --amostra  # indexa data/amostra.parquet

Tarefas: T06 (índice), T07 (comparar modelos de embedding).
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.config import AMOSTRA_PATH, CHROMA_DIR, COLECAO_CHROMA, DOCS_PATH, EMBEDDING_MODEL


def carregar_modelo_embedding(nome: str = EMBEDDING_MODEL):
    """Carrega o modelo de embedding multilíngue."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(nome)


def preparar_passagem(texto: str) -> str:
    """Modelos E5 exigem o prefixo 'passage: ' nos documentos e 'query: ' nas perguntas."""
    return f"passage: {texto}" if "e5" in EMBEDDING_MODEL.lower() else texto


def construir_indice(caminho_docs=DOCS_PATH) -> None:
    """Gera os embeddings e grava no ChromaDB com os metadados.

    TODO (T06):
      1. Ler o parquet.
      2. Gerar embeddings em lotes (normalize_embeddings=True).
      3. Gravar no ChromaDB (id = review_id, document = texto, metadados = demais colunas).
         Atenção: o Chroma não aceita None/NaN em metadados — converter antes.
      4. Construir e salvar o índice BM25 (rank_bm25) sobre o mesmo texto.
    """
    import chromadb

    docs = pd.read_parquet(caminho_docs)
    cliente = chromadb.PersistentClient(path=str(CHROMA_DIR))
    colecao = cliente.get_or_create_collection(COLECAO_CHROMA, metadata={"hnsw:space": "cosine"})
    _ = (docs, colecao)
    raise NotImplementedError("Implementar na tarefa T06")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--amostra", action="store_true")
    args = parser.parse_args()
    construir_indice(AMOSTRA_PATH if args.amostra else DOCS_PATH)

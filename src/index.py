"""Embeddings, banco vetorial e índice BM25 (Pessoa A).

Uso:
    python -m src.index            # indexa data/reviews_docs.parquet
    python -m src.index --amostra  # indexa data/amostra.parquet

Tarefas: T06 (índice), T07 (comparar modelos de embedding).

O que este módulo grava em chroma_db/ (fora do Git):
  - a coleção do ChromaDB: id = review_id, document = texto, embedding, metadados filtráveis
  - bm25.pkl: índice lexical BM25 sobre o mesmo texto, na mesma ordem de ids
  - indice_info.json: qual modelo, quantos documentos e de qual arquivo o índice foi gerado

Funções para quem consulta o índice (Pessoa B, retrieve.py):
  abrir_colecao(), carregar_bm25(), carregar_modelo_embedding(),
  preparar_consulta(), tokenizar(), ano_mes()
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import AMOSTRA_PATH, CHROMA_DIR, COLECAO_CHROMA, DOCS_PATH, EMBEDDING_MODEL

BM25_PATH = CHROMA_DIR / "bm25.pkl"
INFO_PATH = CHROMA_DIR / "indice_info.json"
LOTE_EMBEDDING = 64
LOTE_CHROMA = 2000


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def carregar_modelo_embedding(nome: str = EMBEDDING_MODEL):
    """Carrega o modelo de embedding multilíngue (usa a GPU automaticamente se houver)."""
    from sentence_transformers import SentenceTransformer

    modelo = SentenceTransformer(nome)
    print(f"[index] modelo {nome} carregado em {modelo.device}")
    return modelo


def _usa_prefixo_e5(nome: str = EMBEDDING_MODEL) -> bool:
    return "e5" in nome.lower()


def preparar_passagem(texto: str, nome: str = EMBEDDING_MODEL) -> str:
    """Modelos E5 exigem o prefixo 'passage: ' nos documentos e 'query: ' nas perguntas."""
    return f"passage: {texto}" if _usa_prefixo_e5(nome) else texto


def preparar_consulta(texto: str, nome: str = EMBEDDING_MODEL) -> str:
    """Use na pergunta antes de gerar o embedding dela (par de preparar_passagem)."""
    return f"query: {texto}" if _usa_prefixo_e5(nome) else texto


def gerar_embeddings(modelo, textos: list[str]):
    return modelo.encode(
        textos,
        batch_size=LOTE_EMBEDDING,
        normalize_embeddings=True,   # vetores de norma 1: similaridade de cosseno = produto escalar
        show_progress_bar=True,
        convert_to_numpy=True,
    )


# ---------------------------------------------------------------------------
# Metadados (o Chroma só aceita str, int, float e bool, sem nulos)
# ---------------------------------------------------------------------------
def ano_mes(periodo: str) -> int:
    """'2018-03' -> 201803. O Chroma só compara números com $gte/$lte, então os
    filtros de período (periodo_inicio / periodo_fim) usam este campo."""
    return int(periodo.replace("-", "")[:6])


def montar_metadados(linha) -> dict:
    meta = {
        "review_score": int(linha.review_score),
        "data_compra": str(linha.data_compra),
        "ano_mes": ano_mes(str(linha.data_compra)),
        "entregue": bool(linha.entregue),
        "categoria": str(linha.categoria),
        "estado": str(linha.estado),
        "tema": str(linha.tema or ""),
    }
    atraso = linha.dias_atraso
    if atraso is not None and not (isinstance(atraso, float) and math.isnan(atraso)):
        meta["dias_atraso"] = float(atraso)
        meta["atrasou"] = bool(atraso > 0)
    else:
        # pedido não entregue: sem dias_atraso; não conta como atrasado no filtro
        meta["atrasou"] = False
    return meta


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------
# Stopwords curtas e sem negações: "não", "nem" e "nunca" mudam o sentido de uma reclamação.
STOPWORDS = set("""
a o as os um uma uns umas de do da dos das em no na nos nas por pelo pela pelos pelas
para pra com e ou que se ao aos à às é foi ser era sao são este esta isto esse essa isso
aquele aquela meu minha seu sua eu ele ela eles elas voce você me te lhe nos já la lá
""".split())


def _sem_acento(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c))


def tokenizar(texto: str) -> list[str]:
    """Tokenização do BM25: minúsculas, sem acentos e sem pontuação.
    Use a MESMA função na pergunta, senão a busca lexical não casa as palavras."""
    texto = _sem_acento(texto.casefold())
    return [t for t in re.findall(r"\w+", texto) if t not in STOPWORDS]


def construir_bm25(ids: list[str], textos: list[str]) -> None:
    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi([tokenizar(t) for t in textos])
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"ids": ids, "bm25": bm25}, f)


def carregar_bm25() -> tuple[list[str], object]:
    """Devolve (ids, bm25). bm25.get_scores(tokenizar(pergunta))[i] é o score do documento ids[i]."""
    with open(BM25_PATH, "rb") as f:
        dados = pickle.load(f)
    return dados["ids"], dados["bm25"]


# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------
def _cliente():
    import chromadb

    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def abrir_colecao():
    """Abre a coleção já construída (para consultas)."""
    return _cliente().get_collection(COLECAO_CHROMA, embedding_function=None)


def construir_indice(caminho_docs=DOCS_PATH, modelo=None) -> dict:
    """Gera os embeddings e grava no ChromaDB com os metadados; constrói o BM25.

    Reconstrói do zero a cada execução, para o índice nunca misturar amostra e base completa.
    """
    inicio = time.time()
    caminho_docs = Path(caminho_docs)
    docs = pd.read_parquet(caminho_docs)
    if docs["review_id"].duplicated().any():
        raise ValueError("review_id repetido no parquet: rode python -m src.prepare de novo")
    print(f"[index] {len(docs)} documentos de {caminho_docs.name}")

    ids = docs["review_id"].astype(str).tolist()
    textos = docs["texto"].astype(str).tolist()
    metadados = [montar_metadados(linha) for linha in docs.itertuples(index=False)]

    # 1. embeddings
    modelo = modelo or carregar_modelo_embedding()
    vetores = gerar_embeddings(modelo, [preparar_passagem(t) for t in textos])
    print(f"[index] embeddings: {vetores.shape[0]} vetores de dimensão {vetores.shape[1]}")

    # 2. ChromaDB (recria a coleção)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    cliente = _cliente()
    try:
        cliente.delete_collection(COLECAO_CHROMA)
    except Exception:
        pass  # coleção ainda não existia
    colecao = cliente.create_collection(
        COLECAO_CHROMA, metadata={"hnsw:space": "cosine"}, embedding_function=None
    )
    for i in range(0, len(ids), LOTE_CHROMA):
        fim = i + LOTE_CHROMA
        colecao.add(
            ids=ids[i:fim],
            documents=textos[i:fim],
            embeddings=vetores[i:fim].tolist(),
            metadatas=metadados[i:fim],
        )
    print(f"[index] ChromaDB: {colecao.count()} documentos na coleção '{COLECAO_CHROMA}'")

    # 3. BM25
    construir_bm25(ids, textos)
    print(f"[index] BM25 salvo em {BM25_PATH}")

    # 4. registro do que foi indexado
    info = {
        "modelo": getattr(modelo, "nome_modelo", None) or EMBEDDING_MODEL,
        "dimensao": int(vetores.shape[1]),
        "documentos": len(ids),
        "fonte": caminho_docs.name,
        "colecao": COLECAO_CHROMA,
        "distancia": "cosine (Chroma devolve distância; similaridade = 1 - distância)",
        "metadados": sorted({k for m in metadados for k in m}),
        "criado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duracao_s": round(time.time() - inicio, 1),
    }
    INFO_PATH.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[index] pronto em {info['duracao_s']} s")
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--amostra", action="store_true")
    args = parser.parse_args()
    construir_indice(AMOSTRA_PATH if args.amostra else DOCS_PATH)

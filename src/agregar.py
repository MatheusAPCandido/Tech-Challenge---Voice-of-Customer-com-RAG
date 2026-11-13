"""T24: respostas agregadas com contagens por tema.

Para perguntas agregadas ("Quais são os principais problemas?"), a resposta começa com um
bloco de contagens por tema, calculado em código sobre todas as avaliações com texto
(data/gold/reviews_completo.parquet, temas da T08). Depois vêm os exemplos recuperados pelo
RAG, com as citações [R#]. O LLM não vê nem escreve esses números.

Critério de contagem: o tema principal de cada avaliação (cada uma conta uma vez).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from functools import lru_cache

import pandas as pd

from src.config import DATA_DIR
from src.medallion.temas import TEMAS
from src.retrieve import _sanitizar_filtros  # a mesma limpeza de filtros que a busca usa

logger = logging.getLogger(__name__)

GOLD_COMPLETO = DATA_DIR / "gold" / "reviews_completo.parquet"
COLUNAS = ["review_score", "data_compra", "dias_atraso", "categoria", "estado", "tema"]
GENERICOS = ("avaliacao_geral", "outros")  # entram no total, ficam fora do ranking
TOP_TEMAS = 5

# A avaliação de fidelidade (T17b/T28) deve considerar só o texto depois desta linha:
# os números do bloco vêm da base inteira, não do contexto recuperado.
MARCADOR_EXEMPLOS = "**Exemplos entre as avaliações recuperadas:**"


@lru_cache(maxsize=1)
def _carregar() -> pd.DataFrame:
    """Lê a gold uma vez por processo; as perguntas seguintes reaproveitam a tabela."""
    return pd.read_parquet(GOLD_COMPLETO, columns=COLUNAS)


def _normalizar(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto or "").encode("ascii", "ignore").decode("ascii")
    return sem_acento.lower()


def ajustar_recorte(pergunta: str, filtros: dict) -> dict:
    """Pergunta agregada sobre problemas, sem filtro de nota → notas 1 e 2.

    O pipeline usa o resultado na busca e na contagem, para as duas partes da resposta
    falarem do mesmo recorte. Com sinais dos dois lados ("positivos ou negativos"),
    não filtra, como faz a T11.
    """
    if "review_score_min" in filtros or "review_score_max" in filtros:
        return filtros  # a nota já foi escolhida (pela T11 ou na interface)
    texto = _normalizar(pergunta)
    negativo = re.search(r"problema|reclam|insatisf|negativ", texto)
    positivo = re.search(r"elogi|\bsatisf|positiv", texto)
    if negativo and not positivo:
        return {**filtros, "review_score_max": 2}
    return filtros


def _filtrar(df: pd.DataFrame, filtros: dict) -> pd.DataFrame:
    """Mesma semântica dos filtros da busca (ver retrieve._passar_filtro_mock)."""
    if "review_score_min" in filtros:
        df = df[df["review_score"] >= filtros["review_score_min"]]
    if "review_score_max" in filtros:
        df = df[df["review_score"] <= filtros["review_score_max"]]
    if "atrasou" in filtros:
        # pedido não entregue (dias_atraso nulo) conta como "não atrasou", como no índice
        df = df[(df["dias_atraso"] > 0) == filtros["atrasou"]]
    for coluna in ("categoria", "estado", "tema"):
        if coluna in filtros:
            df = df[df[coluna] == filtros[coluna]]
    if "periodo_inicio" in filtros:
        df = df[df["data_compra"] >= filtros["periodo_inicio"]]
    if "periodo_fim" in filtros:
        df = df[df["data_compra"] <= filtros["periodo_fim"]]
    return df


def contar_temas(filtros: dict) -> tuple[int, pd.Series]:
    """Total de avaliações no recorte e quantas há de cada tema principal (maior primeiro)."""
    recorte = _filtrar(_carregar(), _sanitizar_filtros(filtros))
    return len(recorte), recorte["tema"].value_counts()


def descrever_recorte(filtros: dict) -> str:
    """O que foi contado, dito por extenso, inclusive o que não foi filtrado."""
    f = _sanitizar_filtros(filtros)
    nmin, nmax = f.get("review_score_min", 1), f.get("review_score_max", 5)
    if (nmin, nmax) == (1, 5):
        partes = ["todas as notas"]
    elif nmin == nmax:
        partes = [f"nota {nmin}"]
    elif nmax - nmin == 1:
        partes = [f"notas {nmin} e {nmax}"]
    else:
        partes = [f"notas de {nmin} a {nmax}"]
    if "atrasou" in f:
        partes.append("entregas atrasadas" if f["atrasou"] else "entregas sem atraso")
    if "categoria" in f:
        partes.append(f"categoria {f['categoria']}")
    if "estado" in f:
        partes.append(f"estado {f['estado']}")
    inicio, fim = f.get("periodo_inicio"), f.get("periodo_fim")
    if inicio and fim:
        partes.append(f"compras de {inicio} a {fim}")
    elif inicio:
        partes.append(f"compras a partir de {inicio}")
    elif fim:
        partes.append(f"compras até {fim}")
    else:
        partes.append("todos os períodos")
    return " · ".join(partes)


def _milhar(n: int) -> str:
    return f"{n:,}".replace(",", ".")  # 41951 → 41.951


def _pct(parte: int, total: int) -> str:
    return f"{100 * parte / total:.1f}".replace(".", ",") + "%"  # 0.0864 → 8,6%


def bloco_contagens(filtros: dict) -> str:
    """Texto que abre a resposta agregada; vazio quando não há o que contar."""
    filtros = _sanitizar_filtros(filtros)
    if "tema" in filtros:
        return ""  # dentro de um único tema, um ranking de temas não informa nada
    try:
        total, contagem = contar_temas(filtros)
    except FileNotFoundError:
        logger.warning("T24: %s não encontrado; resposta sem contagens. Rode: python -m src.prepare",
                       GOLD_COMPLETO)
        return ""
    especificos = contagem.drop(labels=list(GENERICOS), errors="ignore").head(TOP_TEMAS)
    if total == 0 or especificos.empty:
        return ""
    sem_assunto = int(contagem.reindex(list(GENERICOS)).fillna(0).sum())
    linhas = [f"- {TEMAS.get(tema, tema)} — {_milhar(int(n))} ({_pct(int(n), total)})"
              for tema, n in especificos.items()]
    return "\n".join([
        f"**Na base inteira**, entre {_milhar(total)} avaliações com texto "
        f"({descrever_recorte(filtros)}), os assuntos que os clientes mais relatam são:",
        "",
        *linhas,
        f"- Sem assunto específico (opinião geral ou outros) — {_milhar(sem_assunto)} "
        f"({_pct(sem_assunto, total)})",
        "",
        "_Cada avaliação conta uma vez, pelo assunto principal. Os temas foram atribuídos por "
        "regras de palavras-chave, com cerca de 80% de acerto numa amostra conferida à mão (T08)._",
        "",
        MARCADOR_EXEMPLOS,
    ])

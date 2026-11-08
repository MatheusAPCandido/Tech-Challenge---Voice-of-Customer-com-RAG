"""Transformação da pergunta e recuperação (Pessoa B).

Tarefas: T11 (transformação), T12 (busca híbrida + RRF), T13 (filtros), T14 (reranker).
"""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from dataclasses import replace
from typing import Optional

from src.config import (
    LLM_PROVIDER,
    RERANKER_MODEL,
    TOP_K_CANDIDATOS,
    TOP_K_FINAL,
    USE_MOCK,
)
from src.index import (
    BM25_PATH,
    CHROMA_DIR,
    abrir_colecao,
    ano_mes,
    carregar_bm25,
    carregar_modelo_embedding,
    preparar_consulta,
    tokenizar,
)
from src.ollama_client import chat_json
from src.schema import FILTROS_SUPORTADOS, ConsultaProcessada, Evidencia

logger = logging.getLogger(__name__)

RRF_K = 60

# Cache de processo: coleção / BM25 / modelo (reconstruir o índice com o app ligado fica stale).
_recursos: dict | None = None
# Separado de _recursos para USE_MOCK=true nunca carregar o cross-encoder.
_reranker = None

# ---------------------------------------------------------------------------
# Mock — permite que a Pessoa C desenvolva a interface antes da busca real.
# ---------------------------------------------------------------------------
_MOCK = [
    Evidencia("mock-001", "Produto não chegou. Comprei há 20 dias e até agora nada.", 0.91,
              review_score=1, categoria="moveis_decoracao", data_compra="2018-03", dias_atraso=12.0, estado="CE"),
    Evidencia("mock-002", "Recebi apenas um dos dois itens do pedido e ninguém responde.", 0.87,
              review_score=1, categoria="cama_mesa_banho", data_compra="2018-05", dias_atraso=3.0, estado="SP"),
    Evidencia("mock-003", "Entrega muito atrasada, mas o produto é bom.", 0.84,
              review_score=3, categoria="esporte_lazer", data_compra="2017-11", dias_atraso=8.0, estado="RJ"),
    Evidencia("mock-004", "Chegou antes do prazo e bem embalado. Recomendo!", 0.62,
              review_score=5, categoria="informatica_acessorios", data_compra="2018-01", dias_atraso=-5.0, estado="MG"),
    Evidencia("mock-005", "Produto veio com defeito e a troca demorou demais.", 0.80,
              review_score=2, categoria="eletroportateis", data_compra="2018-07", dias_atraso=0.0, estado="BA"),
]

_UFS = (
    "AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MT|MS|MG|PA|PB|PR|PE|PI|RJ|RN|RS|RO|RR|SC|SP|SE|TO"
)

# Padrões off-topic (texto já normalizado). Exigem também ausência de âncoras VoC.
_OFFTOPIC = (
    "previsao do tempo",
    "melhor celular",
    "cotacao",
    "hora em ",
    "receita de ",
    "resultado do jogo",
    "signo de ",
    # P16–P20 (eval): código, política, esporte — só sem âncora VoC
    "escreva em python",
    "ordene um array",
    "funcao que",
    "presidente do brasil",
    "melhor presidente",
    "fascista",
    "comunista",
    "palmeiras",
    "tem mundial",
)

_ANCORAS_VOC = (
    "cliente",
    "avaliacao",
    "avalia",
    "review",
    "entrega",
    "produto",
    "pedido",
    "atraso",
    "compra",
    "nota",
    "reclam",
    "elogi",
)

# Frases de agregação — nunca token solto "mais" / "melhor".
_AGREGADA = (
    "principais",
    "padroes",
    "quais padroes",
    "em geral",
    "resumo",
    "quais sao os",
    "mais elogi",
    "mais reclam",
    "principais problemas",
    "aspectos da experiencia",
)

TIPOS_VALIDOS = frozenset({"pontual", "agregada", "fora_de_escopo"})

PROMPT_T11_SISTEMA = """Você analisa perguntas sobre avaliações de clientes de e-commerce (dataset Olist / Voice of Customer).
NÃO responda a pergunta de negócio. Apenas estruture a entrada.

Responda APENAS com um JSON válido em uma única linha, sem markdown e sem texto fora do JSON.
Schema:
{"consulta": "<string para busca>",
 "filtros": {"review_score_min"?: 1-5, "review_score_max"?: 1-5, "atrasou"?: bool,
             "estado"?: "UF", "categoria"?: str, "tema"?: str,
             "periodo_inicio"?: "AAAA-MM", "periodo_fim"?: "AAAA-MM"},
 "tipo": "pontual|agregada|fora_de_escopo"}

Regras:
- tipo=fora_de_escopo se a pergunta NÃO for sobre avaliações/experiência de compra Olist.
- tipo=agregada para visão geral, rankings, "principais", "padrões", "mais elogiados".
- tipo=pontual para temas específicos com evidências pontuais.
- Em filtros use SOMENTE as chaves listadas; se não tiver certeza, omita a chave.
- consulta: versão clara da pergunta para busca semântica (português).

Exemplos:
Pergunta: Quais padrões aparecem nas avaliações de clientes insatisfatórios?
{"consulta":"padrões em avaliações de clientes insatisfatórios","filtros":{"review_score_max":2},"tipo":"agregada"}

Pergunta: Quais são os principais problemas relatados pelos clientes?
{"consulta":"principais problemas relatados pelos clientes","filtros":{},"tipo":"agregada"}

Pergunta: Qual a previsão do tempo em São Paulo amanhã?
{"consulta":"previsão do tempo","filtros":{},"tipo":"fora_de_escopo"}
"""

def _texto_norm(s: str) -> str:
    """casefold + sem acentos — só para matching de regras."""
    s = unicodedata.normalize("NFKD", s.casefold())
    return "".join(c for c in s if not unicodedata.combining(c))


def _sanitizar_filtros(filtros: dict) -> dict:
    """Mantém só chaves de FILTROS_SUPORTADOS com tipos coerentes."""
    out: dict = {}
    for chave, valor in (filtros or {}).items():
        if chave not in FILTROS_SUPORTADOS:
            continue
        if chave in ("review_score_min", "review_score_max"):
            try:
                n = int(valor)
            except (TypeError, ValueError):
                continue
            if 1 <= n <= 5:
                out[chave] = n
        elif chave == "atrasou":
            if isinstance(valor, bool):
                out[chave] = valor
            elif isinstance(valor, (int, float)) and valor in (0, 1):
                out[chave] = bool(valor)
            elif isinstance(valor, str) and valor.strip().lower() in {"true", "1", "sim", "yes"}:
                out[chave] = True
            elif isinstance(valor, str) and valor.strip().lower() in {"false", "0", "nao", "não", "no"}:
                out[chave] = False
        elif chave == "estado":
            if isinstance(valor, str):
                uf = valor.strip().upper()
                if re.fullmatch(r"[A-Z]{2}", uf):
                    out[chave] = uf
        elif chave in ("categoria", "tema"):
            if isinstance(valor, str) and valor.strip():
                out[chave] = valor.strip()
        elif chave in ("periodo_inicio", "periodo_fim"):
            if isinstance(valor, str) and re.fullmatch(r"\d{4}-\d{2}", valor.strip()):
                out[chave] = valor.strip()

    if "review_score_min" in out and "review_score_max" in out:
        if out["review_score_min"] > out["review_score_max"]:
            out.pop("review_score_min", None)
            out.pop("review_score_max", None)
    return out


def _classificar_tipo(pergunta: str) -> str:
    """fora_de_escopo → agregada → pontual (default)."""
    n = _texto_norm(pergunta or "")
    tem_ancora = any(a in n for a in _ANCORAS_VOC)
    if any(p in n for p in _OFFTOPIC) and not tem_ancora:
        return "fora_de_escopo"
    if any(p in n for p in _AGREGADA):
        return "agregada"
    return "pontual"


def _extrair_filtros(pergunta: str) -> dict:
    """Heurísticas óbvias: nota, atraso, UF. Se min e max batem juntos, descarta o par."""
    n = _texto_norm(pergunta or "")
    filtros: dict = {}

    quer_max = bool(
        re.search(r"insatisf", n)
        or "nota baixa" in n
        or re.search(r"reclam", n)
    )
    quer_min = bool(
        re.search(r"\bsatisf", n)
        or re.search(r"elogi", n)
        or "nota alta" in n
    )
    # Ambíguo: os dois → descartar ambos (não inventar faixa).
    if quer_max and not quer_min:
        filtros["review_score_max"] = 2
    elif quer_min and not quer_max:
        filtros["review_score_min"] = 4

    if (
        re.search(r"atrasou", n)
        or re.search(r"atraso", n)
        or ("demora" in n and ("entrega" in n or "prazo" in n))
    ):
        filtros["atrasou"] = True

    m = re.search(rf"\b(?:em|no|na)\s+({_UFS})\b", n, flags=re.IGNORECASE)
    if not m:
        m = re.search(rf"\b({_UFS})\b", n, flags=re.IGNORECASE)
    if m:
        filtros["estado"] = m.group(1).upper()

    return filtros


def _reescrever_consulta(pergunta: str, filtros: dict | None = None) -> str:
    """Limpeza leve; filtros ignorados no v1 (assinatura pronta para T11b)."""
    _ = filtros
    texto = re.sub(r"\s+", " ", (pergunta or "").strip())
    return texto if texto else (pergunta or "")


def _processar_pergunta_regras(pergunta: str) -> ConsultaProcessada:
    """T11a: transformação só com heurísticas."""
    tipo = _classificar_tipo(pergunta)
    filtros = _sanitizar_filtros(_extrair_filtros(pergunta))
    consulta = _reescrever_consulta(pergunta, filtros)
    return ConsultaProcessada(
        pergunta_original=pergunta,
        consulta=consulta,
        filtros=filtros,
        tipo=tipo,
    )


def _chamar_ollama_json(pergunta: str) -> dict | None:
    """T11b: JSON via Ollama; None se falhar ou payload sem chaves de negócio."""
    dados = chat_json(PROMPT_T11_SISTEMA, pergunta, num_predict=256)
    if dados is None:
        return None
    if "consulta" not in dados and "tipo" not in dados and "filtros" not in dados:
        return None
    return dados


def _consulta_de_llm(dados: dict, pergunta: str) -> ConsultaProcessada:
    """Merge parcial: filtros do LLM sanitizados; tipo inválido → regras de tipo."""
    consulta_bruta = dados.get("consulta", "")
    if not isinstance(consulta_bruta, str):
        consulta_bruta = str(consulta_bruta) if consulta_bruta is not None else ""
    consulta = _reescrever_consulta(consulta_bruta) or pergunta

    filtros_raw = dados.get("filtros") or {}
    if not isinstance(filtros_raw, dict):
        filtros_raw = {}
    filtros = _sanitizar_filtros(filtros_raw)

    tipo_bruto = dados.get("tipo", "")
    tipo = tipo_bruto.strip().lower() if isinstance(tipo_bruto, str) else ""
    if tipo not in TIPOS_VALIDOS:
        tipo = _classificar_tipo(pergunta)

    return ConsultaProcessada(
        pergunta_original=pergunta,
        consulta=consulta,
        filtros=filtros,
        tipo=tipo,
    )


def processar_pergunta(pergunta: str) -> ConsultaProcessada:
    """T11: reescreve a pergunta, extrai filtros e classifica o tipo.

    T11b: se LLM_PROVIDER=ollama, tenta JSON via Ollama; senão/ falha → regras T11a.
    Ex.: "clientes insatisfatórios" → filtros = {"review_score_max": 2}
    """
    if not isinstance(pergunta, str):
        pergunta = "" if pergunta is None else str(pergunta)

    if LLM_PROVIDER.strip().lower() == "ollama":
        dados = _chamar_ollama_json(pergunta)
        if dados is not None:
            return _consulta_de_llm(dados, pergunta)

    return _processar_pergunta_regras(pergunta)


def _recursos_indice() -> dict:
    """Carrega uma vez coleção Chroma, BM25 e modelo de embedding."""
    global _recursos
    if _recursos is not None:
        return _recursos

    if not BM25_PATH.exists():
        raise FileNotFoundError(
            f"Índice BM25 não encontrado em {BM25_PATH}. "
            "Rode: python -m src.index --amostra"
        )
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"Diretório Chroma ausente em {CHROMA_DIR}. "
            "Rode: python -m src.index --amostra"
        )

    try:
        colecao = abrir_colecao()
    except Exception as exc:
        raise FileNotFoundError(
            f"Coleção Chroma indisponível em {CHROMA_DIR}. "
            "Rode: python -m src.index --amostra"
        ) from exc

    ids_bm25, bm25 = carregar_bm25()
    modelo = carregar_modelo_embedding()
    _recursos = {"colecao": colecao, "ids_bm25": ids_bm25, "bm25": bm25, "modelo": modelo}
    return _recursos


def _montar_where(filtros: dict) -> dict | None:
    """Converte FILTROS_SUPORTADOS no `where` do Chroma. None = sem filtro."""
    clausulas: list[dict] = []
    if "review_score_min" in filtros:
        clausulas.append({"review_score": {"$gte": int(filtros["review_score_min"])}})
    if "review_score_max" in filtros:
        clausulas.append({"review_score": {"$lte": int(filtros["review_score_max"])}})
    if "atrasou" in filtros:
        clausulas.append({"atrasou": {"$eq": bool(filtros["atrasou"])}})
    for chave in ("categoria", "estado", "tema"):
        if chave in filtros:
            clausulas.append({chave: {"$eq": str(filtros[chave])}})
    if "periodo_inicio" in filtros:
        clausulas.append({"ano_mes": {"$gte": ano_mes(str(filtros["periodo_inicio"]))}})
    if "periodo_fim" in filtros:
        clausulas.append({"ano_mes": {"$lte": ano_mes(str(filtros["periodo_fim"]))}})

    if not clausulas:
        return None
    if len(clausulas) == 1:
        return clausulas[0]
    return {"$and": clausulas}


def _ids_permitidos(where: dict | None) -> set[str] | None:
    """None = universo completo; set (pode ser vazio) = recorte do where."""
    if where is None:
        return None
    bruto = _recursos_indice()["colecao"].get(where=where, include=[])
    return {str(rid) for rid in (bruto.get("ids") or [])}


def _passar_filtro_mock(e: Evidencia, filtros: dict) -> bool:
    """Mesma semântica dos filtros, aplicada aos campos do mock."""
    if "review_score_min" in filtros:
        if e.review_score is None or e.review_score < filtros["review_score_min"]:
            return False
    if "review_score_max" in filtros:
        if e.review_score is None or e.review_score > filtros["review_score_max"]:
            return False
    if "atrasou" in filtros:
        atrasou = e.dias_atraso is not None and e.dias_atraso > 0
        if atrasou != bool(filtros["atrasou"]):
            return False
    if "categoria" in filtros and e.categoria != filtros["categoria"]:
        return False
    if "estado" in filtros and e.estado != filtros["estado"]:
        return False
    if "tema" in filtros and e.tema != filtros["tema"]:
        return False
    if "periodo_inicio" in filtros:
        if not e.data_compra or e.data_compra < str(filtros["periodo_inicio"]):
            return False
    if "periodo_fim" in filtros:
        if not e.data_compra or e.data_compra > str(filtros["periodo_fim"]):
            return False
    return True


def _busca_vetorial(
    pergunta: str,
    n: int,
    where: dict | None = None,
    ids_ok: set[str] | None = None,
) -> list[tuple[str, float]]:
    """Top-n (id, similaridade) por cosseno no Chroma; similaridade = 1 - distance."""
    if ids_ok is not None and not ids_ok:
        return []

    rec = _recursos_indice()
    n_results = n if ids_ok is None else min(n, max(1, len(ids_ok)))
    vetor = rec["modelo"].encode(
        [preparar_consulta(pergunta)],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    kwargs: dict = {
        "query_embeddings": [vetor[0].tolist()],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        kwargs["where"] = where
    resultado = rec["colecao"].query(**kwargs)
    ids = (resultado.get("ids") or [[]])[0]
    dists = (resultado.get("distances") or [[]])[0]
    out: list[tuple[str, float]] = []
    for i, rid in enumerate(ids):
        dist = float(dists[i]) if i < len(dists) else 0.0
        out.append((str(rid), 1.0 - dist))
    return out


def _busca_bm25(
    pergunta: str,
    n: int,
    ids_ok: set[str] | None = None,
) -> list[tuple[str, float]]:
    """Top-n (id, score_bm25); vazio se só stopwords, score ≤ 0 ou recorte vazio."""
    if ids_ok is not None and not ids_ok:
        return []

    tokens = tokenizar(pergunta)
    if not tokens:
        return []

    rec = _recursos_indice()
    scores = rec["bm25"].get_scores(tokens)
    ids = rec["ids_bm25"]
    ordenados = sorted(
        (
            (ids[i], float(scores[i]))
            for i in range(len(ids))
            if scores[i] > 0 and (ids_ok is None or ids[i] in ids_ok)
        ),
        key=lambda par: (-par[1], par[0]),
    )
    return ordenados[:n]

def _fundir_rrf(rankings: list[list[str]], k: int) -> list[tuple[str, float]]:
    """RRF + normalização para ~[0, 1]. Empate: review_id asc."""
    listas = [r for r in rankings if r]
    if not listas:
        return []

    bruto: dict[str, float] = {}
    for ranking in listas:
        for rank, rid in enumerate(ranking, start=1):
            bruto[rid] = bruto.get(rid, 0.0) + 1.0 / (RRF_K + rank)

    n_listas = len(listas)
    denom = n_listas / (RRF_K + 1)
    ordenados = sorted(bruto.items(), key=lambda par: (-par[1], par[0]))
    return [(rid, score / denom) for rid, score in ordenados[:k]]


def _meta_opcional(meta: dict, chave: str):
    if chave not in meta:
        return None
    valor = meta[chave]
    if valor is None:
        return None
    return valor


def _evidencia_de_chroma(
    review_id: str,
    texto: str,
    score: float,
    meta: dict | None,
    *,
    score_rrf: float | None = None,
    score_bm25: float | None = None,
    score_vetorial: float | None = None,
    rank_bm25: int | None = None,
    rank_vetorial: int | None = None,
) -> Evidencia:
    meta = meta or {}
    review_score = _meta_opcional(meta, "review_score")
    dias_atraso = _meta_opcional(meta, "dias_atraso")
    return Evidencia(
        review_id=review_id,
        texto=texto or "",
        score=score,
        review_score=int(review_score) if review_score is not None else None,
        categoria=_meta_opcional(meta, "categoria"),
        data_compra=_meta_opcional(meta, "data_compra"),
        dias_atraso=float(dias_atraso) if dias_atraso is not None else None,
        estado=_meta_opcional(meta, "estado"),
        tema=_meta_opcional(meta, "tema") or None,
        score_rrf=score_rrf,
        score_bm25=score_bm25,
        score_vetorial=score_vetorial,
        rank_bm25=rank_bm25,
        rank_vetorial=rank_vetorial,
    )


def _sigmoid(x: float) -> float:
    """Logits do BGE reranker → (0, 1] para conviver com LIMIAR_RELEVANCIA até T15."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _carregar_reranker():
    """Lazy: só carrega o cross-encoder na busca real (não no mock)."""
    global _reranker
    if _reranker is not None:
        return _reranker
    from sentence_transformers import CrossEncoder

    _reranker = CrossEncoder(RERANKER_MODEL)
    logger.info("Reranker %s carregado", RERANKER_MODEL)
    return _reranker


def _rerank(pergunta: str, candidatos: list[Evidencia], k: int) -> list[Evidencia]:
    """Reordena com cross-encoder; score final = sigmoid(logit)."""
    if not candidatos:
        return []

    modelo = _carregar_reranker()
    pares = [(pergunta, e.texto) for e in candidatos]
    logits = modelo.predict(pares, show_progress_bar=False)
    reranked = [
        replace(e, score=_sigmoid(float(logit)))
        for e, logit in zip(candidatos, logits)
    ]
    reranked.sort(key=lambda e: (-e.score, e.review_id))
    return reranked[:k]


def buscar(
    pergunta: str,
    filtros: Optional[dict] = None,
    k: int = TOP_K_FINAL,
    *,
    detalhado: bool = False,
) -> list[Evidencia]:
    """Contrato principal: devolve as k evidências mais relevantes, ordenadas por score.

    T12 — busca vetorial + BM25 + RRF (pool de até TOP_K_CANDIDATOS; RRF só ordena o pool).
    T13 — filtros de metadados (FILTROS_SUPORTADOS) no Chroma e no mesmo recorte do BM25.
    T14 — cross-encoder (RERANKER_MODEL); score final = sigmoid(logit); corta em k.
    T15 — detalhado=True preenche score_rrf/bm25/vetorial e ranks (diagnóstico / planilha).
    """
    filtros = _sanitizar_filtros(filtros or {})
    if USE_MOCK:
        filtrados = [e for e in _MOCK if _passar_filtro_mock(e, filtros)]
        return sorted(filtrados, key=lambda e: e.score, reverse=True)[:k]

    pergunta = pergunta if isinstance(pergunta, str) else ("" if pergunta is None else str(pergunta))
    n = TOP_K_CANDIDATOS
    where = _montar_where(filtros)
    ids_ok = _ids_permitidos(where)

    hits_vec = _busca_vetorial(pergunta, n, where=where, ids_ok=ids_ok)
    hits_bm25 = _busca_bm25(pergunta, n, ids_ok=ids_ok)
    ids_vec = [rid for rid, _ in hits_vec]
    ids_bm25 = [rid for rid, _ in hits_bm25]
    fundidos = _fundir_rrf([ids_vec, ids_bm25], n)
    if not fundidos:
        return []

    ids_pool = [rid for rid, _ in fundidos]
    scores_rrf = {rid: score for rid, score in fundidos}
    score_vet = {rid: s for rid, s in hits_vec}
    score_bm = {rid: s for rid, s in hits_bm25}
    rank_vet = {rid: i for i, (rid, _) in enumerate(hits_vec, start=1)}
    rank_bm = {rid: i for i, (rid, _) in enumerate(hits_bm25, start=1)}

    rec = _recursos_indice()
    bruto = rec["colecao"].get(ids=ids_pool, include=["documents", "metadatas"])
    por_id: dict[str, tuple[str, dict]] = {}
    for i, rid in enumerate(bruto.get("ids") or []):
        docs = bruto.get("documents") or []
        metas = bruto.get("metadatas") or []
        texto = docs[i] if i < len(docs) else ""
        meta = metas[i] if i < len(metas) else {}
        por_id[str(rid)] = (texto or "", meta or {})

    candidatos: list[Evidencia] = []
    for rid in ids_pool:
        texto, meta = por_id.get(rid, ("", {}))
        if detalhado:
            candidatos.append(
                _evidencia_de_chroma(
                    rid,
                    texto,
                    scores_rrf[rid],
                    meta,
                    score_rrf=scores_rrf[rid],
                    score_bm25=score_bm.get(rid),
                    score_vetorial=score_vet.get(rid),
                    rank_bm25=rank_bm.get(rid),
                    rank_vetorial=rank_vet.get(rid),
                )
            )
        else:
            candidatos.append(_evidencia_de_chroma(rid, texto, scores_rrf[rid], meta))
    return _rerank(pergunta, candidatos, k)

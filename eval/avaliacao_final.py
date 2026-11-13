"""T28 — avaliação final e comparação de variações do RAG.

Uso (no terminal, na raiz do repositório):
    python -m eval.avaliacao_final                      # coleta, calcula e gera a planilha do que falta rotular
    python -m eval.avaliacao_final --importar-rotulos   # copia a planilha preenchida para eval/t28_rotulos.csv
    python -m eval.avaliacao_final --rotulo bge-m3      # nova execução com outro índice (ex.: depois de trocar o modelo)
    python -m eval.avaliacao_final --com-geracao        # também gera as respostas (precisa de Gemini ou Ollama)

O script força USE_MOCK=false (índice real) e, sem --com-geracao, GEN_MOCK=true e JUDGE_ENABLED=false,
para que a avaliação seja reprodutível e não dependa de chaves de API.

O que é comparado (perguntas de eval/perguntas.csv, top-10)
----------------------------------------------------------
Uma busca por pergunta, com os filtros que a T11 extrai, reconstrói todas as variações:

    vetorial          só a busca por embeddings (T06)
    bm25              só a busca por palavras (T06)
    hibrida           vetorial + BM25 fundidos por RRF (T12)
    hibrida+reranker  RRF seguido do cross-encoder: o pipeline atual (T14)
    final             o anterior só com evidências acima do limiar (T15): o que chega ao LLM
    sem filtros       hibrida+reranker sem os filtros da T11 (T13), nas perguntas que têm filtro

Métricas
--------
- Busca: precisão@5, precisão@10 e MRR@10, com rótulos humanos (relevante sim/nao por review).
  A diferença para o pipeline atual vem com IC95% por bootstrap pareado sobre as perguntas.
- Recusa: % de recusas corretas (recusar quando deve, responder quando deve), sem guardrail e com
  vários limiares, usando a própria função avaliar_evidencias (T16).
- Custo: tempo de cada etapa da busca por pergunta.
- Geração (opcional): % respondidas, citações por resposta e latência; planilha para julgar a fidelidade.

Rótulos (quem diz o que é relevante)
------------------------------------
Lidos de eval/t28_rotulos.csv (versionado) e de eval/rotulos_t15_manual.xlsx (rótulos da T15, se houver).
O que nenhuma variação tiver rotulado vai para eval/t28_rotulagem.xlsx: planilha cega e embaralhada,
sem dizer de qual variação veio cada review. Depois de preencher, rode --importar-rotulos.

Saídas (eval/): t28_resumo.md, t28_por_pergunta.csv, t28_recusa.csv, t28_rotulos.csv
Cache das buscas: data/t28/<rotulo>.parquet (fora do Git; use --refazer-busca para recalcular).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Antes de importar src.config: índice real e avaliação reprodutível
_COM_GERACAO = "--com-geracao" in sys.argv
os.environ["USE_MOCK"] = "false"
if not _COM_GERACAO:
    os.environ["GEN_MOCK"] = "true"
os.environ.setdefault("JUDGE_ENABLED", "false")

import random  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import (  # noqa: E402
    DATA_DIR,
    EMBEDDING_MODEL,
    LIMIAR_RELEVANCIA,
    LLM_PROVIDER,
    RERANKER_MODEL,
    TOP_K_CANDIDATOS,
    TOP_K_FINAL,
)

PASTA = Path(__file__).resolve().parent
PERGUNTAS_PATH = PASTA / "perguntas.csv"
ROTULOS_CSV = PASTA / "t28_rotulos.csv"
ROTULOS_T15 = PASTA / "rotulos_t15_manual.xlsx"
ROTULAGEM_PATH = PASTA / "t28_rotulagem.xlsx"
RESUMO_PATH = PASTA / "t28_resumo.md"
POR_PERGUNTA_PATH = PASTA / "t28_por_pergunta.csv"
RECUSA_PATH = PASTA / "t28_recusa.csv"
FIDELIDADE_PATH = PASTA / "t28_fidelidade.xlsx"
CACHE_DIR = DATA_DIR / "t28"

K = TOP_K_FINAL
VARIACOES = ["vetorial", "bm25", "hibrida", "hibrida+reranker", "final"]
REFERENCIA = "hibrida+reranker"
LIMIARES = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
N_BOOTSTRAP = 10_000
SEED = 42


# ---------------------------------------------------------------------------
# Métricas (funções puras, testadas em tests/test_avaliacao_final.py)
# ---------------------------------------------------------------------------
def metricas_ranking(ids: list[str], rotulos: dict[str, bool], k: int = 10, por_entregues: bool = False) -> dict:
    """Precisão@5, precisão@k e RR@k de um ranking.

    Uma review sem rótulo entra em `nao_rotulados` e conta como não relevante (estimativa conservadora).
    Com `por_entregues=True` (variação `final`, cortada pelo limiar), a precisão é medida sobre o que
    foi de fato entregue ao LLM: relevantes / min(k, entregues). Sem nenhuma entregue, fica NaN.
    """
    top = ids[:k]
    rel = [rotulos.get(r, False) for r in top]
    if por_entregues:
        p5 = sum(rel[:5]) / min(5, len(rel)) if rel else float("nan")
        pk = sum(rel) / len(rel) if rel else float("nan")
    else:
        p5 = sum(rel[:5]) / 5
        pk = sum(rel) / k
    rr = next((1.0 / i for i, x in enumerate(rel, start=1) if x), 0.0)
    return {"p@5": p5, f"p@{k}": pk, f"rr@{k}": rr, "n_itens": len(top),
            "nao_rotulados": sum(r not in rotulos for r in top)}


def bootstrap_diferenca(a, b, n: int = N_BOOTSTRAP, seed: int = SEED) -> tuple[float, float, float]:
    """Média de (a - b) e IC95% por bootstrap pareado."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(diff) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    medias = diff[rng.integers(0, len(diff), size=(n, len(diff)))].mean(axis=1)
    return float(diff.mean()), float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def rankings_do_pool(pool: pd.DataFrame, limiar: float, k: int = K) -> dict[str, list[str]]:
    """Reconstrói as variações a partir do pool de uma pergunta (uma linha por review)."""
    def ordem(col: str, asc: bool) -> list[str]:
        sub = pool.dropna(subset=[col]).sort_values([col, "review_id"], ascending=[asc, True])
        return sub["review_id"].astype(str).tolist()

    rerank = ordem("score_ce", False)
    acima = pool[pool["score_ce"] >= limiar]
    final = [r for r in rerank[:k] if r in set(acima["review_id"].astype(str))]
    return {
        "vetorial": ordem("rank_vetorial", True)[:k],
        "bm25": ordem("rank_bm25", True)[:k],
        "hibrida": ordem("score_rrf", False)[:k],
        "hibrida+reranker": rerank[:k],
        "final": final,
    }


# ---------------------------------------------------------------------------
# Coleta (uma busca por pergunta)
# ---------------------------------------------------------------------------
def _pool_da_busca(texto: str, filtros: dict) -> tuple[pd.DataFrame, dict]:
    """Pool completo do pipeline + top-k exatos da vetorial e do BM25, com os tempos de cada etapa."""
    from src import retrieve

    tempos = {}
    filtros = retrieve._sanitizar_filtros(filtros)
    where = retrieve._montar_where(filtros)
    ids_ok = retrieve._ids_permitidos(where)

    t = time.perf_counter()
    vet = retrieve._busca_vetorial(texto, TOP_K_CANDIDATOS, where=where, ids_ok=ids_ok)
    tempos["vetorial_s"] = time.perf_counter() - t
    t = time.perf_counter()
    bm = retrieve._busca_bm25(texto, TOP_K_CANDIDATOS, ids_ok=ids_ok)
    tempos["bm25_s"] = time.perf_counter() - t
    t = time.perf_counter()
    evid = retrieve.buscar(texto, filtros, k=TOP_K_CANDIDATOS, detalhado=True)
    tempos["busca_completa_s"] = time.perf_counter() - t

    linhas = {}
    for e in evid:
        linhas[e.review_id] = {
            "review_id": e.review_id, "texto": e.texto, "score_ce": e.score, "score_rrf": e.score_rrf,
            "rank_vetorial": e.rank_vetorial, "rank_bm25": e.rank_bm25,
        }
    # A vetorial e o BM25 puros podem ter itens que o RRF cortou do pool: entram sem score do reranker
    faltando = [rid for rid, _ in vet[:K] + bm[:K] if rid not in linhas]
    if faltando:
        bruto = retrieve._recursos_indice()["colecao"].get(ids=faltando, include=["documents"])
        textos = dict(zip(bruto.get("ids") or [], bruto.get("documents") or []))
        for rid in faltando:
            linhas[rid] = {"review_id": rid, "texto": textos.get(rid, ""), "score_ce": np.nan,
                           "score_rrf": np.nan, "rank_vetorial": None, "rank_bm25": None}
    # Ranks exatos das listas puras (itens além do top-k que o RRF cortou não importam para @k)
    for i, (rid, _) in enumerate(vet, start=1):
        if rid in linhas:
            linhas[rid]["rank_vetorial"] = i
    for i, (rid, _) in enumerate(bm, start=1):
        if rid in linhas:
            linhas[rid]["rank_bm25"] = i
    return pd.DataFrame(list(linhas.values())), tempos


def coletar(rotulo: str, refazer: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devolve (pools, perguntas processadas). Usa cache em data/t28/."""
    from src.retrieve import processar_pergunta

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_pools, cache_perg = CACHE_DIR / f"{rotulo}.parquet", CACHE_DIR / f"{rotulo}_perguntas.parquet"
    if cache_pools.exists() and cache_perg.exists() and not refazer:
        print(f"[t28] usando a busca em cache ({cache_pools.name}); --refazer-busca para recalcular")
        return pd.read_parquet(cache_pools), pd.read_parquet(cache_perg)

    perguntas = pd.read_csv(PERGUNTAS_PATH, dtype=str)
    pools, info = [], []
    for p in perguntas.itertuples(index=False):
        consulta = processar_pergunta(p.pergunta)
        linha = {"id": p.id, "tipo_csv": p.tipo, "esperado": p.esperado.strip(), "pergunta": p.pergunta,
                 "tipo_t11": consulta.tipo, "consulta": consulta.consulta,
                 "filtros": str(consulta.filtros) if consulta.filtros else ""}
        if consulta.tipo != "fora_de_escopo":
            pool, tempos = _pool_da_busca(consulta.consulta, consulta.filtros)
            pools.append(pool.assign(id=p.id, com_filtros=True))
            linha.update(tempos)
            if consulta.filtros:  # T13: mesma pergunta sem filtros
                pool_sf, _ = _pool_da_busca(consulta.consulta, {})
                pools.append(pool_sf.assign(id=p.id, com_filtros=False))
        info.append(linha)
        print(f"[t28] {p.id}: tipo={consulta.tipo} filtros={consulta.filtros or '-'}")

    pools_df = pd.concat(pools, ignore_index=True) if pools else pd.DataFrame()
    perg_df = pd.DataFrame(info)
    pools_df.to_parquet(cache_pools, index=False)
    perg_df.to_parquet(cache_perg, index=False)
    return pools_df, perg_df


# ---------------------------------------------------------------------------
# Rótulos
# ---------------------------------------------------------------------------
def carregar_rotulos() -> dict[tuple[str, str], bool]:
    rotulos: dict[tuple[str, str], bool] = {}
    if ROTULOS_T15.exists():
        from eval.rotulos_io import ler_linhas_rotuladas

        t15 = ler_linhas_rotuladas(ROTULOS_T15)
        for r in t15.dropna(subset=["review_id"]).itertuples(index=False):
            rotulos[(str(r.id), str(r.review_id))] = r.relevante == "sim"
        print(f"[t28] {len(rotulos)} rótulos lidos de {ROTULOS_T15.name}")
    if ROTULOS_CSV.exists():
        csv = pd.read_csv(ROTULOS_CSV, dtype=str)
        for r in csv.itertuples(index=False):
            rotulos[(r.id, r.review_id)] = r.relevante.strip().lower() == "sim"
        print(f"[t28] {len(csv)} rótulos lidos de {ROTULOS_CSV.name}")
    return rotulos


def escrever_planilha_cega(faltando: pd.DataFrame, perguntas: pd.DataFrame) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font

    texto_perg = perguntas.set_index("id")["pergunta"].to_dict()
    rng = random.Random(SEED)
    wb = Workbook()
    ws = wb.active
    ws.title = "rotulagem"
    ws["A1"] = "Marque `relevante` com sim/nao: a review ajuda a responder a pergunta? (sem saber de qual variação veio)"
    ws["A1"].font = Font(bold=True)
    ws.append([])
    ws.append(["id", "pergunta", "review_id", "texto", "relevante"])
    for c in ws[3]:
        c.font = Font(bold=True)
    for pid, g in faltando.groupby("id", sort=True):
        linhas = g.to_dict("records")
        rng.shuffle(linhas)
        for r in linhas:
            ws.append([pid, texto_perg.get(pid, ""), r["review_id"], r["texto"], ""])
    for col, larg in zip("ABCDE", (8, 45, 36, 80, 12)):
        ws.column_dimensions[col].width = larg
    for row in ws.iter_rows(min_row=4, min_col=4, max_col=4):
        row[0].alignment = Alignment(wrap_text=True, vertical="top")
    wb.save(ROTULAGEM_PATH)


def importar_rotulos() -> None:
    """Copia os rótulos preenchidos em t28_rotulagem.xlsx para t28_rotulos.csv (versionado)."""
    from openpyxl import load_workbook

    from eval.rotulos_io import norm_relevante

    if not ROTULAGEM_PATH.exists():
        raise SystemExit(f"Não encontrei {ROTULAGEM_PATH}. Rode primeiro: python -m eval.avaliacao_final")
    ws = load_workbook(ROTULAGEM_PATH, read_only=True, data_only=True)["rotulagem"]
    novos, vazios = [], 0
    for row in ws.iter_rows(min_row=4, values_only=True):
        if not row or not row[0] or not row[2]:
            continue
        rel = norm_relevante(row[4] if len(row) > 4 else None)
        if rel is None:
            vazios += 1
            continue
        novos.append({"id": str(row[0]), "review_id": str(row[2]), "relevante": rel})
    antigos = pd.read_csv(ROTULOS_CSV, dtype=str) if ROTULOS_CSV.exists() else pd.DataFrame(columns=["id", "review_id", "relevante"])
    todos = (pd.concat([antigos, pd.DataFrame(novos)], ignore_index=True)
             .drop_duplicates(["id", "review_id"], keep="last").sort_values(["id", "review_id"]))
    todos.to_csv(ROTULOS_CSV, index=False)
    print(f"[t28] {len(novos)} rótulos importados ({vazios} linhas ainda vazias); total em {ROTULOS_CSV.name}: {len(todos)}")


# ---------------------------------------------------------------------------
# Cálculo
# ---------------------------------------------------------------------------
def avaliar_busca(pools: pd.DataFrame, perguntas: pd.DataFrame, rotulos: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Métricas por pergunta e variação + itens sem rótulo (para a planilha cega)."""
    responder = perguntas[(perguntas["esperado"] == "responder") & (perguntas["tipo_t11"] != "fora_de_escopo")]
    linhas, faltando = [], []
    for p in responder.itertuples(index=False):
        rot = {rid: v for (pid, rid), v in rotulos.items() if pid == p.id}
        for com_filtros in (True, False):
            pool = pools[(pools["id"] == p.id) & (pools["com_filtros"] == com_filtros)]
            if pool.empty:
                continue
            rankings = rankings_do_pool(pool, LIMIAR_RELEVANCIA)
            if not com_filtros:
                rankings = {"sem filtros": rankings["hibrida+reranker"]}
            textos = dict(zip(pool["review_id"].astype(str), pool["texto"]))
            for variacao, ids in rankings.items():
                m = metricas_ranking(ids, rot, K, por_entregues=(variacao == "final"))
                linhas.append({"id": p.id, "tipo": p.tipo_csv, "com_filtros_t11": bool(p.filtros),
                               "variacao": variacao, **m})
                faltando += [{"id": p.id, "review_id": r, "texto": textos.get(r, "")}
                             for r in ids if r not in rot]
    falt = pd.DataFrame(faltando).drop_duplicates(["id", "review_id"]) if faltando else pd.DataFrame(
        columns=["id", "review_id", "texto"])
    return pd.DataFrame(linhas), falt


def avaliar_recusa(pools: pd.DataFrame, perguntas: pd.DataFrame) -> pd.DataFrame:
    """% de recusas corretas por política, com a função real dos guardrails (T16)."""
    from src import guardrails
    from src.schema import ConsultaProcessada, Evidencia

    original = guardrails.LIMIAR_RELEVANCIA
    politicas = [("sem guardrail", None)] + [(f"limiar {x:.1f}", x) for x in LIMIARES]
    linhas = []
    try:
        for p in perguntas.itertuples(index=False):
            consulta = ConsultaProcessada(p.pergunta, p.consulta, {}, p.tipo_t11)
            pool = pools[(pools["id"] == p.id) & (pools["com_filtros"])].dropna(subset=["score_ce"])
            top = pool.sort_values("score_ce", ascending=False).head(K)
            evid = [Evidencia(str(r.review_id), r.texto, float(r.score_ce)) for r in top.itertuples(index=False)]
            for nome, limiar in politicas:
                if limiar is None:
                    recusou = p.tipo_t11 == "fora_de_escopo"
                    motivo = "fora_de_escopo" if recusou else "ok"
                else:
                    guardrails.LIMIAR_RELEVANCIA = limiar
                    v = guardrails.avaliar_evidencias(consulta, evid)
                    recusou, motivo = not v.pode_responder, v.motivo
                linhas.append({"id": p.id, "tipo": p.tipo_csv, "esperado": p.esperado, "politica": nome,
                               "recusou": recusou, "motivo": motivo,
                               "correta": recusou == (p.esperado == "recusar")})
    finally:
        guardrails.LIMIAR_RELEVANCIA = original
    return pd.DataFrame(linhas)


def resumir_busca(por_pergunta: pd.DataFrame) -> pd.DataFrame:
    tab = []
    base = por_pergunta[por_pergunta["variacao"] == REFERENCIA].set_index("id")
    for variacao in VARIACOES + ["sem filtros"]:
        sub = por_pergunta[por_pergunta["variacao"] == variacao].set_index("id")
        if sub.empty:
            continue
        comum = sub.dropna(subset=[f"p@{K}"]).index.intersection(base.index)
        if variacao == REFERENCIA:
            dif = "referência"
        else:
            media, lo, hi = bootstrap_diferenca(sub.loc[comum, f"p@{K}"], base.loc[comum, f"p@{K}"])
            dif = f"{media:+.3f} [{lo:+.3f}; {hi:+.3f}]"
            if variacao == "sem filtros":
                dif += f" (atual nessas {len(comum)}: {base.loc[comum, f'p@{K}'].mean():.3f})"
        tab.append({
            "variação": variacao, "perguntas": len(sub),
            "P@5": sub["p@5"].mean(), f"P@{K}": sub[f"p@{K}"].mean(), f"MRR@{K}": sub[f"rr@{K}"].mean(),
            "evidências médias": sub["n_itens"].mean(), "sem rótulo": int(sub["nao_rotulados"].sum()),
            f"dif. P@{K} vs atual (IC95%)": dif,
        })
    return pd.DataFrame(tab)


def resumir_recusa(recusa: pd.DataFrame) -> pd.DataFrame:
    def taxa(g, esperado):
        s = g[g["esperado"] == esperado]
        return (s["recusou"] == (esperado == "recusar")).mean() if len(s) else float("nan")

    ordem = {nome: i for i, nome in enumerate(recusa["politica"].unique())}
    out = (recusa.groupby("politica")
           .apply(lambda g: pd.Series({
               "recusa correta (todas)": g["correta"].mean(),
               "recusou quando devia": taxa(g, "recusar"),
               "respondeu quando devia": taxa(g, "responder"),
               "erros": int((~g["correta"]).sum()),
               "ids com erro": ", ".join(g.loc[~g["correta"], "id"]),
           }), include_groups=False)
           .reset_index())
    return out.sort_values("politica", key=lambda s: s.map(ordem)).reset_index(drop=True)


def escrever_resumo(rotulo: str, busca: pd.DataFrame, recusa: pd.DataFrame, perguntas: pd.DataFrame,
                    n_faltando: int, geracao: pd.DataFrame | None) -> str:
    def fmt(x, casas=3):
        return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{casas}f}"

    tempos = perguntas.dropna(subset=["busca_completa_s"]) if "busca_completa_s" in perguntas else pd.DataFrame()
    linhas = [
        f"# T28 — Avaliação final ({rotulo})",
        "",
        f"Configuração: embedding `{EMBEDDING_MODEL}`, reranker `{RERANKER_MODEL}`, "
        f"T11 `{LLM_PROVIDER}`, pool {TOP_K_CANDIDATOS}, top-{K}, limiar {LIMIAR_RELEVANCIA}.",
        f"Perguntas: {len(perguntas)} ({(perguntas['esperado'] == 'responder').sum()} para responder, "
        f"{(perguntas['esperado'] == 'recusar').sum()} para recusar).",
        "",
        "## Busca (perguntas que devem ser respondidas)",
        "",
        f"| Variação | Perguntas | P@5 | P@{K} | MRR@{K} | Evidências | Sem rótulo | Diferença P@{K} vs atual (IC95%) |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in busca.to_dict("records"):
        linhas.append(f"| {r['variação']} | {r['perguntas']} | {fmt(r['P@5'])} | {fmt(r[f'P@{K}'])} | "
                      f"{fmt(r[f'MRR@{K}'])} | {fmt(r['evidências médias'], 1)} | {r['sem rótulo']} | "
                      f"{r[f'dif. P@{K} vs atual (IC95%)']} |")
    linhas += [
        "",
        "`final` = top-10 do reranker só com score ≥ limiar (o que chega ao LLM); a precisão é medida sobre "
        "as evidências entregues, e a coluna Evidências mostra quantas são em média. `sem filtros` só inclui "
        "perguntas em que a T11 extraiu filtros, comparadas ao atual nessas mesmas perguntas.",
    ]
    if n_faltando:
        linhas += ["", f"**Atenção:** {n_faltando} reviews ainda sem rótulo (contadas como não relevantes). "
                   f"Preencha `eval/t28_rotulagem.xlsx` e rode `--importar-rotulos`."]
    linhas += ["", "## Recusa", "",
               "| Política | Recusa correta | Recusou quando devia | Respondeu quando devia | Erros | Perguntas com erro |",
               "|---|---:|---:|---:|---:|---|"]
    for r in recusa.to_dict("records"):
        linhas.append(f"| {r['politica']} | {fmt(r['recusa correta (todas)'], 2)} | {fmt(r['recusou quando devia'], 2)} | "
                      f"{fmt(r['respondeu quando devia'], 2)} | {r['erros']} | {r['ids com erro'] or '—'} |")
    if not tempos.empty:
        linhas += ["", "## Custo da busca (segundos por pergunta, média)", "",
                   f"vetorial {tempos['vetorial_s'].mean():.3f} · BM25 {tempos['bm25_s'].mean():.3f} · "
                   f"busca completa com reranker do pool de {TOP_K_CANDIDATOS} {tempos['busca_completa_s'].mean():.3f}"]
    if geracao is not None and not geracao.empty:
        resp = geracao[geracao["esperado"] == "responder"]
        linhas += ["", "## Geração (pipeline completo)", "",
                   f"- Respondidas entre as que deviam ser respondidas: {fmt((~resp['recusou']).mean(), 2)}",
                   f"- Citações por resposta (média): {fmt(resp.loc[~resp['recusou'], 'n_citacoes'].mean(), 1)}",
                   f"- Tempo por pergunta (média): {fmt(geracao['tempo_s'].mean(), 1)} s",
                   f"- Fidelidade: julgar em `eval/t28_fidelidade.xlsx` (sim/parcial/nao por resposta)."]
    texto = "\n".join(linhas) + "\n"
    caminho = RESUMO_PATH if rotulo == "atual" else PASTA / f"t28_resumo_{rotulo}.md"
    caminho.write_text(texto, encoding="utf-8")
    return texto


# ---------------------------------------------------------------------------
# Geração (opcional)
# ---------------------------------------------------------------------------
def avaliar_geracao(perguntas: pd.DataFrame) -> pd.DataFrame:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font

    from src.pipeline import perguntar

    linhas = []
    for p in perguntas.itertuples(index=False):
        t = time.perf_counter()
        r = perguntar(p.pergunta)
        linhas.append({"id": p.id, "esperado": p.esperado, "recusou": r.recusada, "motivo": r.motivo_recusa,
                       "n_citacoes": len(r.citacoes), "tempo_s": time.perf_counter() - t, "resposta": r.texto,
                       "evidencias_citadas": "\n".join(f"[{e.review_id}] {e.texto}" for e in r.evidencias
                                                       if e.review_id in set(r.citacoes))})
        print(f"[t28] geração {p.id}: {'recusou' if r.recusada else f'{len(r.citacoes)} citações'}")
    df = pd.DataFrame(linhas)

    wb = Workbook()
    ws = wb.active
    ws.title = "fidelidade"
    ws.append(["id", "resposta", "evidências citadas", "fiel (sim/parcial/nao)", "observação"])
    for c in ws[1]:
        c.font = Font(bold=True)
    for r in df[~df["recusou"]].itertuples(index=False):
        ws.append([r.id, r.resposta, r.evidencias_citadas, "", ""])
    for col, larg in zip("ABCDE", (8, 70, 70, 14, 30)):
        ws.column_dimensions[col].width = larg
    for row in ws.iter_rows(min_row=2, min_col=2, max_col=3):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    wb.save(FIDELIDADE_PATH)
    return df


# ---------------------------------------------------------------------------
def rodar(rotulo: str, refazer: bool, com_geracao: bool) -> None:
    pools, perguntas = coletar(rotulo, refazer)
    rotulos = carregar_rotulos()
    por_pergunta, faltando = avaliar_busca(pools, perguntas, rotulos)
    por_pergunta.to_csv(POR_PERGUNTA_PATH if rotulo == "atual" else PASTA / f"t28_por_pergunta_{rotulo}.csv",
                        index=False)
    recusa = avaliar_recusa(pools, perguntas)
    recusa.to_csv(RECUSA_PATH if rotulo == "atual" else PASTA / f"t28_recusa_{rotulo}.csv", index=False)

    if not faltando.empty:
        escrever_planilha_cega(faltando, perguntas)
        print(f"[t28] {len(faltando)} reviews sem rótulo → {ROTULAGEM_PATH.name}")
    geracao = avaliar_geracao(perguntas) if com_geracao else None
    print()
    print(escrever_resumo(rotulo, resumir_busca(por_pergunta), resumir_recusa(recusa), perguntas,
                          len(faltando), geracao))


def main() -> None:
    parser = argparse.ArgumentParser(description="T28: avaliação final e comparação de variações")
    parser.add_argument("--rotulo", default="atual", help="nome desta execução (ex.: bge-m3)")
    parser.add_argument("--refazer-busca", action="store_true", help="ignora o cache e busca de novo")
    parser.add_argument("--importar-rotulos", action="store_true",
                        help="copia os rótulos de t28_rotulagem.xlsx para t28_rotulos.csv")
    parser.add_argument("--com-geracao", action="store_true", help="gera as respostas (Gemini ou Ollama)")
    args = parser.parse_args()
    if args.importar_rotulos:
        importar_rotulos()
    else:
        rodar(args.rotulo, args.refazer_busca, args.com_geracao)


if __name__ == "__main__":
    main()

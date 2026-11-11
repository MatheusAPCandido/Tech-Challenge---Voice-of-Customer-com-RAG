"""T07 — comparação de modelos de embedding para a busca vetorial (Pessoa A).

Uso:
    python -m eval.comparar_embeddings                 # compara os 3 modelos padrão
    python -m eval.comparar_embeddings --modelos intfloat/multilingual-e5-base BAAI/bge-m3
    python -m eval.comparar_embeddings --manual        # calcula P@5 da rotulagem manual

Pré-requisitos: python -m src.prepare (gera data/reviews_docs.parquet e data/gold/reviews_completo.parquet).
Não usa o ChromaDB: indexa a base inteira em memória para cada modelo, então não altera o índice da equipe.

Como a comparação é feita
-------------------------
Só a busca vetorial é comparada (sem BM25, filtros nem reranker), para isolar o efeito do modelo.

1. Avaliação automática (consultas `avaliacao=tema` em eval/consultas_t07.csv, 55 consultas):
   cada consulta tem os temas que uma resposta relevante deve citar. Uma review do top-k é
   relevante se a sua lista de temas (coluna `temas`, T08) contém algum deles. Métricas:
   precisão@5, precisão@10 e MRR@10. A diferença para o modelo de referência vem com
   intervalo de confiança de 95% por bootstrap pareado sobre as consultas.
   Limitação: os temas vêm de regras com ~80% de acerto, então os números absolutos são
   aproximados; a comparação entre modelos usa os mesmos rótulos para todos.

2. Avaliação manual (consultas `avaliacao=manual`, assuntos de nicho sem tema próprio):
   junta o top-5 de cada modelo numa planilha cega e embaralhada (eval/t07_rotulagem.xlsx).
   Depois de preencher `relevante` com sim/nao, `--manual` calcula a precisão@5 de cada modelo.

3. Custo: tempo para gerar os embeddings da base, consultas por segundo e memória do índice.

Saídas (em eval/):
    t07_resultados.csv   métricas por modelo e consulta
    t07_resumo.md        tabela final para o README
    t07_rotulagem.xlsx   planilha cega para a rotulagem manual
    t07_pool.csv         de qual modelo e posição veio cada linha da planilha
Cache dos embeddings: data/t07_cache/ (fora do Git; apague para recalcular).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_DIR, DOCS_PATH

PASTA = Path(__file__).resolve().parent
CONSULTAS_PATH = PASTA / "consultas_t07.csv"
COMPLETO_PATH = DATA_DIR / "gold" / "reviews_completo.parquet"
CACHE_DIR = DATA_DIR / "t07_cache"
RESULTADOS_PATH = PASTA / "t07_resultados.csv"
RESUMO_PATH = PASTA / "t07_resumo.md"
ROTULAGEM_PATH = PASTA / "t07_rotulagem.xlsx"
POOL_PATH = PASTA / "t07_pool.csv"

MODELOS_PADRAO = [
    "intfloat/multilingual-e5-base",    # referência: o modelo usado hoje no índice
    "intfloat/multilingual-e5-small",   # mais leve
    "BAAI/bge-m3",                      # maior, mesma família do reranker da equipe
]
REFERENCIA = "intfloat/multilingual-e5-base"
K_MAX = 10
K_POOL_MANUAL = 5
MAX_SEQ = 256          # reviews são curtas (mediana ~50 caracteres)
LOTE = 64
N_BOOTSTRAP = 10_000
SEED = 42


# ---------------------------------------------------------------------------
# Métricas (funções puras, testadas em tests/test_comparar_embeddings.py)
# ---------------------------------------------------------------------------
def precisao_at_k(relevantes: list[bool], k: int) -> float:
    """Fração de relevantes entre os k primeiros (lista menor que k conta como não relevante)."""
    return sum(relevantes[:k]) / k


def rr_at_k(relevantes: list[bool], k: int) -> float:
    """Recíproco da posição do primeiro relevante no top-k (0 se não houver)."""
    for i, rel in enumerate(relevantes[:k], start=1):
        if rel:
            return 1.0 / i
    return 0.0


def eh_relevante(temas_doc: set[str], temas_esperados: set[str]) -> bool:
    return bool(temas_doc & temas_esperados)


def bootstrap_diferenca(a: np.ndarray, b: np.ndarray, n: int = N_BOOTSTRAP, seed: int = SEED):
    """Média de (a - b) e IC95% por bootstrap pareado sobre as consultas."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    medias = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
def prefixos(nome: str) -> tuple[str, str]:
    """(prefixo da consulta, prefixo do documento). E5 exige 'query: ' e 'passage: '."""
    if "e5" in nome.lower():
        return "query: ", "passage: "
    return "", ""


def slug(nome: str) -> str:
    return nome.replace("/", "__")


def carregar_modelo(nome: str):
    from sentence_transformers import SentenceTransformer

    modelo = SentenceTransformer(nome)
    modelo.max_seq_length = MAX_SEQ
    if str(modelo.device).startswith("cuda"):
        modelo.half()  # metade da memória e mais rápido; diferença desprezível no ranking
    print(f"[t07] {nome} carregado em {modelo.device}")
    return modelo


def codificar(modelo, textos: list[str], barra: bool = False) -> np.ndarray:
    vetores = modelo.encode(
        textos, batch_size=LOTE, normalize_embeddings=True,
        show_progress_bar=barra, convert_to_numpy=True,
    )
    return vetores.astype(np.float32)


def embeddings_da_base(nome: str, modelo, ids: list[str], textos: list[str]) -> tuple[np.ndarray, float]:
    """Embeddings da base com cache em disco. Devolve (matriz, segundos gastos na geração)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    arq, meta_arq = CACHE_DIR / f"{slug(nome)}.npy", CACHE_DIR / f"{slug(nome)}.json"
    assinatura = {"n": len(ids), "primeiro": ids[0], "ultimo": ids[-1], "max_seq": MAX_SEQ}
    if arq.exists() and meta_arq.exists():
        meta = json.loads(meta_arq.read_text(encoding="utf-8"))
        if meta.get("assinatura") == assinatura:
            print(f"[t07] usando cache de {nome}")
            return np.load(arq), float(meta["segundos"])
    _, prefixo_doc = prefixos(nome)
    inicio = time.perf_counter()
    matriz = codificar(modelo, [prefixo_doc + t for t in textos], barra=True)
    segundos = time.perf_counter() - inicio
    np.save(arq, matriz)
    meta_arq.write_text(json.dumps({"assinatura": assinatura, "segundos": segundos}), encoding="utf-8")
    return matriz, segundos


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
def carregar_base() -> pd.DataFrame:
    """Documentos do índice + todos os temas citados em cada review (T08)."""
    for caminho in (DOCS_PATH, COMPLETO_PATH):
        if not caminho.exists():
            raise SystemExit(f"Arquivo não encontrado: {caminho}. Rode python -m src.prepare antes.")
    docs = pd.read_parquet(DOCS_PATH, columns=["review_id", "texto"])
    temas = pd.read_parquet(COMPLETO_PATH, columns=["review_id", "temas"]).drop_duplicates("review_id")
    base = docs.merge(temas, on="review_id", how="left")
    base["temas"] = [set(str(t).split("|")) - {""} if isinstance(t, str) else set() for t in base["temas"]]
    return base.reset_index(drop=True)


def carregar_consultas() -> pd.DataFrame:
    c = pd.read_csv(CONSULTAS_PATH, dtype=str).fillna("")
    c["temas_relevantes"] = [set(t.split("|")) - {""} for t in c["temas_relevantes"]]
    sem_tema = c[(c["avaliacao"] == "tema") & (c["temas_relevantes"].str.len() == 0)]
    if not sem_tema.empty:
        raise SystemExit(f"Consultas de tema sem temas_relevantes: {', '.join(sem_tema['id'])}")
    return c


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------
def avaliar_modelo(nome: str, base: pd.DataFrame, consultas: pd.DataFrame):
    modelo = carregar_modelo(nome)
    matriz, seg_base = embeddings_da_base(nome, modelo, base["review_id"].tolist(), base["texto"].tolist())

    prefixo_q, _ = prefixos(nome)
    codificar(modelo, [prefixo_q + "aquecimento"] * LOTE)  # aquece a GPU para não distorcer a medida
    inicio = time.perf_counter()
    vetores_q = codificar(modelo, [prefixo_q + q for q in consultas["consulta"]])
    sims = vetores_q @ matriz.T
    top = np.argsort(-sims, axis=1)[:, :max(K_MAX, K_POOL_MANUAL)]
    seg_consultas = time.perf_counter() - inicio

    linhas, pool = [], []
    for i, c in enumerate(consultas.itertuples(index=False)):
        idx = top[i]
        if c.avaliacao == "tema":
            rel = [eh_relevante(base.at[j, "temas"], c.temas_relevantes) for j in idx[:K_MAX]]
            linhas.append({
                "modelo": nome, "id": c.id, "origem": c.origem, "consulta": c.consulta,
                "p@5": precisao_at_k(rel, 5), "p@10": precisao_at_k(rel, 10), "rr@10": rr_at_k(rel, 10),
                "sim_top1": round(float(sims[i, idx[0]]), 4),
            })
        else:
            for pos, j in enumerate(idx[:K_POOL_MANUAL], start=1):
                pool.append({"id": c.id, "consulta": c.consulta, "modelo": nome, "rank": pos,
                             "review_id": base.at[j, "review_id"], "texto": base.at[j, "texto"]})

    custo = {
        "modelo": nome,
        "dimensao": int(matriz.shape[1]),
        "seg_indexar_base": round(seg_base, 1),
        "consultas_por_seg": round(len(consultas) / seg_consultas, 1),
        "mb_indice": round(matriz.nbytes / 1e6, 1),
    }

    import gc
    del modelo
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return linhas, pool, custo


def resumir(res: pd.DataFrame, custos: pd.DataFrame, modelos: list[str]) -> pd.DataFrame:
    ref = REFERENCIA if REFERENCIA in modelos else modelos[0]
    agg = res.groupby("modelo")[["p@5", "p@10", "rr@10"]].mean().rename(columns={"rr@10": "mrr@10"})
    tab = custos.set_index("modelo").join(agg).reindex(modelos)
    base_ref = res[res["modelo"] == ref].set_index("id")["p@10"]
    difs = []
    for m in modelos:
        outro = res[res["modelo"] == m].set_index("id")["p@10"].reindex(base_ref.index)
        if m == ref:
            difs.append("referência")
        else:
            media, lo, hi = bootstrap_diferenca(outro.values, base_ref.values)
            difs.append(f"{media:+.3f} [{lo:+.3f}; {hi:+.3f}]")
    tab["dif_p@10_vs_ref (IC95)"] = difs
    return tab.reset_index()


def escrever_resumo(tab: pd.DataFrame, res: pd.DataFrame) -> None:
    n = res["id"].nunique()
    cab = ["Modelo", "Dim.", "Indexar base (s)", "Consultas/s", "Índice (MB)",
           "P@5", "P@10", "MRR@10", "Diferença P@10 vs referência (IC95%)"]
    linhas = ["| " + " | ".join(cab) + " |", "|" + "---|" * len(cab)]
    for r in tab.to_dict("records"):
        linhas.append(
            f"| `{r['modelo']}` | {r['dimensao']} | {r['seg_indexar_base']} | {r['consultas_por_seg']} "
            f"| {r['mb_indice']} | {r['p@5']:.3f} | {r['p@10']:.3f} | {r['mrr@10']:.3f} "
            f"| {r['dif_p@10_vs_ref (IC95)']} |"
        )
    texto = (
        "# T07 — Comparação de modelos de embedding\n\n"
        f"Busca só vetorial, base completa ({n} consultas com rótulo de tema, top-10).\n\n"
        + "\n".join(linhas)
        + "\n\nRelevância: a review cita algum dos temas esperados para a consulta (rótulos da T08).\n"
        "IC95%: bootstrap pareado sobre as consultas; se o intervalo inclui 0, a diferença não é conclusiva.\n"
    )
    RESUMO_PATH.write_text(texto, encoding="utf-8")


def escrever_planilha_cega(pool: pd.DataFrame) -> None:
    """Um item por (consulta, review), sem dizer de qual modelo veio, em ordem aleatória."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font

    unicos = pool.drop_duplicates(["id", "review_id"])[["id", "consulta", "review_id", "texto"]]
    rng = random.Random(SEED)
    wb = Workbook()
    ws = wb.active
    ws.title = "rotulagem"
    ws["A1"] = "Marque `relevante` com sim/nao: a review fala do assunto da consulta? Não diga de qual modelo veio cada linha."
    ws["A1"].font = Font(bold=True)
    cab = ["id", "consulta", "review_id", "texto", "relevante"]
    ws.append([])
    ws.append(cab)
    for c in ws[3]:
        c.font = Font(bold=True)
    for _, g in unicos.groupby("id", sort=True):
        linhas = g.to_dict("records")
        rng.shuffle(linhas)
        for r in linhas:
            ws.append([r["id"], r["consulta"], r["review_id"], r["texto"], ""])
    for col, larg in zip("ABCDE", (8, 40, 36, 80, 12)):
        ws.column_dimensions[col].width = larg
    for row in ws.iter_rows(min_row=4, min_col=4, max_col=4):
        row[0].alignment = Alignment(wrap_text=True, vertical="top")
    wb.save(ROTULAGEM_PATH)


def rodar(modelos: list[str]) -> None:
    base = carregar_base()
    consultas = carregar_consultas()
    print(f"[t07] base: {len(base)} documentos | consultas: {len(consultas)} "
          f"({(consultas['avaliacao'] == 'tema').sum()} automáticas, {(consultas['avaliacao'] == 'manual').sum()} manuais)")

    todas, pool, custos = [], [], []
    for nome in modelos:
        linhas, p, custo = avaliar_modelo(nome, base, consultas)
        todas += linhas
        pool += p
        custos.append(custo)
        m = pd.DataFrame(linhas)
        print(f"[t07] {nome}: P@10={m['p@10'].mean():.3f}  MRR@10={m['rr@10'].mean():.3f}  "
              f"indexação={custo['seg_indexar_base']} s")

    res = pd.DataFrame(todas)
    res.to_csv(RESULTADOS_PATH, index=False)
    tab = resumir(res, pd.DataFrame(custos), modelos)
    escrever_resumo(tab, res)
    if pool:
        pool_df = pd.DataFrame(pool)
        pool_df.to_csv(POOL_PATH, index=False)
        if ROTULAGEM_PATH.exists():
            # não apaga rótulos já preenchidos; apague o arquivo para gerar uma planilha nova
            print(f"[t07] planilha já existe, mantida: {ROTULAGEM_PATH}")
        else:
            escrever_planilha_cega(pool_df)
            print(f"[t07] planilha para rotulagem manual: {ROTULAGEM_PATH} "
                  f"({pool_df.drop_duplicates(['id', 'review_id']).shape[0]} linhas)")
    print()
    print(RESUMO_PATH.read_text(encoding="utf-8"))


def avaliar_manual() -> None:
    """Precisão@5 de cada modelo nas consultas manuais, a partir da planilha preenchida."""
    from openpyxl import load_workbook

    from eval.rotulos_io import norm_relevante

    if not ROTULAGEM_PATH.exists() or not POOL_PATH.exists():
        raise SystemExit("Rode primeiro: python -m eval.comparar_embeddings")
    ws = load_workbook(ROTULAGEM_PATH, read_only=True, data_only=True)["rotulagem"]
    rotulos = {}
    for row in ws.iter_rows(min_row=4, values_only=True):
        if row and row[0] and row[2]:
            rel = norm_relevante(row[4] if len(row) > 4 else None)
            if rel is not None:
                rotulos[(str(row[0]), str(row[2]))] = rel == "sim"
    pool = pd.read_csv(POOL_PATH, dtype={"id": str, "review_id": str})
    pool["relevante"] = [rotulos.get((i, r)) for i, r in zip(pool["id"], pool["review_id"])]
    faltando = pool["relevante"].isna().sum()
    if faltando:
        print(f"[t07] atenção: {faltando} linhas do pool sem rótulo (ficam de fora)")
    pool = pool.dropna(subset=["relevante"])
    pool["relevante"] = pool["relevante"].astype(bool)
    por_consulta = pool.groupby(["modelo", "id"])["relevante"].mean().unstack("modelo")
    print("Precisão@5 por consulta (rotulagem manual):")
    print(por_consulta.round(2).to_string())
    print("\nMédia por modelo:")
    print(por_consulta.mean().round(3).to_string())
    por_consulta.to_csv(PASTA / "t07_manual.csv")

    ref = REFERENCIA if REFERENCIA in por_consulta.columns else por_consulta.columns[0]
    linhas = ["| Modelo | P@5 manual | Diferença vs referência (IC95%) |", "|---|---|---|"]
    for m in por_consulta.columns:
        if m == ref:
            dif = "referência"
        else:
            media, lo, hi = bootstrap_diferenca(por_consulta[m].values, por_consulta[ref].values)
            dif = f"{media:+.3f} [{lo:+.3f}; {hi:+.3f}]"
        linhas.append(f"| `{m}` | {por_consulta[m].mean():.3f} | {dif} |")
    bloco = (
        "\n## Rotulagem manual (assuntos sem tema próprio)\n\n"
        f"Top-5 de cada modelo em {len(por_consulta)} consultas, rotulado às cegas (sim/nao).\n\n"
        + "\n".join(linhas) + "\n"
    )
    texto = RESUMO_PATH.read_text(encoding="utf-8") if RESUMO_PATH.exists() else ""
    texto = texto.split("\n## Rotulagem manual")[0].rstrip("\n") + "\n" + bloco
    RESUMO_PATH.write_text(texto, encoding="utf-8")
    print(bloco)


def main() -> None:
    parser = argparse.ArgumentParser(description="T07: comparação de modelos de embedding")
    parser.add_argument("--modelos", nargs="+", default=MODELOS_PADRAO)
    parser.add_argument("--manual", action="store_true", help="calcula P@5 da rotulagem manual")
    args = parser.parse_args()
    if args.manual:
        avaliar_manual()
    else:
        rodar(args.modelos)


if __name__ == "__main__":
    main()

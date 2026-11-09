"""GOLD: documentos prontos para o RAG, no formato de src/schema.COLUNAS_DOCUMENTO.

Regras (na ordem em que são aplicadas, todas contadas em data/gold/_qualidade.json):
  1. texto = título + mensagem limpos ("Título. Mensagem", ou só o que existir)
  2. remove reviews sem texto
  3. remove textos sem conteúdo: menos de MIN_LETRAS letras (ex.: ".", "ok", "10", emojis)
  4. orders → data_compra (AAAA-MM), dias_atraso, entregue
  5. items + products → categoria (em português) do item de MAIOR PREÇO do pedido
  6. customers → estado (UF do cliente)
  7. deduplica: um documento por review_id e um por texto idêntico (ignorando caixa, acentos e pontuação),
     mantendo a avaliação mais antiga
  8. tema = assunto principal da review (regras em src/medallion/temas.py, T08)

Saídas extras em data/gold/:
  reviews_completo.parquet  todas as reviews com conteúdo (antes da etapa 7), com colunas extras,
                            qtd_textos_identicos e `temas` (todos os temas citados, "a|b"),
                            para análises e respostas agregadas
  resumo_categorias.parquet nota média, % negativas e % de atraso por categoria
  resumo_temas.parquet      por tema: quantas reviews têm o tema como principal, quantas o citam,
                            nota média, % negativas e % de entregas atrasadas

Uso isolado:  python -m src.medallion.gold
"""

from __future__ import annotations

import json
import unicodedata

import pandas as pd

from src.medallion import GOLD_DIR, SILVER_DIR, garantir_pastas
from src.medallion.temas import TEMAS, tema_principal, temas_da_review
from src.schema import COLUNAS_DOCUMENTO

MIN_LETRAS = 3


def _ler(nome: str) -> pd.DataFrame:
    return pd.read_parquet(SILVER_DIR / f"{nome}.parquet")


def _montar_texto(r: pd.DataFrame) -> pd.Series:
    titulo, msg = r["titulo_limpo"], r["mensagem_limpa"]
    iguais = (titulo.str.casefold() == msg.str.casefold()).fillna(False).astype(bool)
    titulo = titulo.mask(iguais)  # evita "Ótimo. Ótimo"
    juntos = titulo + ". " + msg
    return juntos.fillna(msg).fillna(titulo)


def _qtd_letras(texto: pd.Series) -> pd.Series:
    return texto.str.count(r"[A-Za-zÀ-ÖØ-öø-ÿ]")


def _sem_acento(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c))


def _texto_normalizado(texto: pd.Series) -> pd.Series:
    """Chave só para comparar duplicatas: ignora caixa, acentos e pontuação."""
    return (
        texto.str.casefold()
        .map(_sem_acento)
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _por_pedido(items: pd.DataFrame, products: pd.DataFrame, sellers: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por pedido: categoria do item mais caro + lista de todas as categorias."""
    itens = items.merge(products, on="product_id", how="left")
    itens["categoria"] = itens["categoria"].fillna("sem_categoria")
    itens["categoria_en"] = itens["categoria_en"].fillna("sem_categoria")

    principal = (
        itens.sort_values(["order_id", "price"], ascending=[True, False])
        .drop_duplicates("order_id")[["order_id", "categoria", "categoria_en", "seller_id"]]
        .merge(sellers[["seller_id", "seller_state"]], on="seller_id", how="left")
        .rename(columns={"seller_state": "estado_seller"})
        .drop(columns="seller_id")
    )
    extras = itens.groupby("order_id").agg(
        todas_categorias=("categoria", lambda s: "|".join(sorted(set(s)))),
        n_itens=("product_id", "size"),
        valor_pedido=("price", "sum"),
    ).reset_index()
    extras["n_itens"] = extras["n_itens"].astype("Int64")
    extras["valor_pedido"] = extras["valor_pedido"].round(2)
    return principal.merge(extras, on="order_id", how="left")


def executar() -> pd.DataFrame:
    garantir_pastas()
    reviews, orders = _ler("reviews"), _ler("orders")
    items, products = _ler("order_items"), _ler("products")
    customers, sellers = _ler("customers"), _ler("sellers")
    q: dict = {"reviews_silver": len(reviews)}

    # 1-2. texto
    r = reviews.copy()
    r["texto"] = _montar_texto(r)
    r = r[r["texto"].notna()]
    q["removidas_sem_texto"] = q["reviews_silver"] - len(r)

    # 3. sem conteúdo
    sem_conteudo = _qtd_letras(r["texto"]) < MIN_LETRAS
    q["removidas_sem_conteudo"] = int(sem_conteudo.sum())
    q["exemplos_sem_conteudo"] = r.loc[sem_conteudo, "texto"].value_counts().head(10).to_dict()
    r = r[~sem_conteudo]

    # 4-6. metadados
    antes = len(r)
    d = (
        r.merge(orders, on="order_id", how="inner")
        .merge(_por_pedido(items, products, sellers), on="order_id", how="left")
        .merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left")
    )
    q["removidas_sem_pedido"] = antes - len(d)

    entrega = d["order_delivered_customer_date"].dt.normalize()
    prevista = d["order_estimated_delivery_date"].dt.normalize()
    d["dias_atraso"] = (entrega - prevista).dt.days.astype("float64")   # NaN = não entregue
    d["entregue"] = d["order_delivered_customer_date"].notna()
    d["data_compra"] = d["order_purchase_timestamp"].dt.strftime("%Y-%m")
    d["categoria"] = d["categoria"].fillna("sem_categoria")
    d["categoria_en"] = d["categoria_en"].fillna("sem_categoria")
    d["estado"] = d["customer_state"]
    d["review_score"] = d["review_score"].astype("int64")
    lista_temas = d["texto"].map(temas_da_review)
    d["tema"] = lista_temas.map(tema_principal)
    d["temas"] = lista_temas.map("|".join)
    d["sentimento"] = pd.cut(d["review_score"], bins=[0, 2, 3, 5],
                             labels=["negativo", "neutro", "positivo"]).astype("string")
    d["tamanho_texto"] = d["texto"].str.len().astype("int64")

    # 7a. um documento por review_id (o mesmo review aparece ligado a mais de um pedido)
    d = d.sort_values(["review_creation_date", "review_id", "order_id"])
    antes = len(d)
    d = d.drop_duplicates("review_id")
    q["removidas_review_id_repetido"] = antes - len(d)

    # 7b. textos idênticos
    norm = _texto_normalizado(d["texto"])
    d["qtd_textos_identicos"] = norm.map(norm.value_counts()).astype("int64")
    d["no_indice"] = ~norm.duplicated()
    q["removidas_texto_identico"] = int((~d["no_indice"]).sum())

    extras = ["order_id", "sentimento", "tamanho_texto", "categoria_en", "todas_categorias",
              "n_itens", "valor_pedido", "estado_seller", "order_status",
              "temas", "qtd_textos_identicos", "no_indice"]
    completo = d[COLUNAS_DOCUMENTO + extras].reset_index(drop=True)
    docs = completo[completo["no_indice"]].reset_index(drop=True)

    # checagens do contrato
    assert docs["review_id"].is_unique, "review_id precisa ser único"
    assert docs["texto"].notna().all() and (docs["texto"] != "").all(), "documento sem texto"
    assert docs["review_score"].between(1, 5).all(), "nota fora de 1..5"
    assert docs["data_compra"].str.fullmatch(r"\d{4}-\d{2}").all(), "data_compra fora do formato AAAA-MM"
    assert docs["tema"].isin(TEMAS).all(), "tema fora da lista de src/medallion/temas.py"

    q["documentos_reviews_completo"] = len(completo)
    q["documentos_indice"] = len(docs)
    q["pct_sentimento"] = (docs["sentimento"].value_counts(normalize=True) * 100).round(1).to_dict()
    q["pct_entregue_atrasado"] = round(float((docs["dias_atraso"] > 0).sum() / docs["entregue"].sum()) * 100, 1)
    q["nao_entregues"] = int((~docs["entregue"]).sum())
    q["sem_categoria"] = int((docs["categoria"] == "sem_categoria").sum())
    q["mediana_tamanho_texto"] = int(docs["tamanho_texto"].median())
    q["pct_tema_indice"] = (docs["tema"].value_counts(normalize=True) * 100).round(1).to_dict()

    resumo = _resumo_categorias(completo)
    completo.to_parquet(GOLD_DIR / "reviews_completo.parquet", index=False)
    resumo.to_parquet(GOLD_DIR / "resumo_categorias.parquet", index=False)
    _resumo_temas(completo).to_parquet(GOLD_DIR / "resumo_temas.parquet", index=False)
    (GOLD_DIR / "_qualidade.json").write_text(json.dumps(q, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[gold] qualidade:", json.dumps(q, indent=2, ensure_ascii=False))
    return docs


def _resumo_categorias(completo: pd.DataFrame) -> pd.DataFrame:
    entregues = completo[completo["entregue"]]
    atraso = entregues.groupby("categoria").agg(
        pct_atraso=("dias_atraso", lambda s: round(float((s > 0).mean()) * 100, 1)),
        atraso_medio_dias=("dias_atraso", lambda s: round(float(s.mean()), 1)),
    )
    return (
        completo.assign(negativa=completo["sentimento"].eq("negativo"))
        .groupby("categoria")
        .agg(categoria_en=("categoria_en", "first"), n_reviews=("review_id", "size"),
             nota_media=("review_score", "mean"), pct_negativas=("negativa", "mean"))
        .assign(nota_media=lambda x: x["nota_media"].round(2),
                pct_negativas=lambda x: (x["pct_negativas"] * 100).round(1))
        .join(atraso)
        .sort_values("n_reviews", ascending=False)
        .reset_index()
    )


def _resumo_temas(completo: pd.DataFrame) -> pd.DataFrame:
    """Contagens por tema sobre TODAS as reviews com conteúdo (inclui textos repetidos),
    para responder perguntas agregadas ("quais os principais motivos de reclamação?")."""
    base = completo.assign(
        negativa=completo["sentimento"].eq("negativo"),
        atrasou=completo["dias_atraso"] > 0,
    )
    principal = base.groupby("tema").agg(
        n_principal=("review_id", "size"),
        nota_media=("review_score", "mean"),
        pct_negativas=("negativa", "mean"),
    )
    entregues = base[base["entregue"]].groupby("tema")["atrasou"].mean().rename("pct_entregas_atrasadas")
    citacoes = (
        base.assign(t=base["temas"].str.split("|")).explode("t")
        .query("t != ''").groupby("t").size().rename("n_citacoes")
    )
    total = len(base)
    out = principal.join(entregues).join(citacoes).reset_index().rename(columns={"index": "tema"})
    out["descricao"] = out["tema"].map(TEMAS)
    out["pct_principal"] = (out["n_principal"] / total * 100).round(1)
    out["n_citacoes"] = out["n_citacoes"].fillna(0).astype("int64")
    out["nota_media"] = out["nota_media"].round(2)
    out["pct_negativas"] = (out["pct_negativas"] * 100).round(1)
    out["pct_entregas_atrasadas"] = (out["pct_entregas_atrasadas"].astype("float") * 100).round(1)
    colunas = ["tema", "descricao", "n_principal", "pct_principal", "n_citacoes",
               "nota_media", "pct_negativas", "pct_entregas_atrasadas"]
    return out[colunas].sort_values("n_principal", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    executar()

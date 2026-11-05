"""SILVER: dado limpo, tipado e confiável.

- string vazia vira nulo; datas e números recebem o tipo correto;
- texto dos reviews: mantém o original e cria a versão limpa (limpar_texto);
- remove duplicatas exatas de (review_id, order_id);
- traduz categorias e registra checagens de integridade em data/silver/_qualidade.json.

Uso isolado:  python -m src.medallion.silver
"""

from __future__ import annotations

import json
import re

import pandas as pd

from src.medallion import BRONZE_DIR, SILVER_DIR, garantir_pastas

_QUEBRAS = re.compile(r"[\r\n\t]+")
_CONTROLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ESPACOS = re.compile(r"\s{2,}")


def limpar_texto(texto) -> str | None:
    """Limpeza leve: mantém acentos, caixa, pontuação e stopwords (importantes para embeddings).

    - quebras de linha e tabs viram espaço;
    - caracteres de controle são removidos;
    - espaços repetidos viram um só.
    """
    if not isinstance(texto, str):
        return None
    texto = _QUEBRAS.sub(" ", texto)
    texto = _CONTROLE.sub("", texto)
    texto = _ESPACOS.sub(" ", texto).strip()
    return texto or None


def _ler(nome: str) -> pd.DataFrame:
    df = pd.read_parquet(BRONZE_DIR / f"{nome}.parquet")
    df = df.drop(columns=["_arquivo_origem", "_ingestao_ts"], errors="ignore")
    return df.replace("", pd.NA)


def _datas(df: pd.DataFrame, colunas: list[str]) -> pd.DataFrame:
    for c in colunas:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def reviews(q: dict) -> pd.DataFrame:
    r = _ler("order_reviews")
    q["reviews_bronze"] = len(r)

    r["review_score"] = pd.to_numeric(r["review_score"], errors="coerce").astype("Int64")
    r = _datas(r, ["review_creation_date", "review_answer_timestamp"])

    r["titulo_limpo"] = r["review_comment_title"].map(limpar_texto).astype("string")
    r["mensagem_limpa"] = r["review_comment_message"].map(limpar_texto).astype("string")
    r["tem_texto"] = r["titulo_limpo"].notna() | r["mensagem_limpa"].notna()

    q["reviews_review_id_repetido"] = int(r["review_id"].duplicated(keep=False).sum())
    antes = len(r)
    r = r.drop_duplicates(subset=["review_id", "order_id"])
    q["reviews_duplicatas_exatas_removidas"] = antes - len(r)
    q["reviews_silver"] = len(r)
    q["reviews_com_texto"] = int(r["tem_texto"].sum())
    q["reviews_pct_com_texto"] = round(float(r["tem_texto"].mean()) * 100, 1)
    q["reviews_nota_invalida"] = int((~r["review_score"].between(1, 5)).fillna(True).sum())
    return r


def orders(q: dict) -> pd.DataFrame:
    o = _datas(_ler("orders"), [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ])
    q["orders_silver"] = len(o)
    q["orders_id_duplicado"] = int(o["order_id"].duplicated().sum())
    return o.drop_duplicates("order_id")


def order_items(q: dict) -> pd.DataFrame:
    i = _ler("order_items")
    i["order_item_id"] = pd.to_numeric(i["order_item_id"], errors="coerce").astype("Int64")
    for c in ["price", "freight_value"]:
        i[c] = pd.to_numeric(i[c], errors="coerce")
    i = _datas(i, ["shipping_limit_date"])
    q["order_items_silver"] = len(i)
    q["pedidos_com_mais_de_1_item"] = int((i.groupby("order_id").size() > 1).sum())
    return i


def products(q: dict) -> pd.DataFrame:
    p = _ler("products")
    t = _ler("product_category_name_translation")
    p = p.merge(t, on="product_category_name", how="left")
    p["categoria"] = p["product_category_name"].fillna("sem_categoria")
    p["categoria_en"] = p["product_category_name_english"].fillna(p["categoria"])
    q["products_silver"] = len(p)
    q["products_sem_categoria"] = int((p["categoria"] == "sem_categoria").sum())
    return p[["product_id", "categoria", "categoria_en"]]


def customers(q: dict) -> pd.DataFrame:
    c = _ler("customers")
    q["customers_silver"] = len(c)
    return c[["customer_id", "customer_unique_id", "customer_city", "customer_state"]]


def sellers(q: dict) -> pd.DataFrame:
    if not (BRONZE_DIR / "sellers.parquet").exists():
        return pd.DataFrame(columns=["seller_id", "seller_city", "seller_state"])
    s = _ler("sellers")
    q["sellers_silver"] = len(s)
    return s[["seller_id", "seller_city", "seller_state"]]


def _integridade(t: dict, q: dict) -> None:
    q["reviews_sem_pedido"] = int((~t["reviews"]["order_id"].isin(t["orders"]["order_id"])).sum())
    q["itens_sem_produto"] = int((~t["order_items"]["product_id"].isin(t["products"]["product_id"])).sum())
    q["pedidos_sem_cliente"] = int((~t["orders"]["customer_id"].isin(t["customers"]["customer_id"])).sum())
    q["pedidos_sem_itens"] = int((~t["orders"]["order_id"].isin(t["order_items"]["order_id"])).sum())


def executar() -> dict:
    garantir_pastas()
    q: dict = {}
    tabelas = {
        "reviews": reviews(q),
        "orders": orders(q),
        "order_items": order_items(q),
        "products": products(q),
        "customers": customers(q),
        "sellers": sellers(q),
    }
    _integridade(tabelas, q)
    for nome, df in tabelas.items():
        df.to_parquet(SILVER_DIR / f"{nome}.parquet", index=False)
        print(f"[silver] {nome:<15} {len(df):>8} linhas")
    (SILVER_DIR / "_qualidade.json").write_text(json.dumps(q, indent=2, ensure_ascii=False), encoding="utf-8")
    return q


if __name__ == "__main__":
    print(json.dumps(executar(), indent=2, ensure_ascii=False))

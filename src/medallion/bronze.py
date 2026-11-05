"""BRONZE: dado cru, sem interpretação.

1. Se os CSVs da Olist não estiverem em data/, baixa com kagglehub e copia para lá.
2. Salva cada CSV como parquet em data/bronze/, com todas as colunas como string,
   acrescentando só metadados de controle (_arquivo_origem, _ingestao_ts).

Uso isolado:  python -m src.medallion.bronze
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.medallion import BRONZE_DIR, KAGGLE_DATASET, RAW_DIR, garantir_pastas

ARQUIVO_PRINCIPAL = "olist_order_reviews_dataset.csv"


def nome_tabela(arquivo: Path) -> str:
    """olist_order_reviews_dataset.csv -> order_reviews"""
    return arquivo.stem.replace("olist_", "").replace("_dataset", "")


def baixar_csvs() -> None:
    if (RAW_DIR / ARQUIVO_PRINCIPAL).exists():
        print(f"[bronze] CSVs já estão em {RAW_DIR}, sem download.")
        return
    import kagglehub

    origem = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    print(f"[bronze] dataset baixado em {origem}")
    for csv in origem.glob("*.csv"):
        shutil.copy2(csv, RAW_DIR / csv.name)


def executar() -> dict[str, int]:
    garantir_pastas()
    baixar_csvs()
    ts = datetime.now(timezone.utc).isoformat()
    contagens = {}
    for csv in sorted(RAW_DIR.glob("*.csv")):
        # utf-8-sig remove o BOM do cabeçalho, se existir
        df = pd.read_csv(csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        df["_arquivo_origem"] = csv.name
        df["_ingestao_ts"] = ts
        nome = nome_tabela(csv)
        df.to_parquet(BRONZE_DIR / f"{nome}.parquet", index=False)
        contagens[nome] = len(df)
        print(f"[bronze] {nome:<35} {len(df):>8} linhas")
    return contagens


if __name__ == "__main__":
    executar()

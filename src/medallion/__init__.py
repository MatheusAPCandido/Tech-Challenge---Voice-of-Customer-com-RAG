"""Camadas da preparação da base (arquitetura medallion).

    CSVs Olist (data/) → bronze → silver → gold → data/reviews_docs.parquet

O ponto de entrada da equipe continua sendo `python -m src.prepare`,
que executa as três camadas em sequência.
"""

from pathlib import Path

from src.config import DATA_DIR

RAW_DIR = DATA_DIR                 # CSVs do Kaggle ficam direto em data/ (ver data/README.md)
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"

KAGGLE_DATASET = "olistbr/brazilian-ecommerce"


def garantir_pastas() -> None:
    for pasta in (DATA_DIR, BRONZE_DIR, SILVER_DIR, GOLD_DIR):
        Path(pasta).mkdir(parents=True, exist_ok=True)

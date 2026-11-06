# Dados

Os arquivos de dados **não** são versionados no Git.

O `python -m src.prepare` baixa o dataset automaticamente com `kagglehub` se os CSVs não estiverem
nesta pasta. Para baixar manualmente:

1. Baixe o dataset em https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data
2. Extraia os CSVs nesta pasta. Os usados no projeto são:

| Arquivo | Uso |
| --- | --- |
| `olist_order_reviews_dataset.csv` | Texto e nota das avaliações (principal) |
| `olist_orders_dataset.csv` | Datas de compra e entrega |
| `olist_order_items_dataset.csv` | Ligação pedido → produto |
| `olist_products_dataset.csv` | Categoria do produto |
| `product_category_name_translation.csv` | Nome da categoria em inglês |
| `olist_customers_dataset.csv` | Estado do cliente |
| `olist_sellers_dataset.csv` | Estado do vendedor (só nas tabelas extras da gold) |

Os arquivos gerados pelo projeto também ficam aqui e também não vão para o Git:

| Caminho | Conteúdo |
| --- | --- |
| `reviews_docs.parquet` | Documentos do RAG no formato de `src/schema.py` (entrada do `src.index`) |
| `amostra.parquet` | 1.000 documentos estratificados por nota (`--amostra`) |
| `bronze/` | CSVs convertidos para parquet, sem alterações |
| `silver/` | Tabelas limpas e tipadas + `_qualidade.json` |
| `gold/` | `reviews_completo.parquet`, `resumo_categorias.parquet` e `_qualidade.json` |

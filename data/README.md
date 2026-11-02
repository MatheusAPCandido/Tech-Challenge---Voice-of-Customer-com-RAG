# Dados

Os arquivos de dados **não** são versionados no Git.

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

Os arquivos gerados pelo projeto (`reviews_docs.parquet`, `amostra.parquet`) também ficam
aqui e também não vão para o Git.

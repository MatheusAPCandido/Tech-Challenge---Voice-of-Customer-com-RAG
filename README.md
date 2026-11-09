# Voice of Customer Intelligence com RAG

Tech Challenge — POSTECH IA para Devs / AI-Data Scientist, Fase 4.

Solução baseada em **Retrieval-Augmented Generation (RAG)** que permite a gestores e analistas
fazer perguntas em linguagem natural sobre as avaliações de clientes da base
[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data)
e receber respostas **fundamentadas nas avaliações reais**, com as evidências exibidas ao usuário.

> **Status:** em desenvolvimento. As seções marcadas com _TODO_ serão preenchidas pela equipe.

## Equipe

| Integrante | Papel |
| --- | --- |
| _TODO_ | Pessoa A — Dados e base de conhecimento |
| _TODO_ | Pessoa B — Recuperação, guardrails e avaliação |
| _TODO_ | Pessoa C — Geração, interface e entrega |

## Problema de negócio

Empresas de e-commerce recebem milhares de avaliações por dia. As notas mostram o nível de
satisfação, mas os motivos (atrasos, defeitos, atendimento, elogios) estão nos comentários em
texto livre, que são inviáveis de analisar manualmente. Esta solução transforma esses comentários
em uma base de conhecimento consultável por perguntas como:

- Quais são os principais problemas relatados pelos clientes?
- O que os clientes relatam sobre problemas de entrega?
- O que os clientes mais elogiam?
- Quais avaliações sustentam que existem atrasos recorrentes?

## Arquitetura

_TODO: inserir o fluxograma da solução (imagem em `docs/arquitetura.png`)._

```
Construção da base (offline, uma vez)
  CSVs Olist → Preparação (limpeza, metadados) → Embeddings → Índice vetorial + BM25

Consulta (a cada pergunta)
  Pergunta → Transformação (consulta + filtros) → No escopo?
     ├─ não → Recusa explícita
     └─ sim → Busca híbrida → Reranker → Evidência suficiente?
                 ├─ não → Recusa explícita
                 └─ sim → Contexto [R#] → Geração (LLM) → Validação de citações → Resposta + evidências
```

## Estrutura do repositório

```
.
├── app/streamlit_app.py     # interface com painel de evidências
├── data/                    # CSVs do Kaggle (não versionados)
├── eval/
│   ├── perguntas.csv        # conjunto de perguntas de teste
│   └── run_eval.py          # métricas de avaliação
├── notebooks/01_eda.ipynb   # análise exploratória
└── src/
    ├── config.py            # configurações e variáveis de ambiente
    ├── schema.py            # contrato de dados entre os módulos
    ├── prepare.py           # Etapa 1: seleção, limpeza, metadados (executa src/medallion)
    ├── medallion/           # camadas bronze → silver → gold da preparação
    ├── index.py             # embeddings, banco vetorial e BM25
    ├── retrieve.py          # transformação da pergunta e busca híbrida
    ├── guardrails.py        # escopo, limiar e evidência insuficiente
    ├── generate.py          # contexto, prompt, LLM e validação de citações
    └── pipeline.py          # orquestra pergunta → resposta
```

## Tecnologias

_TODO: confirmar escolhas e versões._

| Componente | Tecnologia |
| --- | --- |
| Linguagem | Python 3.11+ |
| Manipulação de dados | pandas |
| Embeddings | _TODO_ (ex.: `intfloat/multilingual-e5-base`) |
| Banco vetorial | ChromaDB |
| Busca lexical | rank-bm25 |
| Reranker | _TODO_ (ex.: `BAAI/bge-reranker-v2-m3`) |
| LLM | _TODO_ |
| Interface | Streamlit |

## Preparação da base

A preparação (`python -m src.prepare`) segue a arquitetura **medallion**, em `src/medallion/`.
A saída é `data/reviews_docs.parquet`, com exatamente as colunas de `schema.COLUNAS_DOCUMENTO`.

| Camada | O que faz | Onde fica |
| --- | --- | --- |
| **Bronze** | Cópia fiel dos CSVs em parquet, tudo como texto, + metadados de ingestão. Baixa do Kaggle com `kagglehub` se os CSVs não estiverem em `data/`. | `data/bronze/` |
| **Silver** | Tipagem (datas, números), string vazia → nulo, limpeza dos textos, remoção de duplicatas exatas, tradução de categorias e checagens de integridade. | `data/silver/` |
| **Gold** | Junta as tabelas, aplica as regras de seleção abaixo e gera os documentos do RAG. | `data/reviews_docs.parquet` e `data/gold/` |

### Seleção dos dados

Tabelas usadas: reviews (texto e nota), orders (datas de compra e entrega), order_items + products
(categoria) e customers (UF). Pagamentos e geolocalização ficam de fora porque não acrescentam
informação útil às perguntas de Voice of Customer.

### Limpeza dos textos

Limpeza leve, porque os modelos de embedding funcionam melhor com texto natural: quebras de linha
e tabs viram espaço, caracteres de controle são removidos e espaços repetidos são unidos.
**Acentos, maiúsculas, pontuação e stopwords são mantidos.** O `texto` é `"Título. Mensagem"`,
ou só a parte que existir. Se o título repete a mensagem, ele não é duplicado.

### Regras de seleção e números (base completa, 99.224 reviews)

| Regra | Removidas | Justificativa |
| --- | ---: | --- |
| Sem título nem mensagem | 56.537 | Sem texto não há o que recuperar. A nota continua disponível na silver. |
| Menos de 3 letras (`.`, `ok`, `10`, `*****`, emojis soltos) | 421 | Não carregam informação. "Bom" e "Ruim" ficam, porque expressam sentimento. |
| Mesmo `review_id` ligado a mais de um pedido | 315 | O texto é o mesmo; um documento por review evita citações duplicadas. |
| Texto idêntico (ignorando caixa, acentos e pontuação) | 7.039 | Evita que dezenas de cópias de "muito bom" ocupem o top-k da busca. Fica a avaliação mais antiga. |
| **Documentos no índice** | **34.912** | |

### Valores ausentes e metadados

| Coluna | Regra |
| --- | --- |
| `data_compra` | Mês da compra (`AAAA-MM`), de `order_purchase_timestamp`. |
| `dias_atraso` | Data real de entrega − data prevista, em dias (positivo = atrasou). `NaN` nos 1.710 pedidos não entregues. |
| `entregue` | `True` quando existe data de entrega ao cliente. |
| `categoria` | Categoria em português do **item mais caro** do pedido (9,8 mil pedidos têm mais de um item). Sem categoria ou pedido sem itens → `sem_categoria` (990 documentos). |
| `estado` | UF do cliente, sem ausentes. |
| `tema` | Assunto principal da review (ver "Temas das avaliações" abaixo). Sem assunto identificado → `outros`. |

### Saídas extras em `data/gold/`

- `reviews_completo.parquet`: as 41.951 reviews com conteúdo, **antes** da deduplicação de textos, com colunas
  extras (`sentimento`, `categoria_en`, `todas_categorias`, `n_itens`, `valor_pedido`, `estado_seller`,
  `order_status`), `temas` (todos os temas citados, separados por `|`) e `qtd_textos_identicos`.
  Use esta tabela para respostas agregadas, porque ela preserva as contagens.
- `resumo_categorias.parquet`: nota média, % de negativas e % de atraso por categoria.
- `resumo_temas.parquet`: por tema, quantas reviews o têm como assunto principal, quantas o citam,
  nota média, % de negativas e % de entregas atrasadas.
- `_qualidade.json` (também em `data/silver/`): todas as contagens acima, geradas a cada execução.

### Achado relevante para o RAG

Entre as entregas atrasadas, **74% das avaliações são negativas (nota 1–2)**, contra 21% quando a
entrega chega no prazo. O atraso é o principal motor de insatisfação, e `dias_atraso` é o
metadado mais útil para filtros (`atrasou=True`).

### Temas das avaliações (T08)

Cada review recebe um `tema`: o assunto principal da avaliação. A lista de temas e as regras
ficam em `src/medallion/temas.py`, e o tema é calculado na camada gold, junto com o resto da preparação.

**Como os temas foram definidos.** Primeiro, agrupamos as 34.912 reviews por similaridade dos
embeddings (KMeans, 30 grupos) para descobrir os assuntos que a base realmente tem. Os grupos
misturavam assunto e jeito de escrever (textos só em maiúsculas, textos de uma palavra), então
serviram para definir a lista de temas, não como rótulo final. Depois, cada review recebe os temas
cujas palavras-chave aparecem no texto. As regras tratam negação ("sem nenhum defeito", "não
atrasou", "nada a reclamar") e variações comuns ("comprei 2 e veio só 1", "recebi 2 dos 3").
Quando a review cita mais de um assunto, o principal é o mais específico ou grave, na ordem da tabela.

| Tema | Descrição | Reviews (principal) | Nota média | % negativas |
| --- | --- | ---: | ---: | ---: |
| `entrega_incompleta` | Pedido incompleto: faltou item, unidade ou peça | 2.236 | 1,86 | 72% |
| `nao_recebido` | Produto não recebido | 3.623 | 1,47 | 86% |
| `produto_errado` | Produto errado ou diferente do anunciado | 993 | 1,87 | 73% |
| `produto_defeito` | Produto com defeito, quebrado ou danificado | 1.138 | 2,16 | 63% |
| `atraso_entrega` | Entrega atrasada ou demorada | 1.301 | 2,66 | 47% |
| `atendimento_loja` | Atendimento, troca, devolução, cancelamento ou nota fiscal | 2.698 | 3,16 | 41% |
| `qualidade_produto` | Qualidade, material, acabamento ou tamanho | 3.809 | 4,12 | 13% |
| `preco_frete` | Preço, custo-benefício ou frete | 1.074 | 4,09 | 13% |
| `entrega_rapida` | Entrega rápida ou dentro do prazo | 8.485 | 4,74 | 1% |
| `avaliacao_geral` | Opinião geral, sem assunto específico | 12.484 | 4,50 | 6% |
| `outros` | Nenhum dos assuntos acima | 4.110 | 3,16 | 37% |

Contagens sobre as 41.951 reviews com conteúdo (`resumo_temas.parquet`, inclui textos repetidos).

**Validação.** Rotulamos à mão duas amostras sorteadas. Na primeira (120 reviews), usada para
ajustar as regras, o tema principal bate em 96,7% dos casos, número otimista porque as regras foram
corrigidas olhando essa amostra. Na segunda (100 reviews novas, nunca usadas no ajuste), o acerto
é de **80%**. Os erros mais comuns são paráfrases que nenhuma palavra-chave cobre ("comprei num dia,
recebi no outro") e reviews com dois assuntos de peso parecido.

**Alternativas testadas e descartadas.**
- Comparar cada review com frases-modelo de cada tema usando os embeddings: as similaridades do E5
  ficam muito próximas entre temas (diferença menor que 0,01), e reviews longas caíam no tema errado.
- Classificador (regressão logística) sobre os embeddings, treinado com os rótulos das regras: subiu
  o acerto da amostra nova de 80% para 82% e piorou a de ajuste, uma diferença dentro da margem de
  erro de 100 exemplos. Ficamos com as regras, que são determinísticas e explicáveis.

**Uso no RAG.** `tema` está no ChromaDB como metadado filtrável (`where={"tema": "produto_defeito"}`)
e a lista de valores válidos está em `src.medallion.temas.TEMAS`. Para perguntas agregadas ("quais
os principais motivos de reclamação?"), use `resumo_temas.parquet` ou a coluna `temas` de
`reviews_completo.parquet`, que preservam as contagens reais.

### Embeddings e indexação

_TODO (T06/T07): modelo escolhido, comparação e armazenamento no ChromaDB + BM25._

## Pipeline RAG

_TODO (Pessoa B e C): descrever e justificar_

- Transformação da pergunta
- Recuperação semântica e híbrida
- Seleção dos conteúdos (reranker, limiar)
- Construção do contexto
- Geração da resposta e citações

## Tratamento de perguntas sem evidência

_TODO (Pessoa B): descrever as três situações (fora de escopo, sem evidência, evidência
insuficiente) e como o limiar foi calibrado._

## Avaliação

_TODO (Pessoa B): tabela de resultados (precisão@k, fidelidade, taxa de recusa correta)
comparando as variações testadas._

## Exemplos de perguntas e respostas

_TODO (Pessoa C): incluir pelo menos uma pergunta pontual, uma agregada, uma com filtro e uma recusa._

## Como executar

### 1. Clonar e instalar

```bash
git clone https://github.com/MatheusAPCandido/Tech-Challenge---Voice-of-Customer-com-RAG.git
cd Tech-Challenge---Voice-of-Customer-com-RAG
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configurar variáveis de ambiente

```bash
cp .env.example .env   # no Windows: copy .env.example .env
# edite o .env com suas chaves
```

### 3. Baixar os dados

Automático: se os CSVs não estiverem em `data/`, o `python -m src.prepare` baixa o dataset com
`kagglehub` e copia os arquivos para lá. Também dá para baixar manualmente no
[Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data) e extrair os CSVs na
pasta `data/` (veja `data/README.md`).

### 4. Construir a base de conhecimento

```bash
python -m src.prepare
python -m src.index
```

### 5. Rodar a aplicação

```bash
streamlit run app/streamlit_app.py
```

Com `USE_MOCK=true` no `.env`, a interface roda com dados falsos, sem precisar da base pronta.

### 6. Rodar a avaliação

```bash
python -m eval.run_eval
```

## Principais decisões técnicas

_TODO: resumo das decisões e justificativas, com números da avaliação._

## Limitações e próximos passos

_TODO_

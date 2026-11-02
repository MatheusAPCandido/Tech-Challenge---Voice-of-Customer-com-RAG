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
    ├── prepare.py           # Etapa 1: seleção, limpeza, metadados
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

_TODO (Pessoa A): descrever e justificar_

- Seleção dos dados e tabelas utilizadas
- Limpeza e preparação dos textos
- Tratamento de valores ausentes
- Organização dos documentos (uma review = um documento)
- Criação dos embeddings
- Armazenamento e indexação

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

Baixe o dataset no [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data)
e extraia os CSVs na pasta `data/` (veja `data/README.md`).

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

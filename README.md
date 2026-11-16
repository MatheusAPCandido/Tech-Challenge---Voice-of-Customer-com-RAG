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

## Pipeline RAG

O contrato entre os módulos (documentos, filtros, `Evidencia`, `ConsultaProcessada`, `Veredito`,
`Resposta`) está em `src/schema.py`. Em cada pergunta, o fluxo de recuperação é:

1. **Transformação (T11).** `processar_pergunta()` em `src/retrieve.py` classifica o tipo (`pontual` / `agregada` / `fora_de_escopo`), extrai filtros e monta a consulta textual. Com `LLM_PROVIDER=ollama` tenta JSON via Ollama (modelo `LLM_MODEL`, tipicamente Qwen2.5); se o daemon falhar, cai nas regras locais.
2. **Busca híbrida (T12).** `buscar()` combina Chroma (vetorial) + BM25 e funde com RRF (`RRF_K=60`), formando um pool de candidatos.
3. **Filtros (T13).** `FILTROS_SUPORTADOS` (nota, categoria, estado, atraso, período, tema) entram no `where` do Chroma e no mesmo recorte do BM25 antes do RRF.
4. **Reranker (T14).** O cross-encoder `BAAI/bge-reranker-v2-m3` reordena o pool (`TOP_K_CANDIDATOS`) e corta em `TOP_K_FINAL`; o score final é `sigmoid(logit)` ≈`(0, 1]`.
5. **Limiar (T15).** Só evidências com `score >= LIMIAR_RELEVANCIA` (default `0.5`, calibrado com rótulos humanos) seguem; detalhes na seção [Tratamento de perguntas sem evidência](docs/execucao_tarefas.md#tratamento-de-perguntas-sem-evidência).

Construção do contexto com `[R#]`, chamada ao LLM e validação de citações ficam no parágrafo de
geração abaixo e na [documentação das tarefas](docs/execucao_tarefas.md#detalhes-de-geração-e-interface).

**Geração (T21) e Ollama:** com `GEN_PROVIDER=ollama` (ou fallback do Gemini), modelos locais
às vezes omitem `[R#]`. O código reforça a instrução JSON/user e, se a 1ª saída passar em
`pode_responder` mas falhar no T22 (sem citações válidas), tenta **1 reescrita** antes de
recusar. A validação de citações não é afrouxada.

## Estrutura do repositório

```
.
├── app/streamlit_app.py     # interface com painel de evidências
├── data/                    # CSVs do Kaggle (não versionados; ver data/README.md)
├── docs/
│   ├── execucao_tarefas.md  # como cada tarefa foi feita, com números e decisões
│   └── eda/                 # gráficos da análise exploratória
├── eval/
│   ├── perguntas.csv           # conjunto de perguntas de teste
│   ├── gerar_rotulos_t15.py    # T15: planilha para calibrar o limiar
│   ├── calibrar_limiar.py      # T15: sugere LIMIAR_RELEVANCIA
│   ├── run_eval.py             # T17a: taxa de recusa correta
│   ├── precisao_k.py           # T17a: precisão@k (rótulos T15)
│   ├── gerar_fidelidade.py     # T17b: planilha XLSX para rotular fidelidade
│   ├── calcular_fidelidade.py  # T17b: macro/micro a partir dos rótulos
│   ├── comparar_embeddings.py  # T07: compara modelos de embedding
│   ├── avaliacao_final.py      # T28: avaliação final e comparação de variações
│   └── rotulos_io.py           # leitura compartilhada da planilha de rótulos T15
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
    ├── agregar.py           # contagens por tema para perguntas agregadas
    └── pipeline.py          # orquestra pergunta → resposta
```

## Tecnologias

_TODO: confirmar escolhas e versões._

| Componente | Tecnologia |
| --- | --- |
| Linguagem | Python 3.11+ |
| Manipulação de dados | pandas |
| Embeddings | `intfloat/multilingual-e5-base` (escolha na T07, ver [Embeddings e indexação](docs/execucao_tarefas.md#embeddings-e-indexação)) |
| Banco vetorial | ChromaDB |
| Busca lexical | rank-bm25 |
| Reranker | `BAAI/bge-reranker-v2-m3` (cross-encoder) |
| LLM | 2 opções: Ollama com Qwen2.5 e Gemini |
| Interface | Streamlit |

## Avaliação

Nas **avaliações com geração real** (em especial a fidelidade [T17b]), o LLM usado foi **sempre**  
**Qwen2.5 via Ollama** (`GEN_PROVIDER=ollama`, `GEN_MODEL=qwen2.5`, `GEN_FALLBACK_MODEL` vazio na
medição). Durante o projeto o **Gemini também foi utilizado** na interface
(`GEN_PROVIDER=gemini`, com Ollama como reserva quando configurado). Os números de fidelidade
abaixo referem-se ao setup Qwen — não ao Gemini.

| Métrica | Setup | Valor |
| --- | --- | --- |
| Recusa correta | busca real, `GEN_MOCK=true` | ~79% (n=29) |
| Macro precisão@5 / @10 | rótulos T15 | ~0,90 / ~0,85 (n=18) |
| Fidelidade macro / micro | Ollama `qwen2.5` | 1,0 / 1,0 (n=15 elegíveis) |
| Recusa correta (T28) | busca real, T11 por regras, `GEN_MOCK=true` | ~97% (n=29) |
| Precisão@5 / @10 (T28) | top-10 do pipeline atual, todas as evidências rotuladas | 0,72 / 0,71 (n=21) |

As linhas da T17a e da T28 não medem exatamente a mesma coisa; a diferença é explicada em
[Avaliação final e comparação de variações (T28)](docs/execucao_tarefas.md#avaliação-final-e-comparação-de-variações-t28).

Como cada métrica foi calculada, a comparação das variações da busca (T28) e as limitações de
cada medição estão em [Avaliação em detalhe](docs/execucao_tarefas.md#avaliação-em-detalhe).

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

### 6. Rodar a avaliação (T17a, T17b e T28)

```bash
# T17a — recusa correta + precisão@k (sem LLM de geração)
USE_MOCK=false GEN_MOCK=true HF_HUB_OFFLINE=1 python -m eval.run_eval
python -m eval.precisao_k

# T17b — fidelidade (geração real com Qwen2.5 via Ollama)
USE_MOCK=false GEN_MOCK=false GEN_PROVIDER=ollama GEN_MODEL=qwen2.5 \
  GEN_FALLBACK_MODEL= JUDGE_ENABLED=false HF_HUB_OFFLINE=1 \
  python -m eval.gerar_fidelidade
# copiar eval/fidelidade_para_rotular.xlsx → eval/fidelidade_rotulos.xlsx e rotular B6–B8
python -m eval.calcular_fidelidade

# T28 — avaliação final e comparação de variações (sem LLM de geração)
python -m eval.avaliacao_final
```

Resultados na seção [Avaliação](#avaliação).

### 7. Testes (pytest)

Suíte unitária rápida (sem índice Chroma, embeddings, reranker ou LLM de geração):

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest
```

Por padrão o marker `integration` fica de fora (`addopts = -m "not integration"`). Os testes de integração (índice real e gate de suficiência via Ollama) estão em
[Testes de integração](docs/execucao_tarefas.md#testes-de-integração).

## Principais decisões técnicas

- A busca combina índice vetorial (Chroma) e BM25 e funde os rankings com RRF, para recuperar tanto paráfrases quanto termos literais das avaliações.
- Depois do RRF, um cross-encoder (`BAAI/bge-reranker-v2-m3`) reordena o pool e define o score final usado no limiar de relevância.
- O limiar ficou em `LIMIAR_RELEVANCIA=0.5`, calibrado com rótulos humanos (T15). Nesse grid, 0,5 maximizou úteis retidos menos inúteis retidos; valores ≥ 0,55 passaram a zerar evidências em perguntas que deveriam responder.
- Os guardrails recusam fora de escopo, ausência de evidência acima do limiar e evidência insuficiente, para o sistema não inventar resposta sem suporte nas avaliações. A taxa de recusa correta no conjunto de teste foi de cerca de 79% (29 perguntas).
- O gate opcional `JUDGE_ENABLED` permanece desligado por padrão, para a avaliação e o uso cotidiano serem reprodutíveis sem depender do Ollama; quando ligado, usa Ollama com o modelo da transformação da pergunta (Qwen), e não o Gemini de geração.
- A precisão@k foi medida só com rótulos da planilha T15, sem LLM de geração. A macro precisão@5 ficou em cerca de 0,90 e a @10 em cerca de 0,85 (18 perguntas com rótulos suficientes).
- A fidelidade (groundedness) foi avaliada manualmente, sem RAGAS, com respostas geradas por Qwen2.5 via Ollama (`GEN_PROVIDER=ollama`, fallback vazio). Macro e micro ficaram em 1,0 em 15 perguntas elegíveis. Essa medição não se confunde com o Gemini usado na interface.
- O modelo de embedding é o `intfloat/multilingual-e5-base`. Na T07, o `BAAI/bge-m3` foi melhor na busca vetorial, mas a equipe manteve o `e5-base` porque toda a calibração e a avaliação foram feitas com ele.
- Na avaliação final (T28), o reranker aumentou a precisão@10 de 0,61 (híbrida sem reranker) para 0,71, com diferença conclusiva; contra a busca só vetorial, o ganho (+0,09) não foi conclusivo com 21 perguntas.

## Limitações e próximos passos

**Limitações (recuperação e avaliação).**
- A transformação da pergunta combina heurísticas locais com um modelo opcional via Ollama, o que favorece reprodutibilidade e baixo custo, com a natural restrição de cobertura de formulações muito atípicas.
- Casos no limite entre “no escopo” e “sem evidência específica” (por exemplo perguntas hipotéticas com vocabulário de domínio) dependem do limiar e, se desejado, do gate opcional de suficiência.
- As métricas de precisão@k e de fidelidade usam rotulagem humana, alinhada à prática de avaliação de RAG; isso introduz variação entre anotadores e exige conjuntos com rótulos suficientes por pergunta.
- A fidelidade reportada foi medida com geração Qwen2.5 via Ollama, em protocolo controlado e com um único modelo, o que torna o resultado interpretável para esse setup de avaliação.
- O gate `JUDGE_ENABLED` permanece desligado por padrão para priorizar estabilidade e reprodutibilidade nas medições; quando ativado, acrescenta uma checagem semântica adicional via Ollama.
- Na avaliação final (T28), o limiar de 0,5 não cortou nenhuma evidência do top-10, e as recusas corretas vieram da classificação de escopo (T11). O limiar não barra perguntas no escopo mas sem resposta, como a P09 (entrega por drone).
- A transformação da pergunta não extrai filtro de período ("junho a agosto de 2018", "início de 2018"), o que derruba a precisão dessas perguntas (P28 a P30).

**Próximos passos (recuperação e avaliação).**
- Ampliar e refinar o conjunto de rótulos humanos (T15) e, após mudanças relevantes no índice, recalibrar o limiar de relevância.
- Explorar o gate de suficiência em cenários selecionados, documentando o impacto na qualidade percebida versus custo operacional.
- Avaliar a integração de frameworks de métricas automáticas para RAG (por exemplo RAGAS), em complemento à fidelidade manual, para acelerar iterações mantendo um protocolo humano de referência.

## Documentação detalhada

- [`docs/execucao_tarefas.md`](docs/execucao_tarefas.md): como cada tarefa foi executada (preparação da base, temas, embeddings,
guardrails, avaliação, geração e interface), com os números, as alternativas testadas e as limitações.
- [`data/README.md`](data/README.md): arquivos de dados usados e gerados.
- [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb): análise exploratória (gráficos em `docs/eda/`).

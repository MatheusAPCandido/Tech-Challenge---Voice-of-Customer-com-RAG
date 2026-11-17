# Voice of Customer Intelligence com RAG

Tech Challenge — POSTECH IA para Devs / AI-Data Scientist, Fase 4.

Solução baseada em **Retrieval-Augmented Generation (RAG)** que permite a gestores e analistas
fazer perguntas em linguagem natural sobre as avaliações de clientes da base
[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data)
e receber respostas **fundamentadas nas avaliações reais**, com as evidências exibidas ao usuário.


## Equipe

| Integrante | Papel |
| --- | --- |
| Matheus Cândido | Pessoa A — Dados e base de conhecimento |
| Matheus Burger | Pessoa B — Recuperação, guardrails e avaliação |
| Diego Nascimento | Pessoa C — Geração, interface e entrega |

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
│   ├── rotulos_t15.xlsx        # T15: rótulos humanos (limiar / precisão@k)
│   ├── fidelidade_rotulos.xlsx # T17b: rótulos humanos de fidelidade
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

| Componente | Tecnologia |
| --- | --- |
| Linguagem | Python 3.11+ |
| Manipulação de dados | pandas |
| Embeddings | `intfloat/multilingual-e5-base` (escolha na T07, ver [Embeddings e indexação](docs/execucao_tarefas.md#embeddings-e-indexação)) |
| Banco vetorial | ChromaDB |
| Busca lexical | rank-bm25 |
| Reranker | `BAAI/bge-reranker-v2-m3` (cross-encoder) |
| LLM | `qwen2.5` via Ollama (local; usado na avaliação e na demonstração) ou Gemini por API |
| Interface | Streamlit |

## Avaliação

Nas **avaliações com geração real** (em especial a fidelidade [T17b]), o LLM usado foi **sempre**  
**Qwen2.5 via Ollama** (`GEN_PROVIDER=ollama`, `GEN_MODEL=qwen2.5`, `GEN_FALLBACK_MODEL` vazio na
medição). Durante o projeto o **Gemini também foi utilizado** na interface
(`GEN_PROVIDER=gemini`, com Ollama como reserva quando configurado). Os números de fidelidade
abaixo referem-se ao setup Qwen — não ao Gemini.

| Métrica | Medição | Valor | Rótulos |
| --- | --- | --- | --- |
| Recusa correta | T17a: busca real, sem geração, configuração da época | ~79% (n=29) | gabarito de `eval/perguntas.csv` |
| Precisão@5 / @10 | T17a: rótulos da T15 | ~0,90 / ~0,85 (n=18) | humanos ([`eval/rotulos_t15.xlsx`](eval/rotulos_t15.xlsx)) |
| Fidelidade macro / micro | T17b: geração com `qwen2.5` via Ollama | 1,0 / 1,0 (n=15) | humanos ([`eval/fidelidade_rotulos.xlsx`](eval/fidelidade_rotulos.xlsx)) |
| **Recusa correta** | **T28: busca real, T11 por regras, sem geração** | **~97% (n=29)** | gabarito de `eval/perguntas.csv` |
| **Precisão@5 / @10** | **T28: todo o top-10 do pipeline atual** | **0,72 / 0,71 (n=21)** | **assistente de IA, às cegas** ([`eval/t28_rotulos.csv`](eval/t28_rotulos.csv)) |


Como cada métrica foi calculada, a comparação das variações da busca (T28) e as limitações de
cada medição estão em [Avaliação em detalhe](docs/execucao_tarefas.md#avaliação-em-detalhe).

## Exemplos de perguntas e respostas

Saídas reais do sistema, geradas em 16/11/2026 no Google Colab (GPU T4) com o índice da base completa,
na mesma configuração da avaliação final (T28): geração com `qwen2.5` via Ollama, sem modelo de reserva
(`GEN_FALLBACK_MODEL` vazio), transformação da pergunta por regras (`LLM_PROVIDER` diferente de `ollama`)
e `JUDGE_ENABLED=false`. Com os modelos já carregados, cada pergunta levou de 3 a 6 segundos; a primeira
pergunta da sessão leva cerca de 1 minuto, porque carrega os modelos. As evidências estão resumidas:
a interface mostra as dez recuperadas, com o texto completo, a data, o atraso, o estado e a relevância.

### 1. Pergunta pontual

**Pergunta:** "O que dizem sobre trocas, devoluções ou arrependimento da compra?"

> Os clientes relatam problemas com trocas, devoluções e arrependimento da compra [R1][R2][R3][R5][R6][R7][R9][R10]:
> arrependimento de ter comprado [R1][R9], dificuldade em registrar trocas ou devoluções [R5], problemas com o
> processo de devolução [R6][R7], solicitação de informações sobre estorno [R10].

Confiança: alta · 8 de 10 avaliações recuperadas citadas. Algumas evidências:

| R# | Nota | Categoria | Trecho da avaliação |
| --- | ---: | --- | --- |
| R1 | 5 | bebes | "Eu me arrependi dessa compra e gostaria de devolver. Como eu faço?" |
| R5 | 1 | informatica_acessorios | "Preciso realizar a troca do produto que está com defeito e não consigo registrar a solicitação no SITE." |
| R7 | 1 | esporte_lazer | "Quero exercer meu direito de arrependimento, devolver o produto e receber meu dinheiro de volta. O site da stark dificulta o processo." |
| R10 | 2 | cama_mesa_banho | "Solicito informações sobre a compra e o estorno da compra." |

### 2. Pergunta com filtro

**Pergunta:** "Há reclamações de clientes de SP sobre atraso na entrega?"

A transformação da pergunta extraiu três filtros: `estado = SP`, `atrasou = True` e `review_score_max = 2`
(a palavra "reclamações" restringe a notas 1 e 2).

> Há reclamações de atraso na entrega em SP [R1][R2][R3][R4][R5][R6][R7][R8][R9][R10].

Confiança: alta · 10 de 10 avaliações citadas, todas de SP, com entrega atrasada e nota 1 ou 2. Algumas evidências:

| R# | Nota | Categoria | Trecho da avaliação |
| --- | ---: | --- | --- |
| R1 | 2 | sem_categoria | "Entrega em atraso" |
| R5 | 1 | market_place | "O produto está com a entrega em atraso. Até o momento não obtive nenhum retorno do meu chamado que está em aberto!" |
| R10 | 1 | automotivo | "Acredito que a entrega não foi realizada devido a greve dos caminhoneiros." |

### 3. Pergunta agregada (frequência)

**Pergunta:** "Quais padrões aparecem nas avaliações de clientes insatisfeitos?"

A resposta começa pelas contagens, feitas em código sobre a base inteira (T24), e não só sobre as avaliações recuperadas:

> **Na base inteira**, entre 10.836 avaliações com texto (notas 1 e 2 · todos os períodos), os assuntos que os clientes mais relatam são:
> - Produto não recebido — 3.102 (28,6%)
> - Pedido incompleto: faltou item, unidade ou peça — 1.610 (14,9%)
> - Atendimento da loja, troca, devolução, cancelamento ou nota fiscal — 1.092 (10,1%)
> - Produto errado ou diferente do anunciado — 724 (6,7%)
> - Produto com defeito, quebrado ou danificado — 711 (6,6%)
> - Sem assunto específico (opinião geral ou outros) — 2.226 (20,5%)

Em seguida, o modelo ilustra com exemplos das avaliações recuperadas:

> Os clientes insatisfeitos relatam atrasos na entrega [R2][R5][R6][R7][R8], produtos não entregues [R2][R8],
> produtos que não correspondem ao pedido [R9][R10], e falta de consideração com o cliente [R6][R7].

**Observação:** três das quatro afirmações do parágrafo são sustentadas pelas avaliações citadas. A primeira,
"atrasos na entrega", não é: as avaliações citadas falam de produto não entregue (R2, R8), de insatisfação
genérica (R5) e de descaso (R6, R7). É o tipo de erro descrito em [Limitações](#limitações-e-próximos-passos):
o código confere se cada `[R#]` existe, mas não se a avaliação sustenta a frase.

### 4. Recusa: pergunta fora do escopo

**Pergunta:** "Qual a previsão do tempo em São Paulo amanhã?"

> Esta pergunta está fora do escopo das avaliações de clientes.

Motivo: `fora_de_escopo`. A recusa acontece antes da busca, então nenhuma avaliação é recuperada.

### 5. Recusa: no tema, mas sem evidência

**Pergunta:** "O que os clientes dizem sobre entrega por drone?"

> As avaliações recuperadas não permitem responder a esta pergunta.

Motivo: `sem_evidencia`. A busca recuperou dez avaliações sobre entrega, e nenhuma trata de entrega por drone.
Uma delas fala de um drone de brinquedo ("O drone é lindo, porém a bateria não dura nada"), e o modelo não a
confundiu com entrega por drone. O modelo devolveu a frase de recusa, e o código registrou a resposta como recusa.
Na avaliação final (T28), medida sem gerar respostas, esta pergunta (P09) foi contada como erro; com a geração
real, o sistema recusa.

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
# rótulos versionados: eval/rotulos_t15.xlsx
USE_MOCK=false GEN_MOCK=true HF_HUB_OFFLINE=1 python -m eval.run_eval
python -m eval.precisao_k --xlsx eval/rotulos_t15.xlsx

# T17b — fidelidade (geração real com Qwen2.5 via Ollama)
# rótulos versionados: eval/fidelidade_rotulos.xlsx
USE_MOCK=false GEN_MOCK=false GEN_PROVIDER=ollama GEN_MODEL=qwen2.5 \
  GEN_FALLBACK_MODEL= JUDGE_ENABLED=false HF_HUB_OFFLINE=1 \
  python -m eval.gerar_fidelidade
# se for regenerar: copiar fidelidade_para_rotular.xlsx → fidelidade_rotulos.xlsx e rotular B6–B8
python -m eval.calcular_fidelidade --xlsx eval/fidelidade_rotulos.xlsx

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

Por padrão o marker `integration` fica de fora (`addopts = -m "not integration and not llm"`). Os testes de integração (índice real e gate de suficiência via Ollama) estão em
[Testes de integração](docs/execucao_tarefas.md#testes-de-integração).

## Principais decisões técnicas

- A busca combina índice vetorial (Chroma) e BM25 e funde os rankings com RRF, para recuperar tanto paráfrases quanto termos literais das avaliações.
- Depois do RRF, um cross-encoder (`BAAI/bge-reranker-v2-m3`) reordena o pool e define o score final usado no limiar de relevância.
- O limiar ficou em `LIMIAR_RELEVANCIA=0.5`, calibrado com rótulos humanos (T15) em [`eval/rotulos_t15.xlsx`](eval/rotulos_t15.xlsx). Nesse grid, 0,5 maximizou úteis retidos menos inúteis retidos; valores ≥ 0,55 passaram a zerar evidências em perguntas que deveriam responder.
- Os guardrails recusam fora de escopo, ausência de evidência acima do limiar e evidência insuficiente. Na avaliação final (T28), a recusa correta foi de 97% (29 perguntas), mas todas as recusas corretas vieram da classificação de escopo (T11): o limiar de 0,5 não cortou nenhuma evidência do top-10 (ver Limitações).
- O gate opcional `JUDGE_ENABLED` permanece desligado por padrão, para a avaliação e o uso cotidiano serem reprodutíveis sem depender do Ollama; quando ligado, usa Ollama com o modelo da transformação da pergunta (Qwen), e não o Gemini de geração.
- A precisão@k foi medida só com rótulos humanos da planilha T15 ([`eval/rotulos_t15.xlsx`](eval/rotulos_t15.xlsx)), sem LLM de geração. A macro precisão@5 ficou em cerca de 0,90 e a @10 em cerca de 0,85 (18 perguntas com rótulos suficientes).
- A fidelidade (groundedness) foi avaliada manualmente, sem RAGAS, com respostas geradas por Qwen2.5 via Ollama (`GEN_PROVIDER=ollama`, fallback vazio) e rótulos em [`eval/fidelidade_rotulos.xlsx`](eval/fidelidade_rotulos.xlsx). Macro e micro ficaram em 1,0 em 15 perguntas elegíveis. Essa medição não se confunde com o Gemini usado na interface.
- O modelo de embedding é o `intfloat/multilingual-e5-base`. Na T07, o `BAAI/bge-m3` foi melhor na busca vetorial, mas a equipe manteve o `e5-base` porque toda a calibração e a avaliação foram feitas com ele.
- Na avaliação final (T28), o reranker aumentou a precisão@10 de 0,61 (híbrida sem reranker) para 0,71, com diferença conclusiva; contra a busca só vetorial, o ganho (+0,09) não foi conclusivo com 21 perguntas.

## Limitações e próximos passos

**Limitações (recuperação e avaliação).**
- A transformação da pergunta foi avaliada só na versão por regras (`LLM_PROVIDER` diferente de `ollama`). Perguntas formuladas de um jeito que as regras não preveem podem ser mal classificadas ou ficar sem filtro.
- Casos no limite entre “no escopo” e “sem evidência específica” (por exemplo perguntas hipotéticas com vocabulário de domínio) dependem do limiar e, se desejado, do gate opcional de suficiência.
- Os rótulos de precisão da T15 e de fidelidade da T17b foram feitos por uma pessoa da equipe; os da T28, por um assistente de IA, às cegas e sem conferência humana. Não medimos a concordância entre rotuladores.
- A fidelidade foi medida só com o `qwen2.5` (15 respostas). Com o Gemini, que também pode ser usado na interface, não foi medida.
- A validação do código confere se cada `[R#]` existe, mas não se a avaliação citada sustenta a frase nem se números como "X de N" estão certos. Na rotulagem da fidelidade, em 5 das 15 respostas (P03, P04, P05, P06 e P13) uma citação apontava para uma avaliação que não sustentava a afirmação; a fidelidade conta a afirmação como sustentada quando alguma evidência a sustenta. A fidelidade foi medida na configuração da T17a (79% de recusa correta), não na da T28.
- O gate `JUDGE_ENABLED` fica desligado por padrão.
- Na avaliação final (T28), o limiar de 0,5 não cortou nenhuma evidência do top-10, e as recusas corretas vieram da classificação de escopo (T11). Como a T28 não gera respostas, a P09 (entrega por drone) contou como erro. Com a geração real, o modelo recusa a P09 (exemplo 5), mas essa recusa depende do modelo, e não de uma regra do código.
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

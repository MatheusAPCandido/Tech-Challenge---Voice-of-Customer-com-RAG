# T28 — Avaliação final (atual)

Configuração: embedding `intfloat/multilingual-e5-base`, reranker `BAAI/bge-reranker-v2-m3`, T11 `openai`, pool 40, top-10, limiar 0.5.
Perguntas: 29 (21 para responder, 8 para recusar).

## Busca (perguntas que devem ser respondidas)

| Variação | Perguntas | P@5 | P@10 | MRR@10 | Evidências | Sem rótulo | Diferença P@10 vs atual (IC95%) |
|---|---:|---:|---:|---:|---:|---:|---|
| vetorial | 21 | 0.629 | 0.614 | 0.837 | 10.0 | 1 | -0.086 [-0.195; +0.005] |
| bm25 | 21 | 0.524 | 0.457 | 0.716 | 10.0 | 0 | -0.243 [-0.367; -0.119] |
| hibrida | 21 | 0.676 | 0.605 | 0.829 | 10.0 | 0 | -0.095 [-0.152; -0.038] |
| hibrida+reranker | 21 | 0.714 | 0.700 | 0.795 | 10.0 | 1 | referência |
| final | 21 | 0.714 | 0.700 | 0.795 | 10.0 | 1 | +0.000 [+0.000; +0.000] |
| sem filtros | 9 | 0.578 | 0.544 | 0.676 | 10.0 | 0 | -0.122 [-0.267; +0.000] (atual nessas 9: 0.667) |

`final` = top-10 do reranker só com score ≥ limiar (o que chega ao LLM); a precisão é medida sobre as evidências entregues, e a coluna Evidências mostra quantas são em média. `sem filtros` só inclui perguntas em que a T11 extraiu filtros, comparadas ao atual nessas mesmas perguntas.

**Atenção:** 1 reviews ainda sem rótulo (contadas como não relevantes). Preencha `eval/t28_rotulagem.xlsx` e rode `--importar-rotulos`.

## Recusa

| Política | Recusa correta | Recusou quando devia | Respondeu quando devia | Erros | Perguntas com erro |
|---|---:|---:|---:|---:|---|
| sem guardrail | 0.97 | 0.88 | 1.00 | 1 | P09 |
| limiar 0.0 | 0.97 | 0.88 | 1.00 | 1 | P09 |
| limiar 0.2 | 0.97 | 0.88 | 1.00 | 1 | P09 |
| limiar 0.3 | 0.97 | 0.88 | 1.00 | 1 | P09 |
| limiar 0.4 | 0.97 | 0.88 | 1.00 | 1 | P09 |
| limiar 0.5 | 0.97 | 0.88 | 1.00 | 1 | P09 |
| limiar 0.6 | 0.59 | 1.00 | 0.43 | 12 | P02, P03, P05, P10, P12, P13, P14, P15, P22, P25, P28, P30 |
| limiar 0.7 | 0.41 | 1.00 | 0.19 | 17 | P02, P03, P04, P05, P10, P12, P13, P14, P15, P21, P22, P23, P24, P25, P28, P29, P30 |
| limiar 0.8 | 0.28 | 1.00 | 0.00 | 21 | P01, P02, P03, P04, P05, P06, P10, P11, P12, P13, P14, P15, P21, P22, P23, P24, P25, P26, P28, P29, P30 |

## Custo da busca (segundos por pergunta, média)

vetorial 0.995 · BM25 0.075 · busca completa com reranker do pool de 40 0.725

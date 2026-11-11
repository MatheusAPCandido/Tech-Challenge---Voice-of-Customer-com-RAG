# T07 — Comparação de modelos de embedding

Busca só vetorial, base completa (55 consultas com rótulo de tema, top-10).

| Modelo | Dim. | Indexar base (s) | Consultas/s | Índice (MB) | P@5 | P@10 | MRR@10 | Diferença P@10 vs referência (IC95%) |
|---|---|---|---|---|---|---|---|---|
| `intfloat/multilingual-e5-base` | 768 | 8.0 | 307.2 | 107.2 | 0.749 | 0.709 | 0.881 | referência |
| `intfloat/multilingual-e5-small` | 384 | 7.0 | 993.4 | 53.6 | 0.658 | 0.625 | 0.838 | -0.084 [-0.147; -0.020] |
| `BAAI/bge-m3` | 1024 | 15.8 | 646.8 | 143.0 | 0.847 | 0.851 | 0.940 | +0.142 [+0.078; +0.209] |

Relevância: a review cita algum dos temas esperados para a consulta (rótulos da T08).
IC95%: bootstrap pareado sobre as consultas; se o intervalo inclui 0, a diferença não é conclusiva.

## Rotulagem manual (assuntos sem tema próprio)

Top-5 de cada modelo em 10 consultas, rotulado às cegas (sim/nao).

| Modelo | P@5 manual | Diferença vs referência (IC95%) |
|---|---|---|
| `BAAI/bge-m3` | 0.680 | +0.160 [-0.060; +0.380] |
| `intfloat/multilingual-e5-base` | 0.520 | referência |
| `intfloat/multilingual-e5-small` | 0.520 | -0.000 [-0.140; +0.140] |

# Como a equipe trabalha neste repositório

## Combinados

1. **Ninguém envia direto para a `main`.** Toda tarefa vai numa branch.
2. **Todo Pull Request é aprovado por outra pessoa** antes do merge.
3. **Antes de começar uma tarefa, atualize a `main`** (Fetch origin → Pull).
4. **Um notebook por pessoa por vez.** Lógica reutilizável vai para `src/`.
5. **Nunca subir dados nem chaves.** O `.gitignore` já bloqueia `data/` e `.env`.

## Fluxo de uma tarefa

1. Pull da `main`.
2. Criar branch com o ID da planilha: `T05-preparacao-base`.
3. Trabalhar, commit e push da branch.
4. Abrir Pull Request descrevendo o que mudou.
5. Outro integrante revisa (aba *Files changed*) e aprova.
6. Merge e apagar a branch.

## Contrato entre os módulos

O arquivo `src/schema.py` define o formato dos dados trocados entre as partes.
**Não mude esses campos sem combinar com o grupo**, porque os outros módulos dependem deles.

| Função | Dono | Entrada | Saída |
| --- | --- | --- | --- |
| `prepare.preparar_base()` | Pessoa A | CSVs em `data/` | `data/reviews_docs.parquet` |
| `index.construir_indice()` | Pessoa A | parquet | índice em `chroma_db/` |
| `retrieve.buscar()` | Pessoa B | pergunta, filtros | `list[Evidencia]` |
| `guardrails.avaliar_evidencias()` | Pessoa B | `list[Evidencia]` | `Veredito` |
| `generate.responder()` | Pessoa C | pergunta, evidências | `Resposta` |
| `pipeline.perguntar()` | Todos | pergunta | `Resposta` |

## Padrão de mensagens de commit

`T05: limpeza dos textos e tratamento de ausentes`

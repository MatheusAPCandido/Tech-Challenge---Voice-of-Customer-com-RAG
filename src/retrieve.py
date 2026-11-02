"""Transformação da pergunta e recuperação (Pessoa B).

Tarefas: T11 (transformação), T12 (busca híbrida + RRF), T13 (filtros), T14 (reranker).
"""

from __future__ import annotations

from typing import Optional

from src.config import TOP_K_CANDIDATOS, TOP_K_FINAL, USE_MOCK
from src.schema import ConsultaProcessada, Evidencia

# ---------------------------------------------------------------------------
# Mock — permite que a Pessoa C desenvolva a interface antes da busca real.
# ---------------------------------------------------------------------------
_MOCK = [
    Evidencia("mock-001", "Produto não chegou. Comprei há 20 dias e até agora nada.", 0.91,
              review_score=1, categoria="moveis_decoracao", data_compra="2018-03", dias_atraso=12.0, estado="CE"),
    Evidencia("mock-002", "Recebi apenas um dos dois itens do pedido e ninguém responde.", 0.87,
              review_score=1, categoria="cama_mesa_banho", data_compra="2018-05", dias_atraso=3.0, estado="SP"),
    Evidencia("mock-003", "Entrega muito atrasada, mas o produto é bom.", 0.84,
              review_score=3, categoria="esporte_lazer", data_compra="2017-11", dias_atraso=8.0, estado="RJ"),
    Evidencia("mock-004", "Chegou antes do prazo e bem embalado. Recomendo!", 0.62,
              review_score=5, categoria="informatica_acessorios", data_compra="2018-01", dias_atraso=-5.0, estado="MG"),
    Evidencia("mock-005", "Produto veio com defeito e a troca demorou demais.", 0.80,
              review_score=2, categoria="eletroportateis", data_compra="2018-07", dias_atraso=0.0, estado="BA"),
]


def processar_pergunta(pergunta: str) -> ConsultaProcessada:
    """T11: reescreve a pergunta, extrai filtros e classifica o tipo.

    TODO: chamada curta ao LLM com saída JSON:
      {"consulta": "...", "filtros": {...}, "tipo": "pontual|agregada|fora_de_escopo"}
    Ex.: "clientes insatisfeitos" → filtros = {"review_score_max": 2}
    """
    return ConsultaProcessada(pergunta_original=pergunta, consulta=pergunta)


def buscar(pergunta: str, filtros: Optional[dict] = None, k: int = TOP_K_FINAL) -> list[Evidencia]:
    """Contrato principal: devolve as k evidências mais relevantes, ordenadas por score.

    TODO:
      T12 — busca vetorial (top TOP_K_CANDIDATOS) + BM25 (top TOP_K_CANDIDATOS) + RRF
      T13 — aplicar filtros de metadados (ver schema.FILTROS_SUPORTADOS)
      T14 — reordenar com cross-encoder e cortar em k
    """
    if USE_MOCK:
        return sorted(_MOCK, key=lambda e: e.score, reverse=True)[:k]
    _ = (filtros, TOP_K_CANDIDATOS)
    raise NotImplementedError("Implementar nas tarefas T12–T14")

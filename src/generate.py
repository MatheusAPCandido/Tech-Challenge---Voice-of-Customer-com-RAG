"""Construção do contexto, geração e validação de citações (Pessoa C).

Tarefas: T19 (prompt + contexto), T21 (geração), T22 (validação), T23/T24 (agregadas).
"""

from __future__ import annotations

import re

from src.config import USE_MOCK
from src.schema import Evidencia, Resposta

PROMPT_SISTEMA = """Você é um analista de Voice of Customer de um e-commerce.
Responda APENAS com base nas avaliações fornecidas entre <avaliacoes>.
Regras:
1. Cite a avaliação que sustenta cada afirmação no formato [R1], [R2]...
2. Não invente informações nem generalize além do que as avaliações mostram.
3. Ao quantificar, use "X de N avaliações recuperadas", nunca "a maioria dos clientes".
4. Se as avaliações não permitirem responder, diga: "As avaliações recuperadas não permitem responder a esta pergunta."
Responda em português."""


def montar_contexto(evidencias: list[Evidencia]) -> tuple[str, dict[str, str]]:
    """Numera as evidências como [R1], [R2]... e devolve o texto e o mapa R# → review_id."""
    linhas, mapa = [], {}
    for i, e in enumerate(evidencias, 1):
        rid = f"R{i}"
        mapa[rid] = e.review_id
        meta = f"nota={e.review_score} | categoria={e.categoria} | compra={e.data_compra} | atraso={e.dias_atraso}"
        linhas.append(f"[{rid}] {meta}\n\"{e.texto}\"")
    return "<avaliacoes>\n" + "\n\n".join(linhas) + "\n</avaliacoes>", mapa


def validar_citacoes(texto: str, mapa: dict[str, str]) -> list[str]:
    """T22: mantém só as citações [R#] que existem no contexto e devolve os review_ids."""
    citadas = dict.fromkeys(re.findall(r"\[(R\d+)\]", texto))
    return [mapa[r] for r in citadas if r in mapa]


def chamar_llm(sistema: str, usuario: str) -> str:
    """T21: chamada ao LLM escolhido (LLM_PROVIDER / LLM_MODEL no .env), temperatura baixa."""
    raise NotImplementedError("Implementar na tarefa T21")


def responder(pergunta: str, evidencias: list[Evidencia]) -> Resposta:
    """Contrato principal: gera a resposta fundamentada nas evidências."""
    contexto, mapa = montar_contexto(evidencias)
    if USE_MOCK:
        texto = (
            "Resposta de exemplo (modo mock). Os clientes relatam atrasos na entrega [R1][R3] "
            "e pedidos entregues de forma parcial [R2]."
        )
    else:
        texto = chamar_llm(PROMPT_SISTEMA, f"{contexto}\n\nPergunta: {pergunta}")
    citacoes = validar_citacoes(texto, mapa)
    return Resposta(texto=texto, citacoes=citacoes, evidencias=evidencias)

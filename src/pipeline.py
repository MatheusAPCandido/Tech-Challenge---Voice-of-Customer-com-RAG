"""Orquestra o fluxo completo: pergunta → resposta com evidências (T27)."""

from __future__ import annotations

from src.agregar import bloco_contagens
from src.generate import responder
from src.guardrails import avaliar_evidencias, filtrar_relevantes
from src.retrieve import buscar, processar_pergunta
from src.schema import Resposta


def perguntar(pergunta: str, filtros_manuais: dict | None = None) -> Resposta:
    consulta = processar_pergunta(pergunta)
    filtros = {**consulta.filtros, **(filtros_manuais or {})}

    if consulta.tipo == "fora_de_escopo":
        veredito = avaliar_evidencias(consulta, [])
        return Resposta(texto=veredito.mensagem, recusada=True, motivo_recusa=veredito.motivo, confianca="baixa")

    evidencias = buscar(consulta.consulta, filtros)
    veredito = avaliar_evidencias(consulta, evidencias)
    if not veredito.pode_responder:
        return Resposta(texto=veredito.mensagem, evidencias=evidencias, recusada=True,
                        motivo_recusa=veredito.motivo, confianca="baixa")

    resposta = responder(pergunta, filtrar_relevantes(evidencias))
    if veredito.motivo == "evidencia_insuficiente":
        resposta.texto = f"{veredito.mensagem}\n\n{resposta.texto}"
        resposta.confianca = "baixa"
    if consulta.tipo == "agregada" and not resposta.recusada:  # T24: contagens antes dos exemplos
        bloco = bloco_contagens(filtros)
        if bloco:
            resposta.texto = f"{bloco}\n\n{resposta.texto}"        
    return resposta


if __name__ == "__main__":
    r = perguntar("O que os clientes relatam sobre problemas de entrega?")
    print(r.texto)
    print("Citações:", r.citacoes)

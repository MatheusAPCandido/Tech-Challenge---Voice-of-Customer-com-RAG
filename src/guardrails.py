"""Tratamento de perguntas sem evidência suficiente (Pessoa B).

Tarefas: T15 (calibrar limiar), T16 (três situações de recusa).
"""

from __future__ import annotations

from src.config import LIMIAR_RELEVANCIA, MIN_EVIDENCIAS
from src.schema import ConsultaProcessada, Evidencia, Veredito

MSG_FORA_DE_ESCOPO = "Esta pergunta está fora do escopo das avaliações de clientes."
MSG_SEM_EVIDENCIA = "Não encontrei avaliações que tratem desse tema."
MSG_INSUFICIENTE = "Encontrei poucas avaliações sobre esse tema; a resposta abaixo é parcial."


def avaliar_evidencias(consulta: ConsultaProcessada, evidencias: list[Evidencia]) -> Veredito:
    """Decide se o pipeline pode gerar uma resposta.

    TODO (T15): calibrar LIMIAR_RELEVANCIA com pares rotulados e documentar no README.
    """
    if consulta.tipo == "fora_de_escopo":
        return Veredito(False, "fora_de_escopo", MSG_FORA_DE_ESCOPO)

    relevantes = [e for e in evidencias if e.score >= LIMIAR_RELEVANCIA]
    if not relevantes:
        return Veredito(False, "sem_evidencia", MSG_SEM_EVIDENCIA)
    if len(relevantes) < MIN_EVIDENCIAS:
        return Veredito(True, "evidencia_insuficiente", MSG_INSUFICIENTE)
    return Veredito(True, "ok")


def filtrar_relevantes(evidencias: list[Evidencia]) -> list[Evidencia]:
    return [e for e in evidencias if e.score >= LIMIAR_RELEVANCIA]

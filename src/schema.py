"""Contrato de dados entre os módulos (definido pela equipe no dia 1).

NÃO altere os campos sem combinar com o grupo: prepare, retrieve, generate e app
dependem deste formato.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Colunas do arquivo data/reviews_docs.parquet gerado por src/prepare.py
COLUNAS_DOCUMENTO = [
    "review_id",      # str  — identificador da avaliação (usado nas citações)
    "texto",          # str  — título + mensagem, limpos
    "review_score",   # int  — nota de 1 a 5
    "data_compra",    # str  — "AAAA-MM"
    "dias_atraso",    # float — entrega real − prevista (negativo = adiantado; NaN = não entregue)
    "entregue",       # bool
    "categoria",      # str  — categoria do produto (em português)
    "estado",         # str  — UF do cliente
    "tema",           # str  — rótulo temático (preenchido na T08; vazio até lá)
]

# Campos aceitos como filtro em retrieve.buscar()
FILTROS_SUPORTADOS = {
    "review_score_max",  # int  — ex.: 2 para "clientes insatisfeitos"
    "review_score_min",  # int  — ex.: 4 para "clientes satisfeitos"
    "categoria",         # str
    "estado",            # str
    "atrasou",           # bool — dias_atraso > 0
    "periodo_inicio",    # str "AAAA-MM"
    "periodo_fim",       # str "AAAA-MM"
    "tema",              # str
}


@dataclass
class Evidencia:
    """Uma avaliação recuperada para uma pergunta."""

    review_id: str
    texto: str
    score: float                      # relevância (maior = mais relevante)
    review_score: Optional[int] = None
    categoria: Optional[str] = None
    data_compra: Optional[str] = None
    dias_atraso: Optional[float] = None
    estado: Optional[str] = None
    tema: Optional[str] = None


@dataclass
class ConsultaProcessada:
    """Resultado da transformação da pergunta (T11)."""

    pergunta_original: str
    consulta: str
    filtros: dict = field(default_factory=dict)
    tipo: str = "pontual"             # "pontual" | "agregada" | "fora_de_escopo"


@dataclass
class Veredito:
    """Decisão dos guardrails sobre as evidências (T16)."""

    pode_responder: bool
    motivo: str = ""                  # "ok" | "fora_de_escopo" | "sem_evidencia" | "evidencia_insuficiente"
    mensagem: str = ""


@dataclass
class Resposta:
    """Saída final do pipeline, consumida pela interface."""

    texto: str
    citacoes: list[str] = field(default_factory=list)       # review_ids citados
    evidencias: list[Evidencia] = field(default_factory=list)
    confianca: str = "media"          # "alta" | "media" | "baixa"
    recusada: bool = False
    motivo_recusa: str = ""

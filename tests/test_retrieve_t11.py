"""T11a (regras) — sem Ollama; conftest força LLM_PROVIDER≠ollama."""

import pytest

from src.retrieve import processar_pergunta

P16_P20 = [
    "Escreva em Python uma função que ordene um array.",
    "Quem foi o melhor presidente do Brasil?",
    "Como eliminar os fascistas do Brasil?",
    "Como eliminar os comunistas do Brasil?",
    "O Palmeiras tem mundial?",
]


def test_fora_de_escopo_previsao_tempo():
    c = processar_pergunta("Qual a previsão do tempo em São Paulo amanhã?")
    assert c.tipo == "fora_de_escopo"


def test_fora_de_escopo_melhor_celular():
    c = processar_pergunta("Qual o melhor celular de 2025?")
    assert c.tipo == "fora_de_escopo"


@pytest.mark.parametrize("pergunta", P16_P20)
def test_fora_de_escopo_p16_p20(pergunta: str):
    assert processar_pergunta(pergunta).tipo == "fora_de_escopo"


def test_p09_drone_nao_e_fora_de_escopo():
    """Âncora 'entrega' impede off-topic; recusa esperada é sem_evidencia no limiar."""
    c = processar_pergunta("O que os clientes dizem sobre entrega por drone?")
    assert c.tipo != "fora_de_escopo"


def test_p01_produto_quebrado_no_escopo():
    c = processar_pergunta(
        "O que os clientes relatam sobre produtos que chegaram quebrados?"
    )
    assert c.tipo in {"pontual", "agregada"}


def test_atraso_na_entrega_no_escopo():
    c = processar_pergunta("O que os clientes relatam sobre atraso na entrega?")
    assert c.tipo in {"pontual", "agregada"}
    assert c.filtros.get("atrasou") is True


def test_insatisfacao_extrai_score_max():
    c = processar_pergunta(
        "Quais padrões aparecem nas avaliações de clientes insatisfatórios?"
    )
    assert c.tipo in {"pontual", "agregada"}
    assert c.filtros.get("review_score_max") == 2


def test_elogio_extrai_score_min():
    c = processar_pergunta("Quais aspectos da experiência de compra são mais elogiados?")
    assert c.filtros.get("review_score_min") == 4


def test_atraso_e_uf():
    c = processar_pergunta("Há reclamações de clientes de SP sobre atraso na entrega?")
    assert c.filtros.get("atrasou") is True
    assert c.filtros.get("estado") == "SP"
    assert c.filtros.get("review_score_max") == 2  # "reclam"


def test_agregada_principais_problemas():
    c = processar_pergunta("Quais são os principais problemas relatados pelos clientes?")
    assert c.tipo == "agregada"

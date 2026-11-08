"""T11a (regras) — sem Ollama; conftest força LLM_PROVIDER≠ollama."""

from src.retrieve import processar_pergunta


def test_fora_de_escopo_previsao_tempo():
    c = processar_pergunta("Qual a previsão do tempo em São Paulo amanhã?")
    assert c.tipo == "fora_de_escopo"


def test_fora_de_escopo_melhor_celular():
    c = processar_pergunta("Qual o melhor celular de 2025?")
    assert c.tipo == "fora_de_escopo"


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

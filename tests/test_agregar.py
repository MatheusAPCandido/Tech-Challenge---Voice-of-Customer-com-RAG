"""T24 — contagens por tema. Dados falsos: não precisa da gold nem do índice."""

from __future__ import annotations

import pandas as pd
import pytest

import src.agregar as ag

# Oito avaliações falsas que cobrem notas, atraso (inclusive pedido não entregue), estado,
# período, categoria e os dois temas genéricos.
DF = pd.DataFrame({
    "review_score": [1, 1, 2, 2, 5, 4, 5, 3],
    "data_compra": ["2018-01", "2018-02", "2018-07", "2017-11", "2018-03", "2017-12", "2018-08", "2018-05"],
    "dias_atraso": [12.0, None, 3.0, -1.0, -2.0, 0.0, -4.0, 8.0],
    "categoria": ["moveis_decoracao"] * 4 + ["informatica_acessorios"] * 4,
    "estado": ["SP", "SP", "RJ", "SP", "MG", "SP", "BA", "RJ"],
    "tema": ["nao_recebido", "nao_recebido", "atraso_entrega", "produto_defeito",
             "entrega_rapida", "avaliacao_geral", "outros", "atraso_entrega"],
})


@pytest.fixture
def gold_falsa(monkeypatch):
    """Troca a leitura do parquet pela tabela falsa acima."""
    monkeypatch.setattr(ag, "_carregar", lambda: DF)


def test_conta_dentro_do_recorte_de_nota(gold_falsa):
    total, contagem = ag.contar_temas({"review_score_max": 2})
    assert total == 4
    assert contagem.to_dict() == {"nao_recebido": 2, "atraso_entrega": 1, "produto_defeito": 1}


def test_pedido_nao_entregue_nao_conta_como_atrasado(gold_falsa):
    total, _ = ag.contar_temas({"atrasou": True})
    assert total == 3  # 12, 3 e 8 dias; o atraso nulo (não entregue) fica de fora, como no índice


def test_periodo_e_estado_como_na_busca(gold_falsa):
    total, contagem = ag.contar_temas({"estado": "sp", "periodo_inicio": "2018-01", "periodo_fim": "2018-06"})
    assert total == 2  # "sp" vira "SP" no saneamento; só jan e fev/2018 entram
    assert contagem.to_dict() == {"nao_recebido": 2}


def test_filtros_passam_pelo_mesmo_saneamento_da_busca(gold_falsa):
    total, _ = ag.contar_temas({"categoria": "  moveis_decoracao ", "review_score_min": "abc"})
    assert total == 4  # espaços removidos; nota inválida descartada


def test_bloco_descreve_recorte_e_deixa_genericos_fora_do_ranking(gold_falsa):
    bloco = ag.bloco_contagens({})
    assert "entre 8 avaliações com texto (todas as notas · todos os períodos)" in bloco
    assert "- Produto não recebido — 2 (25,0%)" in bloco
    assert "Opinião geral, sem assunto específico" not in bloco  # genérico fora do ranking...
    assert "- Sem assunto específico (opinião geral ou outros) — 2 (25,0%)" in bloco  # ...mas reportado
    assert bloco.endswith(ag.MARCADOR_EXEMPLOS)


def test_sem_gold_o_bloco_fica_vazio(monkeypatch):
    def gold_ausente():
        raise FileNotFoundError("data/gold/reviews_completo.parquet")

    monkeypatch.setattr(ag, "_carregar", gold_ausente)
    assert ag.bloco_contagens({}) == ""


def test_recorte_vazio_ou_filtrado_por_tema_nao_gera_bloco(gold_falsa):
    assert ag.bloco_contagens({"estado": "AC"}) == ""          # ninguém do AC na tabela falsa
    assert ag.bloco_contagens({"tema": "nao_recebido"}) == ""  # ranking dentro de um tema só


@pytest.mark.parametrize("filtros, esperado", [
    ({}, "todas as notas · todos os períodos"),
    ({"review_score_max": 2}, "notas 1 e 2 · todos os períodos"),
    ({"review_score_min": 4}, "notas 4 e 5 · todos os períodos"),
    ({"review_score_min": 3}, "notas de 3 a 5 · todos os períodos"),
    ({"atrasou": True, "estado": "SP", "periodo_inicio": "2018-06", "periodo_fim": "2018-08"},
     "todas as notas · entregas atrasadas · estado SP · compras de 2018-06 a 2018-08"),
], ids=["sem-filtro", "notas-baixas", "notas-altas", "nota-3-ou-mais", "varios-filtros"])
def test_descrever_recorte(filtros, esperado):
    assert ag.descrever_recorte(filtros) == esperado


@pytest.mark.integration
@pytest.mark.skipif(not ag.GOLD_COMPLETO.exists(), reason="gold ausente: rode python -m src.prepare")
def test_contagem_bate_com_resumo_temas_da_t08():
    resumo = pd.read_parquet(ag.GOLD_COMPLETO.parent / "resumo_temas.parquet")
    n_principal = resumo.set_index("tema")["n_principal"]
    total, contagem = ag.contar_temas({})
    assert total == int(n_principal.sum())
    assert contagem.sort_index().to_dict() == n_principal.sort_index().to_dict()

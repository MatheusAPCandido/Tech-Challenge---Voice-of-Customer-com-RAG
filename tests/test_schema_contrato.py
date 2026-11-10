"""Congela o contrato do dia 1 em src/schema.py."""

from src.schema import (
    COLUNAS_DOCUMENTO,
    FILTROS_SUPORTADOS,
    ConsultaProcessada,
    Evidencia,
    Veredito,
)


def test_colunas_documento_esperadas():
    assert COLUNAS_DOCUMENTO == [
        "review_id",
        "texto",
        "review_score",
        "data_compra",
        "dias_atraso",
        "entregue",
        "categoria",
        "estado",
        "tema",
    ]


def test_filtros_suportados_esperados():
    assert FILTROS_SUPORTADOS == {
        "review_score_max",
        "review_score_min",
        "categoria",
        "estado",
        "atrasou",
        "periodo_inicio",
        "periodo_fim",
        "tema",
    }


def test_consulta_processada_defaults():
    c = ConsultaProcessada(pergunta_original="x", consulta="x")
    assert c.filtros == {}
    assert c.tipo == "pontual"


def test_veredito_defaults():
    v = Veredito(pode_responder=True)
    assert v.motivo == ""
    assert v.mensagem == ""


def test_evidencia_extras_t15_default_none():
    e = Evidencia(review_id="r1", texto="ok", score=0.5)
    assert e.score_rrf is None
    assert e.score_bm25 is None
    assert e.score_vetorial is None
    assert e.rank_bm25 is None
    assert e.rank_vetorial is None

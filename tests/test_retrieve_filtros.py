"""Sanitização e where de filtros (T13) — sem Chroma."""

from src.retrieve import (
    _MOCK,
    _montar_where,
    _passar_filtro_mock,
    _sanitizar_filtros,
)


def test_sanitizar_descarta_chave_invalida():
    assert _sanitizar_filtros({"foo": 1, "estado": "sp"}) == {"estado": "SP"}


def test_sanitizar_score_e_periodo():
    out = _sanitizar_filtros(
        {
            "review_score_min": "4",
            "review_score_max": 5,
            "periodo_inicio": "2018-01",
            "periodo_fim": "2018-03",
            "atrasou": "sim",
        }
    )
    assert out == {
        "review_score_min": 4,
        "review_score_max": 5,
        "periodo_inicio": "2018-01",
        "periodo_fim": "2018-03",
        "atrasou": True,
    }


def test_sanitizar_min_maior_que_max_remove_par():
    assert _sanitizar_filtros({"review_score_min": 5, "review_score_max": 2}) == {}


def test_sanitizar_score_fora_da_faixa():
    assert _sanitizar_filtros({"review_score_min": 0, "review_score_max": 9}) == {}


def test_sanitizar_periodo_invalido():
    assert _sanitizar_filtros({"periodo_inicio": "2018-1", "periodo_fim": "abc"}) == {}


def test_montar_where_vazio():
    assert _montar_where({}) is None


def test_montar_where_clausula_unica():
    assert _montar_where({"estado": "SP"}) == {"estado": {"$eq": "SP"}}


def test_montar_where_periodo_e_and():
    where = _montar_where(
        {
            "periodo_inicio": "2018-06",
            "periodo_fim": "2018-08",
            "review_score_max": 2,
        }
    )
    assert where == {
        "$and": [
            {"review_score": {"$lte": 2}},
            {"ano_mes": {"$gte": 201806}},
            {"ano_mes": {"$lte": 201808}},
        ]
    }


def test_passar_filtro_mock_score_e_atraso():
    # _MOCK[0]: score=1, dias_atraso=12 (atrasou)
    e = _MOCK[0]
    assert _passar_filtro_mock(e, {"review_score_max": 2}) is True
    assert _passar_filtro_mock(e, {"review_score_min": 4}) is False
    assert _passar_filtro_mock(e, {"atrasou": True}) is True
    assert _passar_filtro_mock(e, {"atrasou": False}) is False


def test_passar_filtro_mock_periodo():
    e = _MOCK[0]  # data_compra=2018-03
    assert _passar_filtro_mock(e, {"periodo_inicio": "2018-01", "periodo_fim": "2018-03"})
    assert not _passar_filtro_mock(e, {"periodo_inicio": "2018-04"})

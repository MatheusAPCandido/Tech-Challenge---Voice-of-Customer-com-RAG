"""RRF puro — sem índice."""

from src.retrieve import RRF_K, _fundir_rrf


def test_fundir_rrf_lista_vazia():
    assert _fundir_rrf([], k=10) == []
    assert _fundir_rrf([[], []], k=10) == []


def test_fundir_rrf_id_no_topo_das_duas_listas_ganha():
    out = _fundir_rrf([["a", "b"], ["a", "c"]], k=10)
    ids = [rid for rid, _ in out]
    assert ids[0] == "a"
    assert set(ids) == {"a", "b", "c"}


def test_fundir_rrf_rank_1_based_score_maximo():
    """Só uma lista: 1º lugar → score normalizado = 1.0."""
    out = _fundir_rrf([["x", "y"]], k=10)
    assert out[0] == ("x", 1.0)
    # 2º: (1/(K+2)) / (1/(K+1))
    esperado_y = (1.0 / (RRF_K + 2)) / (1.0 / (RRF_K + 1))
    assert out[1][0] == "y"
    assert abs(out[1][1] - esperado_y) < 1e-9


def test_fundir_rrf_empate_por_review_id():
    # Mesmo rank em listas distintas → empate de score; desempate por id asc.
    out = _fundir_rrf([["b"], ["a"]], k=10)
    assert [rid for rid, _ in out] == ["a", "b"]
    assert abs(out[0][1] - out[1][1]) < 1e-12


def test_fundir_rrf_corta_em_k():
    ranking = [f"id{i}" for i in range(20)]
    out = _fundir_rrf([ranking], k=5)
    assert len(out) == 5

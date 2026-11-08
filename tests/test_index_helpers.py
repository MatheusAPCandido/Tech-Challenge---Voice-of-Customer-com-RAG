"""Helpers puros de src/index.py."""

from src.index import ano_mes, tokenizar


def test_ano_mes():
    assert ano_mes("2018-03") == 201803
    assert ano_mes("2017-12") == 201712


def test_tokenizar_remove_acentos_e_stopwords():
    tokens = tokenizar("O produto chegou com atraso em São Paulo")
    assert "produto" in tokens
    assert "chegou" in tokens
    assert "atraso" in tokens
    assert "paulo" in tokens
    # "são"/"sao" e artigos/preposições estão em STOPWORDS
    assert "sao" not in tokens
    assert "o" not in tokens
    assert "com" not in tokens
    assert "em" not in tokens


def test_tokenizar_preserva_negacoes():
    """'não' / 'nunca' ficam fora de STOPWORDS de propósito."""
    tokens = tokenizar("não chegou e nunca recebi")
    assert "nao" in tokens
    assert "nunca" in tokens
    assert "chegou" in tokens
    assert "recebi" in tokens

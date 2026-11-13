"""T24 — o pipeline acrescenta as contagens só às perguntas agregadas respondidas."""

from __future__ import annotations

import pytest

from src.pipeline import perguntar
from src.schema import ConsultaProcessada, Evidencia, Resposta

BLOCO = "BLOCO DE CONTAGENS"


@pytest.fixture(autouse=True)
def _ambiente(monkeypatch):
    monkeypatch.setattr("src.guardrails.LIMIAR_RELEVANCIA", 0.5)
    monkeypatch.setattr("src.guardrails.MIN_EVIDENCIAS", 3)
    # Troca o nome que o pipeline importou (src.pipeline.bloco_contagens),
    # não o original em src.agregar: é esse que o perguntar() chama.
    monkeypatch.setattr("src.pipeline.bloco_contagens", lambda filtros: BLOCO)


def _consulta(tipo: str, filtros: dict | None = None) -> ConsultaProcessada:
    return ConsultaProcessada(pergunta_original="q", consulta="q", tipo=tipo, filtros=filtros or {})


def _tres_evidencias(*_a, **_k) -> list[Evidencia]:
    return [Evidencia(review_id=r, texto=f"texto {r}", score=0.9) for r in ("a", "b", "c")]


def _responder_ok(pergunta, evidencias) -> Resposta:
    return Resposta(texto="exemplos [R1]", citacoes=["a"], evidencias=evidencias)


def test_agregada_recebe_bloco_antes_dos_exemplos(monkeypatch):
    monkeypatch.setattr("src.pipeline.processar_pergunta", lambda q: _consulta("agregada"))
    monkeypatch.setattr("src.pipeline.buscar", _tres_evidencias)
    monkeypatch.setattr("src.pipeline.responder", _responder_ok)
    r = perguntar("Quais assuntos os clientes mais comentam?")
    assert r.texto == f"{BLOCO}\n\nexemplos [R1]"


def test_pontual_nao_recebe_bloco(monkeypatch):
    monkeypatch.setattr("src.pipeline.processar_pergunta", lambda q: _consulta("pontual"))
    monkeypatch.setattr("src.pipeline.buscar", _tres_evidencias)
    monkeypatch.setattr("src.pipeline.responder", _responder_ok)
    assert perguntar("O que dizem sobre a entrega?").texto == "exemplos [R1]"


def test_agregada_recusada_pelo_gerador_nao_recebe_bloco(monkeypatch):
    monkeypatch.setattr("src.pipeline.processar_pergunta", lambda q: _consulta("agregada"))
    monkeypatch.setattr("src.pipeline.buscar", _tres_evidencias)
    monkeypatch.setattr(
        "src.pipeline.responder",
        lambda p, e: Resposta(texto="recusa", recusada=True, motivo_recusa="sem_evidencia"),
    )
    assert perguntar("Quais assuntos os clientes mais comentam?").texto == "recusa"


def test_busca_e_contagem_recebem_os_mesmos_filtros(monkeypatch):
    usados = {}

    def _buscar(consulta, filtros):
        usados["busca"] = dict(filtros)
        return _tres_evidencias()

    def _bloco(filtros):
        usados["contagem"] = dict(filtros)
        return BLOCO

    monkeypatch.setattr("src.pipeline.processar_pergunta",
                        lambda q: _consulta("agregada", {"review_score_max": 2}))
    monkeypatch.setattr("src.pipeline.buscar", _buscar)
    monkeypatch.setattr("src.pipeline.responder", _responder_ok)
    monkeypatch.setattr("src.pipeline.bloco_contagens", _bloco)
    perguntar("Quais padrões aparecem nas avaliações de clientes insatisfeitos?", {"atrasou": True})
    assert usados["busca"] == usados["contagem"] == {"review_score_max": 2, "atrasou": True}

"""T21 — roteamento da geração Gemini → Ollama (sem rede: geradores e sleep falsos)."""

from __future__ import annotations

import sys
import types

import pytest

import src.generate as g
from src.schema import Evidencia


class ErroAPI(Exception):
    """Imita os erros do Google/Ollama: só importa o atributo code."""

    def __init__(self, code: int):
        super().__init__(f"{code} falso")
        self.code = code


@pytest.fixture
def ambiente(monkeypatch):
    """Gemini primário + reserva qwen2.5, sleep registrado e Ollama falso."""
    estado = {"chamadas": [], "esperas": [], "ollama_falha": False}
    monkeypatch.setattr(g, "GEN_PROVIDER", "gemini")
    monkeypatch.setattr(g, "GEN_MODEL", "gemini-teste")
    monkeypatch.setattr(g, "GEN_FALLBACK_MODEL", "qwen2.5")
    monkeypatch.setattr(g.time, "sleep", estado["esperas"].append)

    class FakeClient:
        def __init__(self, host, timeout):
            pass

        def chat(self, model, messages, format, options):
            estado["chamadas"].append(f"ollama/{model}")
            assert format == "json"
            assert "JSON" in messages[0]["content"]
            if estado["ollama_falha"]:
                raise ConnectionError("servidor fora (falso)")
            return {"message": {"content":
                '```json\n{"texto": "Atraso na entrega [R1][R2].", "pode_responder": true}\n```'}}

    monkeypatch.setitem(sys.modules, "ollama", types.SimpleNamespace(Client=FakeClient))
    return estado


def _gemini_falso(monkeypatch, estado, erro=None):
    def gerar(modelo, sistema, usuario):
        estado["chamadas"].append(f"gemini/{modelo}")
        if erro is not None:
            raise erro
        return g.SaidaLLM(texto="Resposta do Gemini [R1].", pode_responder=True)

    monkeypatch.setitem(g._GERADORES, "gemini", gerar)


GEMINI = "gemini/gemini-teste"
OLLAMA = "ollama/qwen2.5"


def test_gemini_ok_nao_chama_ollama(monkeypatch, ambiente):
    _gemini_falso(monkeypatch, ambiente)
    assert g.chamar_llm("s", "u").texto == "Resposta do Gemini [R1]."
    assert ambiente["chamadas"] == [GEMINI]


def test_503_repete_e_cai_no_ollama(monkeypatch, ambiente):
    _gemini_falso(monkeypatch, ambiente, ErroAPI(503))
    saida = g.chamar_llm("s", "u")
    assert ambiente["chamadas"] == [GEMINI] * 3 + [OLLAMA]
    assert ambiente["esperas"] == [5, 10]
    assert saida.texto == "Atraso na entrega [R1][R2]."


def test_429_e_transitorio(monkeypatch, ambiente):
    _gemini_falso(monkeypatch, ambiente, ErroAPI(429))
    g.chamar_llm("s", "u")
    assert ambiente["chamadas"] == [GEMINI] * 3 + [OLLAMA]


def test_404_nao_repete(monkeypatch, ambiente):
    _gemini_falso(monkeypatch, ambiente, ErroAPI(404))
    g.chamar_llm("s", "u")
    assert ambiente["chamadas"] == [GEMINI, OLLAMA]
    assert ambiente["esperas"] == []


def test_4xx_embrulhado_em_cause_nao_repete(monkeypatch, ambiente):
    """O LangChain faz raise ... from erro_do_google."""
    embrulhado = RuntimeError("erro do LangChain")
    embrulhado.__cause__ = ErroAPI(401)
    _gemini_falso(monkeypatch, ambiente, embrulhado)
    g.chamar_llm("s", "u")
    assert ambiente["chamadas"] == [GEMINI, OLLAMA]


def test_os_dois_falham(monkeypatch, ambiente):
    ambiente["ollama_falha"] = True
    _gemini_falso(monkeypatch, ambiente, ErroAPI(404))
    with pytest.raises(RuntimeError) as exc:
        g.chamar_llm("s", "u")
    assert GEMINI in str(exc.value) and OLLAMA in str(exc.value)
    assert ambiente["chamadas"] == [GEMINI, OLLAMA]  # ConnectionError não repete
    assert ambiente["esperas"] == []


def test_reserva_desligada(monkeypatch, ambiente):
    monkeypatch.setattr(g, "GEN_FALLBACK_MODEL", "")
    _gemini_falso(monkeypatch, ambiente, ErroAPI(404))
    with pytest.raises(RuntimeError):
        g.chamar_llm("s", "u")
    assert ambiente["chamadas"] == [GEMINI]


def test_sem_chave_vai_direto_ao_ollama(monkeypatch, ambiente):
    """Código real do Gemini: falha antes de chamar a API."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    g.chamar_llm("s", "u")
    assert ambiente["chamadas"] == [OLLAMA]
    assert ambiente["esperas"] == []


def test_sem_pacote_gemini_vai_direto_ao_ollama(monkeypatch, ambiente):
    monkeypatch.setenv("GOOGLE_API_KEY", "chave-falsa")
    monkeypatch.setitem(sys.modules, "langchain_google_genai", None)  # import falha
    g.chamar_llm("s", "u")
    assert ambiente["chamadas"] == [OLLAMA]
    assert ambiente["esperas"] == []


def test_provider_ollama_direto(monkeypatch, ambiente):
    monkeypatch.setattr(g, "GEN_PROVIDER", "ollama")
    monkeypatch.setattr(g, "GEN_MODEL", "qwen2.5")
    g.chamar_llm("s", "u")
    assert ambiente["chamadas"] == [OLLAMA]


def test_provider_invalido(monkeypatch, ambiente):
    monkeypatch.setattr(g, "GEN_PROVIDER", "openai")
    with pytest.raises(ValueError):
        g.chamar_llm("s", "u")


def test_responder_usa_reserva_e_valida_citacoes(monkeypatch, ambiente):
    monkeypatch.setattr(g, "GEN_MOCK", False)
    _gemini_falso(monkeypatch, ambiente, ErroAPI(503))
    evs = [
        Evidencia(review_id="id-a", texto="atrasou", score=0.9),
        Evidencia(review_id="id-b", texto="demorou", score=0.8),
    ]
    r = g.responder("Quais problemas de entrega?", evs)
    assert r.recusada is False
    assert r.citacoes == ["id-a", "id-b"]
    assert r.confianca == "media"

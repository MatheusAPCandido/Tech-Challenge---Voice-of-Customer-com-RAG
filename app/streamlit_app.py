"""Interface Streamlit com painel de evidências (Pessoa C — T20, T25).

Uso:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from src.config import GEN_MOCK, USE_MOCK  # noqa: E402
from src.pipeline import perguntar  # noqa: E402

st.set_page_config(page_title="Voice of Customer — Olist", layout="wide")
st.title("Voice of Customer Intelligence")
st.caption("Pergunte sobre as avaliações de clientes da Olist. Toda resposta mostra as avaliações que a sustentam.")
if USE_MOCK and GEN_MOCK:
    st.info("Modo mock ativo: avaliações e respostas são de exemplo (USE_MOCK=true, GEN_MOCK=true).")
elif USE_MOCK:
    st.info("Busca em modo mock: as avaliações são de exemplo (USE_MOCK=true); "
            "as respostas são geradas pelo LLM a partir delas (GEN_MOCK=false).")
elif GEN_MOCK:
    st.info("Geração em modo mock: avaliações reais, resposta de exemplo (GEN_MOCK=true).")

MOTIVOS_RECUSA = {
    "fora_de_escopo": "a pergunta está fora do escopo das avaliações de clientes",
    "sem_evidencia": "nenhuma avaliação recuperada sustenta uma resposta",
}


def _atraso(e) -> str:
    if e.dias_atraso is None:
        return "não informado"
    return f"{e.dias_atraso:.0f} dias" if e.dias_atraso > 0 else "no prazo"


def _mostrar_evidencia(i: int, e, marca: str = "") -> None:
    with st.expander(f"[R{i}] nota {e.review_score} · {e.categoria} {marca}"):
        st.write(e.texto)
        st.caption(
            f"id: {e.review_id} · compra: {e.data_compra} · atraso: {_atraso(e)} · "
            f"estado: {e.estado} · relevância: {e.score:.2f}"
        )

with st.sidebar:
    st.header("Filtros")
    notas = st.slider("Nota da avaliação", 1, 5, (1, 5))
    atrasou = st.checkbox("Somente entregas atrasadas")
    # TODO (T25): carregar categorias e estados reais da base
    categoria = st.text_input("Categoria (opcional)")

exemplos = [
    "Quais são os principais problemas relatados pelos clientes?",
    "O que os clientes relatam sobre problemas de entrega?",
    "O que os clientes mais elogiam?",
]
pergunta = st.text_input("Sua pergunta", placeholder=exemplos[0])
st.caption("Exemplos: " + " · ".join(exemplos))

if st.button("Perguntar", type="primary") and pergunta:
    filtros = {}
    if notas != (1, 5):
        filtros.update(review_score_min=notas[0], review_score_max=notas[1])
    if atrasou:
        filtros["atrasou"] = True
    if categoria:
        filtros["categoria"] = categoria

    try:
        with st.spinner("Buscando avaliações..."):
            resposta = perguntar(pergunta, filtros)
    except FileNotFoundError as erro:
        st.error("Índice de busca não encontrado. Gere o índice (`python -m src.index --amostra`) "
                 "ou use `USE_MOCK=true` no .env.")
        st.caption(str(erro))
        st.stop()
    except RuntimeError as erro:
        st.error("Não foi possível gerar a resposta: o LLM não respondeu. "
                 "Verifique a chave e o provedor no .env (GEN_PROVIDER, GEN_MODEL).")
        st.caption(str(erro)[:300])
        st.stop()

    col_resp, col_evid = st.columns([3, 2])
    with col_resp:
        st.subheader("Resposta")
        if resposta.recusada:
            st.warning(resposta.texto)
            motivo = MOTIVOS_RECUSA.get(resposta.motivo_recusa, resposta.motivo_recusa or "não informado")
            st.caption(f"Motivo da recusa: {motivo}")
        else:
            st.markdown(resposta.texto)
        st.caption(f"Confiança: {resposta.confianca}")

    with col_evid:
        citadas = set(resposta.citacoes)
        numeradas = list(enumerate(resposta.evidencias, 1))  # mesma numeração R# do contexto
        usadas = [(i, e) for i, e in numeradas if e.review_id in citadas]
        demais = [(i, e) for i, e in numeradas if e.review_id not in citadas]
        st.subheader(f"Evidências citadas ({len(usadas)})")
        for i, e in usadas:
            _mostrar_evidencia(i, e, "✅")
        if demais:
            st.subheader(f"Recuperadas e não citadas ({len(demais)})")
            for i, e in demais:
                _mostrar_evidencia(i, e)

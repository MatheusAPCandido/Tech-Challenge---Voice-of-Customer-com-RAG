"""Interface Streamlit com painel de evidências (Pessoa C — T20, T25).

Uso:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from src.config import USE_MOCK  # noqa: E402
from src.pipeline import perguntar  # noqa: E402

st.set_page_config(page_title="Voice of Customer — Olist", layout="wide")
st.title("Voice of Customer Intelligence")
st.caption("Pergunte sobre as avaliações de clientes da Olist. Toda resposta mostra as avaliações que a sustentam.")
if USE_MOCK:
    st.info("Modo mock ativo: as respostas usam dados de exemplo (USE_MOCK=true no .env).")

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

    with st.spinner("Buscando avaliações..."):
        resposta = perguntar(pergunta, filtros)

    col_resp, col_evid = st.columns([3, 2])
    with col_resp:
        st.subheader("Resposta")
        if resposta.recusada:
            st.warning(resposta.texto)
        else:
            st.markdown(resposta.texto)
        st.caption(f"Confiança: {resposta.confianca}")

    with col_evid:
        st.subheader(f"Evidências ({len(resposta.evidencias)})")
        citadas = set(resposta.citacoes)
        for i, e in enumerate(resposta.evidencias, 1):
            marca = "✅ citada" if e.review_id in citadas else ""
            with st.expander(f"[R{i}] nota {e.review_score} · {e.categoria} {marca}"):
                st.write(e.texto)
                st.caption(
                    f"id: {e.review_id} · compra: {e.data_compra} · atraso: {e.dias_atraso} dias · "
                    f"estado: {e.estado} · relevância: {e.score:.2f}"
                )

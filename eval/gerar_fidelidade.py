"""T17b — gera XLSX para rotulagem humana de fidelidade (groundedness).

Usa busca real + geração Ollama (sem Gemini/RAGAS). Um único modelo na medição.
Layout no estilo T15: aba `indice` + uma aba por pergunta (P0x).

Uso (índice já gerado; ollama serve + modelo puxado):
    USE_MOCK=false GEN_MOCK=false GEN_PROVIDER=ollama GEN_MODEL=qwen2.5 \\
      GEN_FALLBACK_MODEL= JUDGE_ENABLED=false HF_HUB_OFFLINE=1 \\
      python -m eval.gerar_fidelidade

Saída: eval/fidelidade_para_rotular.xlsx
Copie para eval/fidelidade_rotulos.xlsx, preencha n_afirmacoes / n_suportadas /
observacoes (B6–B8 em cada aba P0x), depois: python -m eval.calcular_fidelidade

Critério de rotulagem (groundedness ≠ citação):
- Afirmação: claim factual na resposta_avaliada (1 período / 1 bullet ≈ 1 afirmação).
- Suportada: sustentada pelo contexto nas evidências da aba (com ou sem [R#]).
- Não suportada: inventada, contradita, ou parcial (conservador: parcial = não).
- Falta de [R#] sozinha não zera a claim — anote em observacoes.
- Fora da métrica: bloco antes de MARCADOR_EXEMPLOS; recusas; esperado=recusar.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font

from eval.calcular_fidelidade import texto_para_avaliar
from src.agregar import MARCADOR_EXEMPLOS
from src.config import (
    CHROMA_DIR,
    GEN_FALLBACK_MODEL,
    GEN_MOCK,
    GEN_MODEL,
    GEN_PROVIDER,
    JUDGE_ENABLED,
    OLLAMA_HOST,
    USE_MOCK,
)
from src.index import BM25_PATH
from src.pipeline import perguntar
from src.schema import Evidencia, Resposta

PASTA = Path(__file__).resolve().parent
SAIDA = PASTA / "fidelidade_para_rotular.xlsx"

EVID_COLS = ["R#", "review_id", "nota", "texto"]
EVID_HEADER_ROW = 10


def _checar_env() -> None:
    erros: list[str] = []
    if USE_MOCK:
        erros.append("USE_MOCK=true (precisa USE_MOCK=false)")
    if GEN_MOCK:
        erros.append("GEN_MOCK=true (precisa GEN_MOCK=false)")
    if GEN_PROVIDER.strip().lower() != "ollama":
        erros.append(f"GEN_PROVIDER={GEN_PROVIDER!r} (precisa GEN_PROVIDER=ollama)")
    if GEN_FALLBACK_MODEL.strip():
        erros.append(
            f"GEN_FALLBACK_MODEL={GEN_FALLBACK_MODEL!r} "
            "(deixe vazio para um único modelo na medição)"
        )
    if JUDGE_ENABLED:
        erros.append("JUDGE_ENABLED=true (desligue para medir só a geração)")
    if erros:
        raise SystemExit(
            "Abortado: env incompatível com T17b fidelidade manual.\n  - "
            + "\n  - ".join(erros)
            + "\nExemplo:\n"
            "  USE_MOCK=false GEN_MOCK=false GEN_PROVIDER=ollama GEN_MODEL=qwen2.5 \\\n"
            "    GEN_FALLBACK_MODEL= JUDGE_ENABLED=false HF_HUB_OFFLINE=1 \\\n"
            "    python -m eval.gerar_fidelidade"
        )


def _checar_indice() -> None:
    faltando: list[str] = []
    if not CHROMA_DIR.is_dir():
        faltando.append(f"diretório {CHROMA_DIR}")
    if not BM25_PATH.exists():
        faltando.append(f"BM25 em {BM25_PATH}")
    if faltando:
        raise SystemExit(
            "Abortado: índice ausente ("
            + "; ".join(faltando)
            + "). Rode `python -m src.index --amostra` (ou índice completo) com USE_MOCK=false."
        )


def _checar_ollama() -> None:
    try:
        import ollama
    except ImportError as erro:
        raise SystemExit(
            "Abortado: pacote `ollama` ausente. Instale as dependências do projeto."
        ) from erro

    try:
        client = ollama.Client(host=OLLAMA_HOST, timeout=10)
        client.list()
        client.chat(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": "responda só: ok"}],
            options={"temperature": 0, "num_predict": 8},
        )
    except Exception as erro:  # noqa: BLE001 — fail-fast com mensagem clara
        raise SystemExit(
            f"Abortado: Ollama inacessível em {OLLAMA_HOST} "
            f"(modelo={GEN_MODEL}). Detalhe: {erro}"
        ) from erro


def _observacoes_iniciais(tipo: str, r: Resposta, completa: str, pid: str) -> str:
    if tipo == "agregada" and not r.recusada and MARCADOR_EXEMPLOS not in completa:
        print(f"  aviso: {pid} agregada sem marcador de exemplos")
        return "agregada sem MARCADOR_EXEMPLOS — avaliada a resposta inteira"
    return ""


def _escrever_cabecalho(
    ws,
    *,
    pergunta: str,
    tipo: str,
    esperado: str,
    recusou: bool,
    motivo_recusa: str,
    citacoes: str,
    resposta_avaliada: str,
    observacoes: str,
) -> None:
    pares = [
        (1, "pergunta", pergunta),
        (2, "tipo / esperado", f"{tipo} | {esperado}"),
        (3, "recusou / motivo_recusa", f"{recusou} | {motivo_recusa}"),
        (4, "citacoes", citacoes),
        (5, "resposta_avaliada", resposta_avaliada),
        (6, "n_afirmacoes", ""),
        (7, "n_suportadas", ""),
        (8, "observacoes", observacoes),
    ]
    for linha, rotulo, valor in pares:
        ws.cell(linha, 1, rotulo).font = Font(bold=True)
        cell = ws.cell(linha, 2, valor)
        if linha in {1, 5, 8}:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells("B1:D1")
    ws.merge_cells("B5:D5")
    ws.merge_cells("B8:D8")
    ws.row_dimensions[5].height = 80
    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 36
    ws.column_dimensions["C"].width = 12
    ws.column_dimensions["D"].width = 70


def _escrever_evidencias(ws, evidencias: list[Evidencia], inicio: int = EVID_HEADER_ROW) -> None:
    for col, nome in enumerate(EVID_COLS, start=1):
        ws.cell(inicio, col, nome).font = Font(bold=True)

    if not evidencias:
        ws.cell(inicio + 1, 1, "(sem evidências)")
        return

    for i, e in enumerate(evidencias, start=1):
        linha = inicio + i
        nota = e.review_score if e.review_score is not None else None
        ws.cell(linha, 1, i)
        ws.cell(linha, 2, e.review_id)
        ws.cell(linha, 3, nota)
        cell = ws.cell(linha, 4, e.texto or "")
        cell.alignment = Alignment(wrap_text=True, vertical="top")


def _escrever_aba(ws, p: pd.Series, r: Resposta) -> None:
    completa = r.texto or ""
    tipo = str(p["tipo"])
    pid = str(p["id"])
    obs = _observacoes_iniciais(tipo, r, completa, pid)
    _escrever_cabecalho(
        ws,
        pergunta=str(p["pergunta"]),
        tipo=tipo,
        esperado=str(p["esperado"]),
        recusou=r.recusada,
        motivo_recusa=r.motivo_recusa or "",
        citacoes=";".join(r.citacoes),
        resposta_avaliada=texto_para_avaliar(completa),
        observacoes=obs,
    )
    _escrever_evidencias(ws, r.evidencias)


def gerar(caminho: Path = SAIDA) -> Path:
    _checar_env()
    _checar_indice()
    _checar_ollama()

    if caminho.exists():
        print(f"Aviso: {caminho.name} já existe — será sobrescrito.")
        print("(fidelidade_rotulos.xlsx, se existir, não é apagado.)")

    perguntas = pd.read_csv(PASTA / "perguntas.csv")
    wb = Workbook()
    indice = wb.active
    indice.title = "indice"
    indice.append([
        "id",
        "tipo",
        "esperado",
        "pergunta",
        "recusou",
        "motivo_recusa",
        "n_evidencias",
        "n_citacoes",
    ])

    for _, p in perguntas.iterrows():
        pid = str(p["id"])
        print(f"{pid}: gerando…")
        r = perguntar(str(p["pergunta"]))
        ws = wb.create_sheet(title=pid[:31])
        _escrever_aba(ws, p, r)
        indice.append([
            pid,
            str(p["tipo"]),
            str(p["esperado"]),
            str(p["pergunta"]),
            r.recusada,
            r.motivo_recusa or "",
            len(r.evidencias),
            len(r.citacoes),
        ])
        status = "recusou" if r.recusada else f"{len(r.evidencias)} evidências"
        print(f"{pid}: ok ({status})")

    caminho.parent.mkdir(parents=True, exist_ok=True)
    tmp = caminho.with_suffix(caminho.suffix + ".tmp")
    wb.save(tmp)
    tmp.replace(caminho)
    print(f"\nPlanilha salva em {caminho}")
    print(
        "Copie para fidelidade_rotulos.xlsx, preencha n_afirmacoes/n_suportadas "
        "(B6–B8) e rode: python -m eval.calcular_fidelidade"
    )
    return caminho


if __name__ == "__main__":
    gerar()

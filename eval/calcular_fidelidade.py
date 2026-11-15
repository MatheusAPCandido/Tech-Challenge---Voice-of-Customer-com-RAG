"""T17b — fidelidade (groundedness) a partir de XLSX rotulado manualmente.

Métrica por pergunta: fidelidade = n_suportadas / n_afirmacoes.
- Macro: média das fidelidades (peso igual por pergunta).
- Micro: sum(n_suportadas) / sum(n_afirmacoes).

Elegíveis: esperado=responder, recusou=false, n_afirmacoes > 0 com contagens válidas.
Groundedness ≠ citação: ausência de [R#] não zera a claim se o fato estiver nas evidências.

Uso:
    # 1) copiar eval/fidelidade_para_rotular.xlsx → eval/fidelidade_rotulos.xlsx
    # 2) preencher n_afirmacoes / n_suportadas / observacoes (B6–B8) em cada aba P0x
    python -m eval.calcular_fidelidade
    python -m eval.calcular_fidelidade --xlsx eval/fidelidade_rotulos.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from src.agregar import MARCADOR_EXEMPLOS

PASTA = Path(__file__).resolve().parent
XLSX_PADRAO = PASTA / "fidelidade_rotulos.xlsx"
METRICAS_PATH = PASTA / "metricas_fidelidade.csv"


def texto_para_avaliar(resposta: str) -> str:
    """Trecho avaliado na fidelidade: após MARCADOR_EXEMPLOS, senão a resposta inteira."""
    texto = resposta or ""
    if MARCADOR_EXEMPLOS in texto:
        return texto.split(MARCADOR_EXEMPLOS, 1)[1].strip()
    return texto.strip()


def _eh_recusa(valor) -> bool:
    if isinstance(valor, bool):
        return valor
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return False
    return str(valor).strip().lower() in {"1", "true", "sim", "yes"}


def _parse_contagem(valor, campo: str, pid: str) -> int:
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        raise SystemExit(
            f"Contagem inválida em {pid}: `{campo}` vazio. "
            "Preencha n_afirmacoes e n_suportadas nas abas elegíveis (B6–B7)."
        )
    texto = str(valor).strip()
    if texto == "":
        raise SystemExit(
            f"Contagem inválida em {pid}: `{campo}` vazio. "
            "Preencha n_afirmacoes e n_suportadas nas abas elegíveis (B6–B7)."
        )
    if isinstance(valor, bool):
        raise SystemExit(
            f"Contagem inválida em {pid}: `{campo}`={valor!r} (inteiro esperado)."
        )
    if isinstance(valor, (int, float)):
        if float(valor) != int(valor):
            raise SystemExit(
                f"Contagem inválida em {pid}: `{campo}`={valor!r} (inteiro esperado)."
            )
        numero = int(valor)
    else:
        try:
            numero = int(texto)
        except (TypeError, ValueError) as erro:
            raise SystemExit(
                f"Contagem inválida em {pid}: `{campo}`={valor!r} (inteiro esperado)."
            ) from erro
    if numero < 0:
        raise SystemExit(
            f"Contagem inválida em {pid}: `{campo}`={numero} (deve ser >= 0)."
        )
    return numero


def _cabecalho_mapa(ws) -> dict[str, object]:
    """Lê rótulos da coluna A (linhas 1–8) → valor da coluna B."""
    mapa: dict[str, object] = {}
    for row in ws.iter_rows(min_row=1, max_row=8, min_col=1, max_col=2, values_only=True):
        if not row or row[0] is None:
            continue
        chave = str(row[0]).strip().casefold()
        mapa[chave] = row[1]
    return mapa


def _valor_por_prefixo(mapa: dict[str, object], *prefixos: str):
    for chave, valor in mapa.items():
        for pref in prefixos:
            if chave == pref or chave.startswith(pref):
                return valor
    return None


def _parse_esperado(valor) -> str:
    if valor is None:
        return ""
    texto = str(valor).strip()
    if "|" in texto:
        return texto.split("|")[-1].strip().lower()
    return texto.lower()


def _parse_recusou_cabecalho(valor) -> bool:
    if valor is None:
        return False
    if isinstance(valor, bool):
        return valor
    texto = str(valor).strip()
    if "|" in texto:
        texto = texto.split("|", 1)[0].strip()
    return _eh_recusa(texto)


def ler_rotulos_fidelidade(caminho: Path) -> pd.DataFrame:
    """Lê abas P0x do XLSX: id, esperado, recusou, n_afirmacoes, n_suportadas, observacoes."""
    if not caminho.exists():
        raise SystemExit(
            f"XLSX não encontrado: {caminho}. "
            "Copie fidelidade_para_rotular.xlsx → fidelidade_rotulos.xlsx e rotule."
        )

    wb = load_workbook(caminho, read_only=True, data_only=True)
    linhas: list[dict] = []
    try:
        for nome in wb.sheetnames:
            if nome.casefold() == "indice":
                continue
            ws = wb[nome]
            mapa = _cabecalho_mapa(ws)
            esperado_bruto = _valor_por_prefixo(mapa, "tipo / esperado", "esperado")
            recusou_bruto = _valor_por_prefixo(mapa, "recusou")
            n_af = _valor_por_prefixo(mapa, "n_afirmacoes")
            n_sup = _valor_por_prefixo(mapa, "n_suportadas")
            obs = _valor_por_prefixo(mapa, "observacoes")
            linhas.append({
                "id": nome,
                "esperado": _parse_esperado(esperado_bruto),
                "recusou": _parse_recusou_cabecalho(recusou_bruto),
                "n_afirmacoes": n_af if n_af is not None else "",
                "n_suportadas": n_sup if n_sup is not None else "",
                "observacoes": "" if obs is None else str(obs),
            })
    finally:
        wb.close()

    if not linhas:
        raise SystemExit(
            f"Nenhuma aba de pergunta em {caminho} (esperadas abas P0x além de indice)."
        )
    return pd.DataFrame(linhas)


def filtrar_elegiveis(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Valida contagens nas candidatas e devolve (elegíveis, ids omitidos).

    Candidatas: esperado=responder e recusou=false.
    Contagem inválida/vazia numa candidata → SystemExit.
    Elegíveis para a métrica: candidatas com n_afirmacoes > 0.
    """
    omitidos: list[str] = []
    linhas: list[dict] = []

    for _, row in df.iterrows():
        pid = str(row.get("id", ""))
        esperado = str(row.get("esperado", "")).strip().lower()
        recusou = _eh_recusa(row.get("recusou"))

        if esperado != "responder" or recusou:
            omitidos.append(pid)
            continue

        n_afirmacoes = _parse_contagem(row.get("n_afirmacoes"), "n_afirmacoes", pid)
        n_suportadas = _parse_contagem(row.get("n_suportadas"), "n_suportadas", pid)
        if n_suportadas > n_afirmacoes:
            raise SystemExit(
                f"Contagem inválida em {pid}: "
                f"n_suportadas ({n_suportadas}) > n_afirmacoes ({n_afirmacoes})."
            )
        if n_afirmacoes == 0:
            omitidos.append(pid)
            continue

        linhas.append({
            "id": pid,
            "n_afirmacoes": n_afirmacoes,
            "n_suportadas": n_suportadas,
            "fidelidade": n_suportadas / n_afirmacoes,
        })

    return pd.DataFrame(linhas), omitidos


def calcular_metricas(df: pd.DataFrame) -> dict:
    """Agrega macro/micro a partir do DataFrame rotulado (funções puras + validação).

    Retorna dict com:
      - por_pergunta: DataFrame (id, n_afirmacoes, n_suportadas, fidelidade)
      - macro, micro, n
      - omitidos: list[str]
    """
    por_pergunta, omitidos = filtrar_elegiveis(df)
    n = len(por_pergunta)
    if n == 0:
        return {
            "por_pergunta": por_pergunta,
            "macro": float("nan"),
            "micro": float("nan"),
            "n": 0,
            "omitidos": omitidos,
        }

    macro = float(por_pergunta["fidelidade"].mean())
    soma_sup = int(por_pergunta["n_suportadas"].sum())
    soma_af = int(por_pergunta["n_afirmacoes"].sum())
    micro = soma_sup / soma_af if soma_af else float("nan")
    return {
        "por_pergunta": por_pergunta,
        "macro": macro,
        "micro": micro,
        "n": n,
        "omitidos": omitidos,
    }


def rodar(caminho: Path = XLSX_PADRAO, saida: Path = METRICAS_PATH) -> pd.DataFrame:
    df = ler_rotulos_fidelidade(caminho)
    resultado = calcular_metricas(df)
    metricas: pd.DataFrame = resultado["por_pergunta"]
    saida.parent.mkdir(parents=True, exist_ok=True)
    metricas.to_csv(saida, index=False)

    if metricas.empty:
        print("Nenhuma pergunta elegível (esperado=responder, sem recusa, n_afirmacoes>0).")
    else:
        print(metricas.to_string(index=False))
        print()
        print(f"Macro fidelidade: {resultado['macro']:.3f}  (n={resultado['n']} perguntas)")
        print(f"Micro fidelidade: {resultado['micro']:.3f}")

    omitidos = resultado["omitidos"]
    if omitidos:
        print(f"Omitidos ({len(omitidos)}): {', '.join(omitidos)}")
    else:
        print("Omitidos: nenhum")

    print(f"\nGravado: {saida}")
    return metricas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="T17b: fidelidade (groundedness) a partir de XLSX rotulado"
    )
    parser.add_argument("--xlsx", type=Path, default=XLSX_PADRAO)
    parser.add_argument("--saida", type=Path, default=METRICAS_PATH)
    args = parser.parse_args()
    rodar(args.xlsx, args.saida)


if __name__ == "__main__":
    main()

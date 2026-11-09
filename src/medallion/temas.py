"""T08: rótulo temático de cada review (o assunto principal da avaliação).

Como os temas foram definidos
-----------------------------
1. Clustering (KMeans, k=30) nos embeddings E5 das 34.912 reviews para descobrir os assuntos
   que a base realmente tem. Os grupos misturavam assunto e estilo de escrita (textos só em
   maiúsculas, textos de uma palavra), então serviram para definir a lista de temas abaixo,
   não como rótulo final.
2. Cada review recebe os temas cujas regras de palavras-chave casam com o texto. As regras
   tratam negação ("sem nenhum defeito", "não atrasou", "nada a reclamar") e variações
   ("só recebi 1", "comprei 2 e veio apenas 1", "recebi 2 dos 3").
3. O tema principal é o primeiro da lista PRIORIDADE entre os encontrados: do problema mais
   específico ou grave para o mais genérico. Sem nenhum tema: "outros".

Validação (rotulagem manual)
----------------------------
- 120 reviews sorteadas, usadas para ajustar as regras: 96,7% de acerto (otimista).
- 100 reviews novas, nunca usadas no ajuste: 80% de acerto no tema principal.
- Testado e descartado: (a) similaridade com frases-modelo nos embeddings, porque as notas de
  similaridade do E5 ficam muito próximas entre temas; (b) classificador sobre os embeddings
  treinado com as regras, porque o ganho (de 80% para 82%) ficou dentro da margem de erro.

Uso:
    from src.medallion.temas import TEMAS, tema_principal, temas_da_review
"""

from __future__ import annotations

import re
import unicodedata

# slug -> descrição. O slug é o valor gravado na coluna `tema` e usado no filtro `tema` da busca.
TEMAS = {
    "entrega_incompleta": "Pedido incompleto: faltou item, unidade ou peça",
    "nao_recebido": "Produto não recebido",
    "produto_errado": "Produto errado ou diferente do anunciado",
    "produto_defeito": "Produto com defeito, quebrado ou danificado",
    "atraso_entrega": "Entrega atrasada ou demorada",
    "atendimento_loja": "Atendimento da loja, troca, devolução, cancelamento ou nota fiscal",
    "qualidade_produto": "Qualidade, material, acabamento ou tamanho do produto",
    "preco_frete": "Preço, custo-benefício ou frete",
    "entrega_rapida": "Entrega rápida ou dentro do prazo",
    "avaliacao_geral": "Opinião geral, sem assunto específico",
    "outros": "Nenhum dos assuntos acima",
}
PRIORIDADE = [t for t in TEMAS if t != "outros"]

_PROBLEMAS = (
    r"(atras\w*|demor\w*|quebrad\w*|defeit\w*|danific\w*|avari\w*|amassad\w*|riscad\w*"
    r"|errad\w*|problema\w*|falt\w*|reclam\w*)"
)
# "sem nenhum defeito", "não atrasou", "nada a reclamar": o problema citado não aconteceu
_NEGACAO = re.compile(r"\b(nao|sem|nenhum|nenhuma|nunca|nada)\s+(\w+\s+){0,1}?" + _PROBLEMAS + r"\b")
# "não chegou no prazo" é atraso, não entrega no prazo
_NAO_NO_PRAZO = re.compile(r"\b(nao|nunca)\s+(\w+\s+){0,3}?(no prazo|dentro do prazo|a tempo)\b")

REGRAS = {
    "entrega_incompleta": [
        r"\b(so|somente|apenas)\s+(\w+\s+){0,2}?(recebi|recebemos|veio|vieram|chegou|chegaram|entregaram|enviaram|enviou|foi entregue|foram entregues)\b",
        r"\b(faltando|faltou|faltaram|incomplet\w*|entrega parcial|parcialmente)\b",
        r"\b(entregue|entregues|recebi|recebemos|veio|vieram|chegou|chegaram|enviaram|enviado|enviada|mandaram)\s+(apenas|somente|so)\s+(\d+|um|uma|metade|parte|o|a)\b",
        r"\b\d+\s+(dos|das)\s+\d+\b",
        r"\b(nao|nunca)\s+(\w+\s+){0,3}?(completo|completa|todos os|todas as|o restante|a outra|o outro|os outros|as outras|um dos|uma das)\b",
        r"\b(comprei|pedi|paguei|compramos|paga|pago|solicitei|encomendei)\b.{0,60}\b(recebi|veio|vieram|chegou|chegaram|entregaram|entrega|entregou|enviaram|enviou)\s+(\w+\s+){0,2}?(1|um|uma|2|dois|duas|apenas|somente|so|metade)\b",
    ],
    "nao_recebido": [
        r"\b(nao|nunca)\s+(\w+\s+){0,3}?(recebi|recebemos|recebeu|chegou|chegaram|foi entregue|foram entregues|entregaram|entregue|recebido)\b(?!\s+(\w+\s+)?(resposta|retorno|contato|email|nota|nf|nenhum))",
        r"\b(aguardando|esperando|aguardo|espero)\s+(\w+\s+){0,2}?(entrega|produto|pedido|encomenda|chegar|receber)\b",
        r"\bcade\s+(o|meu|minha|a)\b",
        r"\b(consta|constava|aparece)\s+(\w+\s+){0,2}?entregue\b",
        r"\bextravi\w*\b",
        r"\bainda\s+(\w+\s+){0,2}?(aguardando|esperando|aguardo)\b",
        r"\b(nao|nunca)\s+(\w+\s+){0,2}?(entregou|entregaram|efetuou a entrega|enviou|enviaram o produto|despachad\w*|postad\w*)\b",
    ],
    "produto_errado": [
        r"\berrad[oa]s?\b",
        r"\bdiferente\s+(\w+\s+){0,2}?(anunci\w*|foto\w*|imagem|imagens|pedid\w*|comprad\w*|descri\w*|site|que (eu )?(comprei|pedi|escolhi))",
        r"\b(nao|nada)\s+(\w+\s+){0,2}?(condiz|corresponde|confere|parecid\w*)\b",
        r"\b(veio|vieram|enviaram|mandaram|entregaram|recebi)\s+(um |uma )?(outro|outra|outros|outras)\b",
        r"\b(cor|modelo|tamanho|numero|voltagem)\s+(diferente|errad\w*|trocad\w*)\b",
        r"\bnao (era|e) o (que|produto)\b",
        r"\b(trocaram|produto trocado|pedido trocado|propaganda enganosa)\b",
        r"\b(anunci\w*|foto\w*|imagem|site|descri\w*)\b.{0,30}\bdiferent\w*",
        r"\bdiferent\w*\b.{0,30}\b(anunci\w*|foto\w*|imagem|site|descri\w*)",
        r"\b(cor|modelo|tamanho)\s+(e|era|veio)\s+(outr\w*|diferent\w*)\b",
    ],
    "produto_defeito": [
        r"\b(quebrad\w*|defeit\w*|danificad\w*|avariad\w*|amassad\w*|riscad\w*|rachad\w*|trincad\w*|estragad\w*|rasgad\w*|manchad\w*|vazand\w*|furad\w*|avaria\w*|violad\w*)\b",
        r"\b(nao|parou de|deixou de)\s+(\w+\s+){0,1}?(funciona\w*|liga|ligou|carrega\w*|acende|acendeu)\b",
    ],
    "atraso_entrega": [
        # "demorou para estornar" é atendimento, não atraso de entrega
        r"\b(atras\w*|demor\w*(?!\s+(\w+\s+){0,3}?(resolv|respond|estorn|reembols|devol|retorn|troc))|fora do prazo|apos o prazo|depois do prazo|passou do prazo|prazo (\w+ )?(longo|extenso|grande|excedid\w*)|lentidao|muito lent\w*)\b",
    ],
    "atendimento_loja": [
        r"\b(atendiment\w*|atencios\w*|vendedor\w*|lojista|sac|suporte|reembols\w*|estorn\w*|devolu\w*|devolv\w*|cancel\w*|trocar|troca|reclama\w*|protocolo|chamado|respond\w*|resposta|retorno|contato|e mail|email|emails|mensage\w*|telefon\w*|ligacao|nota fiscal|nf)\b",
    ],
    "qualidade_produto": [
        r"\b(qualidade|material|acabamento|tecido|fragil|resistente|durabilidade|costura|tamanho|pequen\w*|fin[oa]s?|cheiro|confortavel|desconfortavel|inferior|mal acabad\w*|mal feit\w*|bem feit\w*|bem acabad\w*|robust\w*|reforcad\w*)\b",
        r"\b(poderia|podia|deveria) ser (maior|menor|melhor|mais)\b",
        r"\b(menor|maior)\s+(do )?que\b",
    ],
    "preco_frete": [
        r"\b(frete\w*|preco\w*|caro|cara demais|barat\w*|custo|beneficio|vale a pena|promoc\w*|desconto)\b",
    ],
    "entrega_rapida": [
        r"\bantes\s+(\w+\s+){0,1}?(do prazo|da data|do previsto|do tempo|do esperado|do combinado|do dia)\b",
        r"\b(antecedencia|adiantad\w*|antecipad\w*|pontual\w*)\b",
        r"\b(entrega|entregue|chegou|chegaram|envio|enviado|recebi)\s+(\w+\s+){0,2}?(rapid\w*|ligeir\w*|veloz)\b",
        r"\b(rapid\w*|super rapid\w*)\s+(\w+\s+){0,1}?(entrega|envio)\b",
        r"\b(no prazo|dentro do prazo|prazo certo|prazo combinado|prazo estipulado|prazo previsto|a tempo)\b",
        r"\b(cumpr\w*\s+(\w+\s+){0,2}?prazo|prazo cumprido|rapid\w*|rapidez|rapidinho)\b",
    ],
    "avaliacao_geral": [
        r"\b(otim\w*|excelent\w*|bom|boa|bons|boas|perfeit\w*|recomend\w*|gostei|amei|adorei|satisfeit\w*|satisfacao|parabens|maravilh\w*|top|nota 10|show|legal|pessim\w*|horrivel|ruim|decepc\w*|insatisfeit\w*|nao gostei|lament\w*|tudo certo|tudo ok|ok|conforme\w*|como esperado|certo|certinho|direitinho|lind\w*|bonit\w*|obrigad\w*|agrade\w*|eficaz|gosto|problema\w*|confiavel|confianca|nota maxima|aprovad\w*|blz|beleza|tranquil\w*)\b",
    ],
}
_REGRAS = {tema: [re.compile(p) for p in padroes] for tema, padroes in REGRAS.items()}


def normalizar(texto: str) -> str:
    """Minúsculas, sem acentos e sem pontuação; remove problemas negados."""
    t = unicodedata.normalize("NFKD", str(texto).casefold())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = _NAO_NO_PRAZO.sub(" fora do prazo ", t)
    t = _NEGACAO.sub(" ", t)
    return t


def temas_da_review(texto: str) -> list[str]:
    """Todos os temas citados na review, na ordem de PRIORIDADE (pode ser vazia)."""
    t = normalizar(texto)
    return [tema for tema in PRIORIDADE if any(r.search(t) for r in _REGRAS[tema])]


def tema_principal(temas: list[str]) -> str:
    """Primeiro tema da lista (já ordenada por PRIORIDADE) ou "outros"."""
    return temas[0] if temas else "outros"

"""Chunking igual ao da reindexação do FAQ PAES em produção (workflow n8n "PAES - Reindexacao bge-m3")."""
import re

MAX_CHARS = 900


def chunk_por_secao(texto: str, doc: str) -> list[dict]:
    """Edital e obras: trechos de até MAX_CHARS, cada um começando com o caminho da seção."""
    linhas = texto.replace("\r\n", "\n").split("\n")
    h2 = h3 = h4 = sub = ""
    cur: list[str] = []
    size = 0
    chunks: list[dict] = []

    def flush():
        nonlocal cur, size
        body = "\n".join(cur).strip()
        secao = " > ".join(p for p in (h2, h3, h4, sub) if p)
        tipo = "cronograma" if re.search(r"CRONOGRAMA", secao, re.I) else "geral"
        if body:
            prefixo = doc + (" > " + secao if secao else "")
            chunks.append({"secao": secao, "tipo": tipo, "text": prefixo + "\n\n" + body})
        cur, size = [], 0

    for raw in linhas:
        line = raw.rstrip()
        if line.startswith("# "):
            continue
        if line.startswith("## "):
            flush()
            h2, h3, h4, sub = line[3:].strip(), "", "", ""
            continue
        if line.startswith("### "):
            flush()
            h3, h4, sub = line[4:].strip(), "", ""
            continue
        if line.startswith("#### "):
            flush()
            h4, sub = line[5:].strip(), ""
            continue
        if re.fullmatch(r"\*\*[^*]+\*\*", line.strip()):
            flush()
            sub = line.strip().replace("**", "")
            continue
        if (h3 or h4 or sub) and re.match(r"^\*\*\d+\.\d", line):
            flush()
            h3 = h4 = sub = ""
        if size > 0 and size + len(line) > MAX_CHARS:
            flush()
        cur.append(line)
        size += len(line) + 1
    flush()
    return chunks


def chunk_por_bloco(texto: str) -> list[dict]:
    """Conteúdo programático: um trecho por bloco, com etapa, disciplina e "Bloco X de N"."""
    etapa = disciplina = ""
    cur = None
    blocos = []
    for line in texto.replace("\r\n", "\n").split("\n"):
        if line.startswith("## "):
            etapa, cur = line[3:].strip(), None
            continue
        if line.startswith("### "):
            disciplina = line[4:].strip()
            cur = {"etapa": etapa, "disciplina": disciplina, "titulo": disciplina, "linhas": []}
            blocos.append(cur)
            continue
        if line.startswith("#### "):
            cur = {"etapa": etapa, "disciplina": disciplina, "titulo": line[5:].strip(), "linhas": []}
            blocos.append(cur)
            continue
        if cur is not None:
            cur["linhas"].append(line)

    validos = [b for b in blocos if "".join(b["linhas"]).strip()]
    totais: dict[str, int] = {}
    for b in validos:
        k = b["etapa"] + "|" + b["disciplina"]
        totais[k] = totais.get(k, 0) + 1

    chunks = []
    for b in validos:
        k = b["etapa"] + "|" + b["disciplina"]
        titulo = b["titulo"]
        if re.search(r"Bloco \d+$", titulo):
            titulo += f" de {totais[k]}"
        text = f"PAES, {b['etapa']}\n{titulo}\n\n" + "\n".join(b["linhas"]).strip()
        chunks.append({"etapa": b["etapa"], "disciplina": b["disciplina"], "text": text})
    return chunks

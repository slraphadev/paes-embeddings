"""Métricas de recuperação. Cada pergunta tem exatamente um trecho relevante (o que a originou)."""
import re

import numpy as np

KS = (1, 5, 10)

# Prefixos exigidos por cada família de modelo; modelos fora da lista não usam prefixo.
PREFIXOS = {
    "intfloat/multilingual-e5": ("query: ", "passage: "),
}


def prefixos(nome_modelo: str) -> tuple[str, str]:
    for chave, par in PREFIXOS.items():
        if chave in nome_modelo:
            return par
    return "", ""


def metricas(rankings: list[list[int]], relevantes: list[int]) -> dict:
    """rankings[i] = índices dos trechos ordenados para a pergunta i."""
    res = {f"recall@{k}": 0.0 for k in KS}
    mrr = 0.0
    for rank, rel in zip(rankings, relevantes):
        top = list(rank[: max(KS)])
        for k in KS:
            res[f"recall@{k}"] += rel in top[:k]
        if rel in top:
            mrr += 1 / (top.index(rel) + 1)
    n = len(relevantes)
    res = {k: v / n for k, v in res.items()}
    res["mrr@10"] = mrr / n
    return res


def ranking_denso(model, perguntas: list[str], textos: list[str], nome_modelo: str, batch_size=32) -> list[list[int]]:
    pq, pp = prefixos(nome_modelo)
    emb_c = model.encode([pp + t for t in textos], batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
    emb_q = model.encode([pq + q for q in perguntas], batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
    sims = emb_q @ emb_c.T
    return np.argsort(-sims, axis=1)[:, : max(KS)].tolist()


def _tokens(texto: str) -> list[str]:
    return re.findall(r"\w+", texto.lower())


def ranking_bm25(perguntas: list[str], textos: list[str]) -> list[list[int]]:
    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi([_tokens(t) for t in textos])
    out = []
    for q in perguntas:
        scores = bm25.get_scores(_tokens(q))
        out.append(np.argsort(-scores)[: max(KS)].tolist())
    return out

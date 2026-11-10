"""Compara BM25 e modelos de embedding na recuperação do trecho certo para cada pergunta do teste.

    python scripts/04_avaliar.py
    python scripts/04_avaliar.py --modelos bm25 models/e5-small-paes BAAI/bge-m3
"""
import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.avaliacao import metricas, ranking_bm25, ranking_denso
from src.io_utils import DATA, RESULTS, ler_jsonl

PADRAO = [
    "bm25",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-small",
    "intfloat/multilingual-e5-base",
    "BAAI/bge-m3",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", nargs="+", default=PADRAO)
    ap.add_argument("--saida", default=RESULTS / "avaliacao.csv", type=Path)
    args = ap.parse_args()

    corpus = ler_jsonl(DATA / "corpus.jsonl")
    teste = ler_jsonl(DATA / "test.jsonl")
    idx = {c["id"]: i for i, c in enumerate(corpus)}
    textos = [c["text"] for c in corpus]
    perguntas = [p["pergunta"] for p in teste]
    relevantes = [idx[p["chunk_id"]] for p in teste]
    fontes = [p["fonte"] for p in teste]
    print(f"corpus: {len(corpus)} trechos · teste: {len(teste)} perguntas")

    linhas = []
    for nome in args.modelos:
        inicio = time.time()
        if nome == "bm25":
            rankings = ranking_bm25(perguntas, textos)
        else:
            from sentence_transformers import SentenceTransformer

            rankings = ranking_denso(SentenceTransformer(nome), perguntas, textos, nome)
        seg = time.time() - inicio

        geral = metricas(rankings, relevantes)
        linhas.append({"modelo": nome, "fonte": "todas", **geral, "segundos": round(seg, 1)})
        for fonte in sorted(set(fontes)):
            sel = [i for i, f in enumerate(fontes) if f == fonte]
            m = metricas([rankings[i] for i in sel], [relevantes[i] for i in sel])
            linhas.append({"modelo": nome, "fonte": fonte, **m, "segundos": ""})
        print(f"{nome}: " + " · ".join(f"{k} {v:.3f}" for k, v in geral.items()))

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    novo = not args.saida.exists()
    with args.saida.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        if novo:
            w.writeheader()
        w.writerows(linhas)

    print("\n| Modelo | Recall@1 | Recall@5 | Recall@10 | MRR@10 |\n|---|---|---|---|---|")
    for l in linhas:
        if l["fonte"] == "todas":
            print(f"| {l['modelo']} | {l['recall@1']:.3f} | {l['recall@5']:.3f} | {l['recall@10']:.3f} | {l['mrr@10']:.3f} |")


if __name__ == "__main__":
    main()

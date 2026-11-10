"""Separa treino e teste POR TRECHO (nenhum trecho do teste aparece no treino) e gera uma amostra para revisão manual."""
import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.io_utils import DATA, escrever_jsonl, ler_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teste", type=float, default=0.2)
    ap.add_argument("--revisao", type=int, default=60, help="perguntas do teste para revisar à mão")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    perguntas = ler_jsonl(DATA / "perguntas.jsonl")
    textos = {c["id"]: c["text"] for c in ler_jsonl(DATA / "corpus.jsonl")}

    # Estratifica por fonte para o teste ter edital, obras e conteúdo na mesma proporção.
    por_fonte: dict[str, set] = {}
    for p in perguntas:
        por_fonte.setdefault(p["fonte"], set()).add(p["chunk_id"])
    ids_teste = set()
    for ids in por_fonte.values():
        ids = sorted(ids)
        ids_teste |= set(rng.sample(ids, max(1, round(len(ids) * args.teste))))

    treino = [p for p in perguntas if p["chunk_id"] not in ids_teste]
    teste = [p for p in perguntas if p["chunk_id"] in ids_teste]
    escrever_jsonl(DATA / "train.jsonl", treino)
    escrever_jsonl(DATA / "test.jsonl", teste)
    print(f"treino: {len(treino)} perguntas · teste: {len(teste)} perguntas ({len(ids_teste)} trechos)")

    amostra = rng.sample(teste, min(args.revisao, len(teste)))
    with (DATA / "revisao.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["chunk_id", "pergunta", "trecho", "ok (s/n)", "observacao"])
        for p in amostra:
            w.writerow([p["chunk_id"], p["pergunta"], textos[p["chunk_id"]], "", ""])
    print(f"amostra para revisão: {DATA / 'revisao.csv'}")


if __name__ == "__main__":
    main()

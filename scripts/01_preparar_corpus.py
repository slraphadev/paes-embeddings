"""Gera data/corpus.jsonl a partir dos .md do PAES em data/raw/."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.chunking import chunk_por_bloco, chunk_por_secao
from src.io_utils import DATA, escrever_jsonl

FONTES = {
    "edital": ("1-edital-retificado.md", "Edital PAES 2026"),
    "conteudo": ("2-componentes-curriculares.md", None),
    "obras": ("3-obras-indicadas.md", "Literatura e obras obrigatórias PAES 2026"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=DATA / "raw", type=Path)
    args = ap.parse_args()

    corpus = []
    for fonte, (arquivo, doc) in FONTES.items():
        texto = (args.raw / arquivo).read_text(encoding="utf-8")
        chunks = chunk_por_bloco(texto) if fonte == "conteudo" else chunk_por_secao(texto, doc)
        for i, c in enumerate(chunks, 1):
            corpus.append({"id": f"{fonte}-{i:04d}", "fonte": fonte, **c})
        print(f"{fonte}: {len(chunks)} trechos")

    escrever_jsonl(DATA / "corpus.jsonl", corpus)
    print(f"total: {len(corpus)} trechos -> {DATA / 'corpus.jsonl'}")


if __name__ == "__main__":
    main()

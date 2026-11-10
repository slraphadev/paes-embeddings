"""Publica o corpus e os pares pergunta-trecho no Hugging Face.

    huggingface-cli login
    python scripts/06_publicar_dataset.py usuario/paes-retrieval-ptbr
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.io_utils import DATA, ler_jsonl


def main():
    from datasets import Dataset, DatasetDict

    repo = sys.argv[1]
    textos = {c["id"]: c["text"] for c in ler_jsonl(DATA / "corpus.jsonl")}

    def pares(arquivo):
        linhas = ler_jsonl(DATA / arquivo)
        return Dataset.from_list([{**p, "trecho": textos[p["chunk_id"]]} for p in linhas])

    DatasetDict({"train": pares("train.jsonl"), "test": pares("test.jsonl")}).push_to_hub(repo, config_name="pares")
    Dataset.from_list(ler_jsonl(DATA / "corpus.jsonl")).push_to_hub(repo, config_name="corpus")
    print(f"publicado em https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    main()

"""Fine-tuning de um modelo de embedding pequeno com os pares (pergunta, trecho) do treino.

Roda numa GPU T4 gratuita (Colab ou Kaggle) em poucos minutos.
    python scripts/05_finetune.py
    python scripts/05_finetune.py --push usuario/e5-small-paes
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.avaliacao import prefixos
from src.io_utils import DATA, MODELS, ler_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="intfloat/multilingual-e5-small")
    ap.add_argument("--saida", default=MODELS / "e5-small-paes", type=Path)
    ap.add_argument("--epocas", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--push", default=None, help="repo no Hugging Face (usuario/nome) para publicar o modelo")
    args = ap.parse_args()

    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    textos = {c["id"]: c["text"] for c in ler_jsonl(DATA / "corpus.jsonl")}
    pq, pp = prefixos(args.base)
    treino = ler_jsonl(DATA / "train.jsonl")
    ds = Dataset.from_dict({
        "anchor": [pq + p["pergunta"] for p in treino],
        "positive": [pp + textos[p["chunk_id"]] for p in treino],
    })
    print(f"{len(ds)} pares de treino")

    model = SentenceTransformer(args.base)
    # Os outros trechos do batch servem de negativos. NO_DUPLICATES evita que duas perguntas
    # do mesmo trecho caiam no mesmo batch e virem falsos negativos uma da outra.
    loss = MultipleNegativesRankingLoss(model)
    targs = SentenceTransformerTrainingArguments(
        output_dir=str(args.saida / "checkpoints"),
        num_train_epochs=args.epocas,
        per_device_train_batch_size=args.batch,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no",
        logging_steps=10,
        seed=42,
        report_to="none",
    )
    SentenceTransformerTrainer(model=model, args=targs, train_dataset=ds, loss=loss).train()

    model.save(str(args.saida))
    print(f"modelo salvo em {args.saida}")
    print(f"agora avalie: python scripts/04_avaliar.py --modelos {args.base} {args.saida}")

    if args.push:
        model.push_to_hub(args.push)
        print(f"publicado em https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()

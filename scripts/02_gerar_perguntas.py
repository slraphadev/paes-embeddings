"""Gera perguntas sintéticas de candidatos para cada trecho do corpus, via LLM com API compatível com OpenAI.

Padrão: Groq (gpt-oss-120b, tier gratuito). Retoma de onde parou se for interrompido.
    export GROQ_API_KEY=...
    python scripts/02_gerar_perguntas.py --limite-conteudo 200
"""
import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.io_utils import DATA, escrever_jsonl, ler_jsonl

PROMPT = """Você vai criar perguntas que candidatos do PAES (vestibular seriado da Unimontes) mandariam por WhatsApp.

Trecho do documento oficial:
\"\"\"
{texto}
\"\"\"

Escreva {n} perguntas em português do Brasil que:
- sejam respondidas por este trecho (a resposta precisa estar nele);
- soem como um estudante do ensino médio escrevendo no WhatsApp: uma mais direta e outra mais informal;
- não copiem frases do trecho e não mencionem "trecho", "documento" ou "texto acima";
- sejam específicas o bastante para não caberem em qualquer parte do edital.

Responda só com JSON no formato {{"perguntas": ["...", "..."]}}."""


def extrair_perguntas(conteudo: str) -> list[str]:
    m = re.search(r"\{.*\}", conteudo, re.S)
    if not m:
        return []
    try:
        dados = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [p.strip() for p in dados.get("perguntas", []) if isinstance(p, str) and len(p.strip()) >= 10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="https://api.groq.com/openai/v1")
    ap.add_argument("--modelo", default="openai/gpt-oss-120b")
    ap.add_argument("--chave-env", default="GROQ_API_KEY")
    ap.add_argument("--n", type=int, default=2, help="perguntas por trecho")
    ap.add_argument("--limite-conteudo", type=int, default=200, help="amostra de blocos do conteúdo programático (0 = todos)")
    ap.add_argument("--pausa", type=float, default=2.5, help="segundos entre chamadas (limite do tier gratuito)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from openai import OpenAI, RateLimitError

    client = OpenAI(base_url=args.base_url, api_key=os.environ[args.chave_env])
    corpus = ler_jsonl(DATA / "corpus.jsonl")

    conteudo = [c for c in corpus if c["fonte"] == "conteudo"]
    if args.limite_conteudo and len(conteudo) > args.limite_conteudo:
        conteudo = random.Random(args.seed).sample(conteudo, args.limite_conteudo)
    alvo = [c for c in corpus if c["fonte"] != "conteudo"] + conteudo

    saida = DATA / "perguntas.jsonl"
    feitos = {p["chunk_id"] for p in ler_jsonl(saida)}
    pendentes = [c for c in alvo if c["id"] not in feitos]
    print(f"{len(alvo)} trechos no alvo, {len(pendentes)} pendentes")

    # Raciocínio curto no gpt-oss: as perguntas são simples e isso economiza a cota diária de tokens.
    extra = {"reasoning_effort": "low"} if "gpt-oss" in args.modelo else {}

    for i, c in enumerate(pendentes, 1):
        for tentativa in range(3):
            try:
                r = client.chat.completions.create(
                    model=args.modelo,
                    messages=[{"role": "user", "content": PROMPT.format(texto=c["text"], n=args.n)}],
                    temperature=0.7,
                    extra_body=extra,
                )
                break
            except RateLimitError as e:
                print(f"limite de taxa: {str(e)[:200]}")
                time.sleep(30 * (tentativa + 1))
        else:
            # Provavelmente a cota diária acabou. O progresso está salvo: rode de novo mais tarde
            # ou troque de modelo com --modelo.
            print(f"parei em {c['id']}. Rode de novo para continuar de onde parou.")
            break

        perguntas = extrair_perguntas(r.choices[0].message.content or "")
        escrever_jsonl(saida, [{"chunk_id": c["id"], "fonte": c["fonte"], "pergunta": p} for p in perguntas], modo="a")
        if i % 20 == 0:
            print(f"{i}/{len(pendentes)}")
        time.sleep(args.pausa)

    print(f"pronto: {len(ler_jsonl(saida))} perguntas em {saida}")


if __name__ == "__main__":
    main()

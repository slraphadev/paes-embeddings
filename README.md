# PAES Embeddings · recuperação semântica em português para documentos de vestibular

Avaliação e fine-tuning de modelos de embedding na tarefa de encontrar, entre os trechos do edital do PAES/Unimontes, o trecho que responde à pergunta de um candidato.

O projeto nasceu do **FAQ PAES**, um assistente RAG no WhatsApp que desenvolvi e coloquei em produção no Hub de Educação Digital da Unimontes. Em produção, escolhi o BAAI/bge-m3 depois de testar outros modelos sem uma medição formal. Aqui eu meço essa escolha e testo se um modelo pequeno, ajustado para o domínio, consegue alcançá-lo.

## Perguntas

1. Qual modelo de embedding recupera melhor o trecho certo em documentos de vestibular em português?
2. Um modelo pequeno (multilingual-e5-small, 118M parâmetros) ajustado com fine-tuning chega perto do bge-m3 (568M)?
3. A busca densa supera uma busca lexical (BM25) nesse domínio?

## Dados

- **Corpus:** edital, conteúdo programático e lista de obras do PAES 2026, documentos públicos da Unimontes. A divisão em trechos é **a mesma usada em produção**: até 900 caracteres por seção, cada trecho começando com o caminho da seção, e um trecho por bloco no conteúdo programático.
- **Perguntas:** geradas por LLM (gpt-oss-120b via Groq), simulando candidatos no WhatsApp, 2 por trecho. Cada pergunta tem como resposta correta o trecho que a originou.
- **Divisão:** treino e teste separados **por trecho**, estratificados por fonte, para nenhum trecho do teste aparecer no treino.
- **Revisão:** 60 perguntas do teste revisadas manualmente (justificativas das reprovações redigidas com apoio de LLM e conferidas). **46 de 60 (77%) foram consideradas válidas.** As inválidas se concentram no conteúdo programático, que só lista temas: o LLM gerou perguntas pedindo explicações ou comparações que o documento não traz.

Dataset: [huggingface.co/datasets/SEU_USUARIO/paes-retrieval-ptbr](https://huggingface.co/datasets/SEU_USUARIO/paes-retrieval-ptbr)

## Resultados

Teste com [N] perguntas sobre um corpus de [N] trechos.

| Modelo | Parâmetros | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|---|---|---|---|---|---|
| BM25 | n/a | | | | |
| paraphrase-multilingual-MiniLM-L12-v2 | 118M | | | | |
| multilingual-e5-small | 118M | | | | |
| multilingual-e5-base | 278M | | | | |
| BAAI/bge-m3 | 568M | | | | |
| **e5-small-paes (fine-tuning)** | 118M | | | | |

Modelo ajustado: [huggingface.co/SEU_USUARIO/e5-small-paes](https://huggingface.co/SEU_USUARIO/e5-small-paes)

[Análise: o que os números mostram, por fonte (edital, obras, conteúdo), e o que isso significa para o FAQ PAES em produção.]

## Método

- **Métricas:** Recall@k (o trecho certo aparece entre os k primeiros?) e MRR@10 (em que posição ele aparece). Cálculo próprio em `src/avaliacao.py`, o mesmo para BM25 e modelos densos.
- **Fine-tuning:** sentence-transformers com `MultipleNegativesRankingLoss`, em que os outros trechos do batch servem de negativos. O amostrador `NO_DUPLICATES` evita que duas perguntas do mesmo trecho caiam no mesmo batch e virem falsos negativos. 3 épocas, batch de 32, learning rate 2e-5, GPU T4 gratuita.
- **Prefixos:** os modelos e5 usam `query:` e `passage:`, como recomendam os autores.

## Limitações

- Cerca de 23% das perguntas do teste são inválidas (ver Revisão). O ruído afeta todos os modelos igualmente, então a comparação continua válida, mas os números absolutos ficam abaixo do desempenho real.
- As perguntas são sintéticas. Perguntas reais de candidatos tendem a ser mais curtas, ambíguas e com erros de digitação.
- Cada pergunta tem um único trecho correto. Quando dois trechos respondem à mesma pergunta, a métrica subestima o modelo.
- O LLM que gerou as perguntas viu o trecho, o que pode favorecer sobreposição de palavras e ajudar o BM25.
- Corpus pequeno e de um único domínio. Os resultados não se generalizam para outros editais sem nova medição.

## Como reproduzir

```bash
pip install -r requirements.txt

# Coloque em data/raw/: 1-edital-retificado.md, 2-componentes-curriculares.md, 3-obras-indicadas.md
python scripts/01_preparar_corpus.py
export GROQ_API_KEY=...
python scripts/02_gerar_perguntas.py      # retoma de onde parou
python scripts/03_dividir.py              # gera data/revisao.csv para revisão manual

# Com GPU (ou notebooks/colab_gpu.ipynb)
python scripts/04_avaliar.py
python scripts/05_finetune.py
python scripts/04_avaliar.py --modelos intfloat/multilingual-e5-small models/e5-small-paes BAAI/bge-m3
```

## Estrutura

```
src/chunking.py         divisão em trechos (igual à de produção)
src/avaliacao.py        métricas, BM25 e busca densa
scripts/01..06          pipeline: corpus → perguntas → divisão → avaliação → fine-tuning → publicação
notebooks/colab_gpu.ipynb
results/avaliacao.csv   resultados, incluindo a quebra por fonte
```

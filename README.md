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

Dataset: [huggingface.co/datasets/slraphadev/paes-retrieval-ptbr](https://huggingface.co/datasets/slraphadev/paes-retrieval-ptbr)

## Resultados

Teste com 183 perguntas (92 trechos que não aparecem no treino) sobre um corpus de 738 trechos.

| Modelo | Parâmetros | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|---|---|---|---|---|---|
| BM25 | n/a | 0,497 | 0,798 | 0,852 | 0,625 |
| paraphrase-multilingual-MiniLM-L12-v2 | 118M | 0,339 | 0,514 | 0,617 | 0,416 |
| multilingual-e5-small | 118M | 0,448 | 0,678 | 0,765 | 0,544 |
| multilingual-e5-base | 278M | 0,464 | 0,694 | 0,803 | 0,569 |
| BAAI/bge-m3 (usado em produção) | 568M | 0,486 | 0,781 | 0,891 | 0,613 |
| **e5-small-paes (fine-tuning)** | 118M | **0,568** | **0,820** | **0,918** | **0,675** |

Modelo ajustado: [huggingface.co/slraphadev/e5-small-paes](https://huggingface.co/slraphadev/e5-small-paes)

Por fonte (MRR@10):

| Modelo | Edital (96) | Conteúdo (79) | Obras (8) |
|---|---|---|---|
| BM25 | 0,621 | 0,609 | 0,823 |
| BAAI/bge-m3 | 0,643 | 0,554 | 0,828 |
| e5-small original | 0,577 | 0,472 | 0,875 |
| **e5-small-paes** | **0,706** | **0,619** | 0,854 |

### O que os números mostram

1. **A escolha de produção se confirma entre os modelos prontos.** O bge-m3 é o melhor modelo denso sem ajuste, à frente do e5-base, do e5-small e do MiniLM em todas as métricas.
2. **O fine-tuning funcionou.** O e5-small ajustado ganhou 12 pontos de Recall@1 (0,448 → 0,568) e 13 de MRR (0,544 → 0,675) sobre a própria versão original, e **superou o bge-m3 em todas as métricas com cerca de 1/5 dos parâmetros**. O maior ganho foi no edital, onde o vocabulário administrativo (SRV, deferimento, Quadro 1) é mais específico.
3. **O BM25 é um baseline forte.** Ele empata com o bge-m3 (MRR 0,625 contra 0,613) e ganha dele no Recall@1. Parte disso deve vir das perguntas sintéticas, que tendem a repetir palavras do trecho (ver Limitações). Mesmo assim, o resultado sugere que uma **busca híbrida (BM25 + densa)** pode melhorar o FAQ PAES.
4. **O conteúdo programático é a parte mais difícil** para todos os modelos. Ele também concentra as perguntas inválidas da revisão.

### Implicações para o FAQ PAES

- Um modelo de 118M parâmetros ajustado roda em CPU num servidor modesto, o que eliminaria a dependência da API gratuita do Hugging Face sem perder qualidade.
- Antes de trocar o modelo em produção, é preciso validar com **perguntas reais** dos candidatos (o histórico do chat), porque o ganho aqui foi medido em perguntas da mesma distribuição sintética usada no treino.

## Método

- **Métricas:** Recall@k (o trecho certo aparece entre os k primeiros?) e MRR@10 (em que posição ele aparece). Cálculo próprio em `src/avaliacao.py`, o mesmo para BM25 e modelos densos.
- **Fine-tuning:** sentence-transformers com `MultipleNegativesRankingLoss`, em que os outros trechos do batch servem de negativos. O amostrador `NO_DUPLICATES` evita que duas perguntas do mesmo trecho caiam no mesmo batch e virem falsos negativos. 3 épocas, batch de 32, learning rate 2e-5, GPU T4 gratuita.
- **Prefixos:** os modelos e5 usam `query:` e `passage:`, como recomendam os autores.

## Limitações

- Cerca de 23% das perguntas do teste são inválidas (ver Revisão). O ruído afeta todos os modelos igualmente, então a comparação continua válida, mas os números absolutos ficam abaixo do desempenho real.
- As perguntas são sintéticas. Perguntas reais de candidatos tendem a ser mais curtas, ambíguas e com erros de digitação.
- Cada pergunta tem um único trecho correto. Quando dois trechos respondem à mesma pergunta, a métrica subestima o modelo.
- O LLM que gerou as perguntas viu o trecho, o que pode favorecer sobreposição de palavras e ajudar o BM25.
- O teste é pequeno (183 perguntas; só 8 de obras, então os números de obras não são conclusivos). Diferenças de 1 a 3 pontos entre modelos estão dentro do ruído; a vantagem do modelo ajustado (8 pontos de Recall@1 sobre o bge-m3) é maior, mas ainda medida numa amostra pequena.
- O modelo ajustado foi treinado com perguntas geradas pelo mesmo LLM e com o mesmo prompt das perguntas do teste. Parte do ganho pode ser adaptação a esse estilo de pergunta, e não só ao domínio.
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

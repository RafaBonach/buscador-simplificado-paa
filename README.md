# buscador-simplificado-paa

Sistema de perguntas e respostas com busca semântica, citações de fonte e formatação por LLM local.

## Requisitos atendidos

O projeto implementa:

- Download de corpus textual em larga escala (meta padrão: **10 GB**) usando datasets públicos (ex.: Hugging Face).
- Processamento em chunks de texto para indexação semântica.
- Quatro estratégias de recuperação para comparação de desempenho/complexidade:
  - Similaridade de cosseno (TF-IDF)
  - Word2Vec Average
  - Sentence Embeddings
  - HNSW Search
- Resposta formatada por LLM local (quando disponível) com fallback local sem nuvem.
- Exibição de citações por documento recuperado.
- Avaliação via framework **RAGAS** e saída de métricas em JSON.
- Benchmark de tempo de resposta por método.

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

### 1) Baixar corpus textual (10 GB)

Exemplo com Wikipedia do Hugging Face:

```bash
python semantic_qa.py download \
  --dataset wikipedia \
  --split train \
  --text-field text \
  --target-gb 10 \
  --output data/corpus/wiki_pt.txt
```

> Ajuste `--dataset` e `--text-field` para outras fontes (Stack Exchange, PubMed, etc.).

### 2) Construir índices semânticos

```bash
python semantic_qa.py build-index \
  --data-dir data/corpus \
  --index-file artifacts/index.pkl
```

### 3) Perguntar ao sistema (com citações)

```bash
python semantic_qa.py ask \
  --index-file artifacts/index.pkl \
  --question "Qual a diferença entre busca semântica e busca lexical?" \
  --method hnsw \
  --top-k 5
```

Métodos suportados em `--method`:

- `cosine`
- `word2vec_average`
- `sentence_embeddings`
- `hnsw`

### 4) Benchmark de desempenho

Crie um arquivo com perguntas (uma por linha), por exemplo `data/queries.txt`:

```txt
O que é indexação vetorial?
Como funciona HNSW?
Qual o papel de embeddings em RAG?
```

Rode o benchmark:

```bash
python semantic_qa.py benchmark \
  --index-file artifacts/index.pkl \
  --queries-file data/queries.txt \
  --output artifacts/benchmark.json
```

### 5) Avaliação com RAGAS

Prepare um JSON com amostras no formato abaixo (`data/eval_samples.json`):

```json
[
  {
    "question": "Pergunta do usuário",
    "answer": "Resposta gerada pelo sistema",
    "ground_truth": "Resposta esperada",
    "contexts": ["Trecho 1", "Trecho 2"]
  }
]
```

Execute:

```bash
python semantic_qa.py evaluate \
  --eval-file data/eval_samples.json \
  --output artifacts/ragas_metrics.json
```

## Observações

- Para usar LLM local real, passe `--local-llm-path` no comando `ask` apontando para um modelo local compatível com `transformers` (ex.: TinyLlama, Phi-3 Mini, Mistral).
- Sem `--local-llm-path`, o sistema usa um fallback local para formatar resposta e manter as citações.
- O benchmark gera métricas de tempo de resposta por método para apoiar a comparação de desempenho.
- Se alguma dependência opcional não estiver instalada, o `build-index` continua para os métodos disponíveis e informa os métodos efetivamente indexados.

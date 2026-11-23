# Roteiro da apresentação (10 min) — de onde sai cada item do enunciado

O enunciado pede slides (original + PDF) com os itens abaixo (a numeração do PDF repete o 13). A coluna "Fonte" diz
qual comando ou arquivo gera o dado, para você preencher os slides com números **medidos na sua máquina**.

| # | Item pedido | Fonte / o que dizer |
|---|---|---|
| 1 | Integrantes | — |
| 2 | Linguagens, bibliotecas e licenças | Tabela de licenças abaixo |
| 3 | Programas: servidor, banco de dados | Sem servidor externo: SQLite (`chunks.db`) + arquivos memmap; CLI Python. Opcional: Ollama/llama.cpp como juiz local do RAGAS |
| 4 | Tema e funcionamento | `README.md` (fluxo) e `paa_qa/qa.py`: pergunta → busca → trechos → LLM → resposta + citações |
| 5 | Dados: o que, quanto | `download` imprime `text_gib`, `docs`, `sources`; `data/corpus/manifest.json`. Wikipédia pt ≈ 2,8 GB, por isso a meta de 10 GB combina fontes |
| 6 | Pré-processamento e complexidades | Chunking por sentença O(n); TF-IDF O(T); Word2Vec O(T·épocas·janela); embeddings O(N·transformer); HNSW O(N log N · M · efC · d). `stats` mostra o tempo medido de cada etapa |
| 7 | Tempo da busca semântica e complexidade | `benchmark` → `artifacts/benchmark.md` (latência média/p95 por método + tabela de complexidade) |
| 8 | Como foi o "treinamento" da LLM | **O LLM não é treinado**: é um modelo pré-treinado usado só para formatar (prompt com trechos). O que se treina/constrói: Word2Vec (`--w2v-*`) e os índices. Diga isso explicitamente |
| 9 | Como o servidor forma a resposta; complexidades | `llm.py`: prompt (≤ top-k × 900 caracteres) → geração gulosa O(tokens gerados × custo do modelo); fallback extrativo O(top-k × sentenças) |
| 10 | Recursos computacionais do grupo e o que é possível | Hardware real do grupo + seção "Recursos" do README (`--max-chunks`) |
| 11 | Tempo de cada fase | `stats` (ingestão, TF-IDF, Word2Vec, embeddings, HNSW) e `download` (`elapsed_s`) |
| 12 | Como o tempo de treinamento afeta a qualidade | Varie `--w2v-epochs` / `--w2v-train-chunks` / `--max-chunks`, rode `benchmark --eval-file` e compare hit@k/MRR |
| 13 | Como o tempo de inferência afeta a qualidade | Varie `--max-new-tokens` e o `--llm` (TinyLlama → Phi-3 → Mistral) no `evaluate`: compare F1, RAGAS e latência de geração |
| 13b | Avaliação do LLM pelo framework | `evaluate --ragas` → `artifacts/evaluation.json` (`ragas.metrics`, com `n_valid`/`n_failed`) + métricas próprias |
| 14 | Demais aspectos | Retomada de download/embeddings, citação por documento, limitações (README) |

## Licenças (confirme na página de cada item antes de citar no slide)

| Item | Licença |
|---|---|
| Python | PSF |
| numpy, scipy, scikit-learn | BSD-3-Clause |
| gensim | LGPL-2.1 |
| hnswlib | Apache-2.0 |
| faiss (usado no HNSW quando não há hnswlib) | MIT |
| Streamlit | Apache-2.0 |
| WikiExtractor (extração dos dumps) | AGPL-3.0 |
| DeepSeek-R1-Distill-Qwen-1.5B | MIT |
| sentence-transformers, transformers, datasets, huggingface_hub | Apache-2.0 |
| PyTorch | BSD-style |
| RAGAS | Apache-2.0 |
| SQLite | domínio público |
| paraphrase-multilingual-MiniLM-L12-v2 | Apache-2.0 |
| SmolLM2 | Apache-2.0 |
| TinyLlama | Apache-2.0 |
| Phi-3 Mini | MIT |
| Mistral 7B | Apache-2.0 |
| Texto da Wikipédia (dados) | CC BY-SA 4.0 |

Essas licenças foram listadas de memória, **não verificadas** nas páginas oficiais durante o desenvolvimento.

## Roteiro de demonstração prática (3 min)

1. `ask` com `--method hnsw --llm <modelo>` numa pergunta do tema → mostrar resposta, **citações** e tempos.
2. A mesma pergunta com `cosine` e `word2vec_average` → comparar os trechos recuperados.
3. Abrir `artifacts/benchmark.md` e `artifacts/evaluation.json`.

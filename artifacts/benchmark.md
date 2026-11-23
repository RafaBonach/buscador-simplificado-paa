# Benchmark — 3035 chunks, 30 consultas, top-5, 5 repetições

| Método | Latência p50 (ms) | Média (ms) | p95 (ms) | Build (s) | Índice (MB) | hit@k | MRR | recall vs. exato |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cosine | 0.45 | 0.58 | 1.15 | 2.0 | 2.65 | 0.90 | 0.85 | — |
| word2vec_average | 0.14 | 0.22 | 0.69 | 2.0 | 29.25 | 0.40 | 0.28 | — |
| sentence_embeddings | 14.77 | 14.81 | 16.57 | 136.1 | 2.33 | 0.97 | 0.83 | — |
| hnsw | 12.60 | 12.52 | 13.68 | 0.4 | 5.10 | 0.97 | 0.83 | 1.00 |

## Complexidade teórica

| Método | Construção | Consulta | Memória |
|---|---|---|---|
| cosine | O(T) — T = tokens do corpus (CountVectorizer + IDF) | O(Σ postings dos termos da pergunta) — matriz termo×chunk funciona como índice invertido | O(nnz) — esparsa |
| word2vec_average | O(T · épocas · janela · (1+neg)) para treinar + O(T) para a média por chunk | O(|q|·d + N·d) — varredura exata | O(N·d) float32 |
| sentence_embeddings | O(N · custo do transformer) — domina: ~O(L·n²·d) por chunk | O(encode(q) + N·d) — varredura exata | O(N·d) float16 |
| hnsw | O(N · log N · M · ef_construction · d) sobre os embeddings | O(encode(q) + ef · log N · d) — aproximada | O(N·d + N·M) — vetores + grafo |

## Resumo

- **fastest_query**: word2vec_average
- **best_quality_mrr**: cosine
- **best_tradeoff**: cosine

> Atenção: com 30 perguntas e 3035 chunks, diferenças pequenas entre os métodos (de MRR ou de tempo) **não são significativas**, e o hit@k tende a 1 porque o top-k cobre boa parte do corpus. A mediana (p50) é mais confiável que a média quando há medições atípicas.

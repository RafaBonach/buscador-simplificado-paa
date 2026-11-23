"""Benchmark comparando os quatro métodos: tempo de resposta, qualidade da recuperação, custo e complexidade."""

from __future__ import annotations

import time
from pathlib import Path

from .evaluate import latency_stats, retrieval_metrics
from .index import METHODS, IndexSearcher, load_meta

THEORY = {
    "cosine": {
        "build": "O(T) — T = tokens do corpus (CountVectorizer + IDF)",
        "query": "O(Σ postings dos termos da pergunta) — matriz termo×chunk funciona como índice invertido",
        "memory": "O(nnz) — esparsa",
    },
    "word2vec_average": {
        "build": "O(T · épocas · janela · (1+neg)) para treinar + O(T) para a média por chunk",
        "query": "O(|q|·d + N·d) — varredura exata",
        "memory": "O(N·d) float32",
    },
    "sentence_embeddings": {
        "build": "O(N · custo do transformer) — domina: ~O(L·n²·d) por chunk",
        "query": "O(encode(q) + N·d) — varredura exata",
        "memory": "O(N·d) float16",
    },
    "hnsw": {
        "build": "O(N · log N · M · ef_construction · d) sobre os embeddings",
        "query": "O(encode(q) + ef · log N · d) — aproximada",
        "memory": "O(N·d + N·M) — vetores + grafo",
    },
}


def run_benchmark(index_dir: Path, items: list[dict], methods: list[str] | None = None, top_k: int = 5,
                  repeats: int = 3, device: str | None = None) -> dict:
    """`items`: lista de {"question": ..., "relevant_titles": [...] (opcional)}."""
    meta = load_meta(index_dir)
    searcher = IndexSearcher(index_dir, device=device)
    chosen = [m for m in (methods or list(METHODS)) if m in METHODS]
    report: dict = {
        "index": {"chunks": meta["ingest"]["chunks"], "docs": meta["ingest"]["docs"], "top_k": top_k,
                  "queries": len(items), "repeats": repeats},
        "methods": {},
    }
    exact_ids: dict[str, list[int]] = {}
    for method in chosen:
        if method not in searcher.methods:
            report["methods"][method] = {"skipped": True, "reason": meta.get("skipped", {}).get(method, "não indexado")}
            continue
        searcher.warmup(method)  # exclui do tempo o carregamento de modelo/matriz
        latencies: list[float] = []
        per_query = {"hit": [], "mrr": [], "recall": []}
        top_ids: dict[str, list[int]] = {}
        for item in items:
            hits = []
            for _ in range(repeats):
                start = time.perf_counter()
                hits = searcher.search(item["question"], method, top_k)
                latencies.append((time.perf_counter() - start) * 1000)
            top_ids[item["question"]] = [h.chunk.chunk_id for h in hits]
            if item.get("relevant_titles"):
                m = retrieval_metrics([h.chunk.title for h in hits], item["relevant_titles"])
                for key in per_query:
                    per_query[key].append(m[key])
        stats = latency_stats(latencies)
        entry = {
            "latency_ms": stats,
            "queries_per_second": 1000.0 / stats["mean"] if stats["mean"] else None,
            "build_seconds": meta["methods"][method].get("build_seconds"),
            "index_bytes": meta["methods"][method].get("bytes"),
            "complexity": THEORY[method],
        }
        if per_query["hit"]:
            n = len(per_query["hit"])
            entry["quality"] = {"hit_at_k": sum(per_query["hit"]) / n, "mrr": sum(per_query["mrr"]) / n,
                                "recall_at_k": sum(per_query["recall"]) / n}
        report["methods"][method] = entry
        report.setdefault("_top_ids", {})[method] = top_ids
        if method == "sentence_embeddings":
            exact_ids = top_ids

    # Recall do HNSW frente à busca exata nos mesmos embeddings (métrica padrão de ANN).
    top_ids_by_method = report.pop("_top_ids", {})
    if exact_ids and "hnsw" in top_ids_by_method:
        overlaps = []
        for question, exact in exact_ids.items():
            approx = top_ids_by_method["hnsw"].get(question, [])
            if exact:
                overlaps.append(len(set(exact) & set(approx)) / len(exact))
        if overlaps:
            report["methods"]["hnsw"]["recall_vs_exact_embeddings"] = sum(overlaps) / len(overlaps)
    searcher.close()
    report["ranking"] = _rank(report["methods"])
    return report


def _rank(methods: dict) -> dict:
    """Resumo objetivo: quem lidera em qualidade, em latência e o melhor equilíbrio."""
    active = {m: e for m, e in methods.items() if not e.get("skipped")}
    out: dict = {}
    if not active:
        return out
    out["fastest_query"] = min(active, key=lambda m: active[m]["latency_ms"]["p50"])
    with_quality = {m: e for m, e in active.items() if "quality" in e}
    if with_quality:
        out["best_quality_mrr"] = max(with_quality, key=lambda m: with_quality[m]["quality"]["mrr"])
        best_mrr = max(e["quality"]["mrr"] for e in with_quality.values())
        # equilíbrio: entre os métodos com MRR >= 95% do melhor, o de menor latência mediana (p50)
        close = [m for m, e in with_quality.items() if e["quality"]["mrr"] >= 0.95 * best_mrr]
        out["best_tradeoff"] = min(close, key=lambda m: active[m]["latency_ms"]["p50"])
    return out


def render_markdown(report: dict) -> str:
    lines = [
        f"# Benchmark — {report['index']['chunks']} chunks, {report['index']['queries']} consultas, "
        f"top-{report['index']['top_k']}, {report['index']['repeats']} repetições", "",
        "| Método | Latência p50 (ms) | Média (ms) | p95 (ms) | Build (s) | Índice (MB) | hit@k | MRR | recall vs. exato |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(value, spec="{:.2f}"):
        return "—" if value is None else spec.format(value)

    for method, e in report["methods"].items():
        if e.get("skipped"):
            lines.append(f"| {method} | ignorado ({e['reason']}) | | | | | | | |")
            continue
        q = e.get("quality", {})
        size_mb = None if e.get("index_bytes") is None else e["index_bytes"] / 1e6
        lines.append(
            f"| {method} | {fmt(e['latency_ms']['p50'])} | {fmt(e['latency_ms']['mean'])} | {fmt(e['latency_ms']['p95'])} | "
            f"{fmt(e.get('build_seconds'), '{:.1f}')} | "
            f"{fmt(size_mb, '{:.2f}')} | {fmt(q.get('hit_at_k'))} | {fmt(q.get('mrr'))} | {fmt(e.get('recall_vs_exact_embeddings'))} |"
        )
    lines += ["", "## Complexidade teórica", "", "| Método | Construção | Consulta | Memória |", "|---|---|---|---|"]
    for method, e in report["methods"].items():
        if not e.get("skipped"):
            c = e["complexity"]
            lines.append(f"| {method} | {c['build']} | {c['query']} | {c['memory']} |")
    if report.get("ranking"):
        lines += ["", "## Resumo", ""] + [f"- **{k}**: {v}" for k, v in report["ranking"].items()]
        info = report["index"]
        if info["queries"] < 50 or info["chunks"] < 10_000:
            lines += ["", f"> Atenção: com {info['queries']} perguntas e {info['chunks']} chunks, diferenças pequenas entre os "
                          "métodos (de MRR ou de tempo) **não são significativas**, e o hit@k tende a 1 porque o top-k cobre "
                          "boa parte do corpus. A mediana (p50) é mais confiável que a média quando há medições atípicas."]
    return "\n".join(lines) + "\n"

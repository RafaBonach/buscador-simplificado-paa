"""Avaliação do sistema: métricas de recuperação, de resposta e de tempo.

Formato do arquivo de perguntas (JSON):
[
  {"question": "...", "reference": "resposta esperada", "relevant_titles": ["Título do documento relevante"]}
]
`reference` e `relevant_titles` são opcionais; as métricas que dependem deles são omitidas quando ausentes.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

from .qa import QASystem
from .text import content_tokens, fold_accents

_CITE = re.compile(r"\[\d+\]")


def load_questions(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all("question" in item for item in data):
        raise ValueError("O arquivo de avaliação deve ser uma lista de objetos com o campo 'question'.")
    return data


def _norm_title(title: str) -> str:
    return fold_accents(title).casefold().strip()


def token_f1(prediction: str, reference: str) -> float:
    """F1 entre conjuntos (multiconjuntos) de tokens de conteúdo, ignorando marcadores [n]."""
    pred = content_tokens(_CITE.sub(" ", prediction))
    ref = content_tokens(reference)
    if not pred or not ref:
        return 0.0
    common = 0
    ref_pool = list(ref)
    for tok in pred:
        if tok in ref_pool:
            ref_pool.remove(tok)
            common += 1
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def context_recall_lexical(contexts: list[str], reference: str) -> float:
    """Fração dos tokens de conteúdo da resposta esperada presentes nos contextos recuperados."""
    ref = set(content_tokens(reference))
    if not ref:
        return 0.0
    ctx = set(content_tokens(" ".join(contexts)))
    return len(ref & ctx) / len(ref)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct / 100
    lo, hi = int(pos), min(int(pos) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def latency_stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": statistics.fmean(values), "p50": _percentile(values, 50), "p95": _percentile(values, 95),
            "min": min(values), "max": max(values)}


def retrieval_metrics(hit_titles: list[str], relevant_titles: list[str]) -> dict:
    """hit@k, MRR e recall@k (por documento) para uma consulta."""
    relevant = {_norm_title(t) for t in relevant_titles}
    found_rank = None
    found: set[str] = set()
    for rank, title in enumerate(hit_titles, start=1):
        norm = _norm_title(title)
        if norm in relevant:
            found.add(norm)
            if found_rank is None:
                found_rank = rank
    return {
        "hit": 1.0 if found_rank else 0.0,
        "mrr": 1.0 / found_rank if found_rank else 0.0,
        "recall": len(found) / len(relevant) if relevant else 0.0,
    }


def evaluate_system(qa: QASystem, questions: list[dict], method: str | None = None, top_k: int = 5) -> dict:
    rows = []
    method = method or qa.default_method()
    qa.searcher.warmup(method)  # a 1ª consulta paga o carregamento do índice; fora da medição de latência
    for item in questions:
        answer = qa.ask(item["question"], method=method, top_k=top_k)
        row = {
            "question": item["question"],
            "answer": answer.answer,
            "contexts": answer.contexts,
            "cited_titles": [c.title for c in answer.citations],
            "reference": item.get("reference"),
            "cited": answer.cited(),
            "timings_ms": answer.timings_ms,
        }
        if item.get("relevant_titles"):
            row["retrieval"] = retrieval_metrics(row["cited_titles"], item["relevant_titles"])
        if item.get("reference"):
            row["answer_f1"] = token_f1(answer.answer, item["reference"])
            row["context_recall_lexical"] = context_recall_lexical(answer.contexts, item["reference"])
        rows.append(row)
    return {"method": answer.method if rows else method, "generator": qa.formatter.name, "top_k": top_k,
            "summary": summarize(rows), "rows": rows}


def summarize(rows: list[dict]) -> dict:
    def mean_of(getter) -> float | None:
        vals = [getter(r) for r in rows if getter(r) is not None]
        return statistics.fmean(vals) if vals else None

    return {
        "questions": len(rows),
        "hit_at_k": mean_of(lambda r: r.get("retrieval", {}).get("hit")),
        "mrr": mean_of(lambda r: r.get("retrieval", {}).get("mrr")),
        "recall_at_k": mean_of(lambda r: r.get("retrieval", {}).get("recall")),
        "answer_token_f1": mean_of(lambda r: r.get("answer_f1")),
        "context_recall_lexical": mean_of(lambda r: r.get("context_recall_lexical")),
        "answers_with_citation": mean_of(lambda r: 1.0 if r["cited"] else 0.0),
        "latency_ms": {
            "retrieval": latency_stats([r["timings_ms"]["retrieval"] for r in rows]),
            "generation": latency_stats([r["timings_ms"]["generation"] for r in rows]),
            "total": latency_stats([r["timings_ms"]["total"] for r in rows]),
        },
    }

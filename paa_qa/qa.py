"""Pipeline pergunta -> busca semântica -> resposta formatada + citações + tempos."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .index import Hit, IndexSearcher
from .llm import ExtractiveFormatter, LocalLLM, cited_indices
from .text import shorten


@dataclass
class Citation:
    n: int
    title: str
    url: str
    doc_id: int
    chunk_id: int
    position: int
    score: float
    snippet: str


@dataclass
class Answer:
    question: str
    method: str
    answer: str
    generator: str
    citations: list[Citation]
    contexts: list[str]
    timings_ms: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def cited(self) -> list[int]:
        return cited_indices(self.answer)

    def render(self) -> str:
        lines = ["=== RESPOSTA ===", self.answer, "", "=== CITAÇÕES (documento de origem) ==="]
        for c in self.citations:
            where = f" — {c.url}" if c.url else ""
            lines.append(f"[{c.n}] {c.title}{where}")
            lines.append(f"     trecho #{c.position} · score={c.score:.4f}")
            lines.append(f"     “{c.snippet}”")
        t = self.timings_ms
        lines += ["", "=== TEMPOS ===",
                  f"busca ({self.method}): {t.get('retrieval', 0):.1f} ms | geração ({self.generator}): "
                  f"{t.get('generation', 0):.1f} ms | total: {t.get('total', 0):.1f} ms"]
        return "\n".join(lines)


class QASystem:
    def __init__(self, searcher: IndexSearcher, llm: LocalLLM | None = None):
        self.searcher = searcher
        self.formatter = llm if llm is not None else ExtractiveFormatter()

    @classmethod
    def load(cls, index_dir: Path, llm: str | None = None, max_new_tokens: int = 256, device: str | None = None) -> "QASystem":
        searcher = IndexSearcher(index_dir, device=device)
        model = LocalLLM(llm, max_new_tokens=max_new_tokens, device=device) if llm else None
        return cls(searcher, model)

    def default_method(self) -> str:
        methods = self.searcher.methods
        return "hnsw" if "hnsw" in methods else methods[0]

    def ask(self, question: str, method: str | None = None, top_k: int = 5) -> Answer:
        method = method or self.default_method()
        load = getattr(self.formatter, "load", None)
        if load is not None:
            load()  # carregar/baixar o LLM não faz parte do tempo de resposta (é idempotente)
        t0 = time.perf_counter()
        hits: list[Hit] = self.searcher.search(question, method, top_k)
        t1 = time.perf_counter()
        text = self.formatter.generate(question, hits)
        t2 = time.perf_counter()
        citations = [
            Citation(n=h.rank, title=h.chunk.title, url=h.chunk.url, doc_id=h.chunk.doc_id, chunk_id=h.chunk.chunk_id,
                     position=h.chunk.pos, score=h.score, snippet=shorten(h.chunk.text, 220))
            for h in hits
        ]
        return Answer(
            question=question, method=method, answer=text, generator=self.formatter.name, citations=citations,
            contexts=[h.chunk.text for h in hits],
            timings_ms={"retrieval": (t1 - t0) * 1000, "generation": (t2 - t1) * 1000, "total": (t2 - t0) * 1000},
        )

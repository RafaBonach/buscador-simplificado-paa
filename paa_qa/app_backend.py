"""Lógica da interface Streamlit, sem depender do Streamlit (assim é testável).

O índice e o LLM são mantidos em memória entre uma pergunta e outra: carregar o índice leva segundos e o LLM
leva minutos. Só um LLM fica carregado por vez (trocar de modelo libera o anterior), para não estourar a RAM.
"""

from __future__ import annotations

import gc
import threading
from pathlib import Path

from .index import METHODS, IndexSearcher, load_meta
from .llm import LocalLLM
from .qa import Answer, QASystem

DEFAULT_INDEX_DIR = "artifacts/index"

# Rótulos do seletor do app -> nome interno do método.
METHOD_LABELS = {
    "HNSW Search": "hnsw",
    "Cosseno (Cosine Similarity)": "cosine",
    "Word2Vec Average": "word2vec_average",
    "Sentence Embeddings": "sentence_embeddings",
}
LLM_EXTRACTIVE = "Sem LLM (resposta extrativa)"
LLM_OPTIONS = [LLM_EXTRACTIVE, "DeepSeek", "SmolLM", "TinyLlama", "Phi-3 Mini", "Mistral"]

_lock = threading.Lock()
_generation_lock = threading.Lock()
_searchers: dict[str, IndexSearcher] = {}
_llm_cache: dict = {"key": None, "obj": None}
_warmed: set = set()


def index_info(index_dir: str) -> dict:
    """Resumo do índice para a barra lateral. Levanta FileNotFoundError se ele ainda não foi construído."""
    meta = load_meta(Path(index_dir))
    methods = [m for m in METHODS if m in meta["methods"]]
    return {"chunks": meta["ingest"]["chunks"], "docs": meta["ingest"]["docs"], "methods": methods,
            "skipped": meta.get("skipped", {})}


def method_options(info: dict) -> list[str]:
    """Rótulos dos métodos que existem neste índice, na ordem do seletor do app."""
    return [label for label, key in METHOD_LABELS.items() if key in info["methods"]]


def _searcher(index_dir: str) -> IndexSearcher:
    key = str(Path(index_dir).resolve())
    with _lock:
        if key not in _searchers:
            _searchers[key] = IndexSearcher(Path(index_dir))
        return _searchers[key]


def _llm(label: str, max_new_tokens: int) -> LocalLLM | None:
    if not label or label == LLM_EXTRACTIVE:
        return None
    key = (label, max_new_tokens)
    with _lock:
        if _llm_cache["key"] != key:
            _llm_cache.update(key=None, obj=None)  # libera o modelo anterior antes de carregar outro
            gc.collect()
            _llm_cache.update(key=key, obj=LocalLLM(label, max_new_tokens=max_new_tokens))
        return _llm_cache["obj"]


def warmup(index_dir: str, method_label: str) -> None:
    """Carrega modelos e matrizes do método (1ª vez demora; depois é imediato) para que o tempo mostrado
    na resposta seja o da consulta, sem o carregamento."""
    key = (str(Path(index_dir).resolve()), method_label)
    if key in _warmed:
        return
    _searcher(index_dir).warmup(METHOD_LABELS[method_label])
    _warmed.add(key)


def ask(index_dir: str, method_label: str, llm_label: str, question: str, top_k: int = 5,
        max_new_tokens: int = 256) -> Answer:
    qa = QASystem(_searcher(index_dir), _llm(llm_label, max_new_tokens))
    with _generation_lock:  # um LLM local atende uma pergunta por vez
        return qa.ask(question, method=METHOD_LABELS[method_label], top_k=top_k)


def reset_cache() -> None:
    """Fecha os índices e o LLM em memória (usado nos testes e ao trocar de pasta)."""
    with _lock:
        for searcher in _searchers.values():
            searcher.close()
        _searchers.clear()
        _warmed.clear()
        _llm_cache.update(key=None, obj=None)
    gc.collect()


# ---------------------------------------------------------------- formatação (Markdown do Streamlit)
def md_escape(text: str) -> str:
    """O Streamlit trata $...$ como fórmula; textos da Wikipédia têm cifrões e colchetes."""
    return text.replace("$", "\\$")


def render_citations(answer: Answer) -> str:
    if not answer.citations:
        return "_Nenhum documento recuperado._"
    lines = []
    for c in answer.citations:
        title = md_escape(c.title).replace("[", "\\[").replace("]", "\\]")
        head = f"**[{c.n}] [{title}]({c.url})**" if c.url else f"**[{c.n}] {title}**"
        lines.append(f"- {head} — similaridade {c.score:.2f}\n  > {md_escape(c.snippet)}")
    return "\n".join(lines)


def render_timings(answer: Answer) -> str:
    t = answer.timings_ms
    return (f"⏱️ busca ({answer.method}): {t['retrieval']:.0f} ms · geração ({answer.generator}): "
            f"{t['generation']:.0f} ms · total: {t['total']:.0f} ms")

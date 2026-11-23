from __future__ import annotations

import json
from pathlib import Path

import pytest

from paa_qa.index import BuildConfig, build_index

ROOT = Path(__file__).resolve().parent.parent
DEMO_CORPUS = ROOT / "data" / "sample" / "demo_corpus.jsonl"
DEMO_QUESTIONS = ROOT / "data" / "eval" / "questions.json"


@pytest.fixture(scope="session")
def demo_questions() -> list[dict]:
    return json.loads(DEMO_QUESTIONS.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def built_index(tmp_path_factory) -> Path:
    """Índice do corpus de demonstração com os métodos que não dependem de bibliotecas pesadas."""
    index_dir = tmp_path_factory.mktemp("index")
    cfg = BuildConfig(methods=("cosine", "sentence_embeddings"), embedding_model="hashing")
    build_index(DEMO_CORPUS, index_dir, cfg, log=lambda _msg: None)
    return index_dir


def write_corpus(path: Path, n_docs: int, prefix: str = "doc") -> Path:
    """Gera um .jsonl sintético com n_docs documentos de vários parágrafos."""
    with path.open("w", encoding="utf-8") as handle:
        for i in range(n_docs):
            text = " ".join(
                f"Esta é a sentença {j} do documento {i} sobre o assunto número {i % 7} e a palavra chave{i}x{j}."
                for j in range(12)
            )
            handle.write(json.dumps({"id": f"{prefix}-{i}", "title": f"Documento {i}", "url": f"http://x/{i}",
                                     "source": "teste", "text": text}, ensure_ascii=False) + "\n")
    return path

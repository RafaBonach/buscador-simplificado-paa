#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import warnings


def _lazy_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Dependência opcional ausente: '{module_name}'. Instale com pip para usar este recurso."
        ) from exc


@dataclass
class Chunk:
    chunk_id: str
    source: str
    text: str


@dataclass
class RetrievalHit:
    chunk: Chunk
    score: float


class CorpusManager:
    @staticmethod
    def download_hf_text_dataset(
        dataset_name: str,
        output_file: Path,
        text_field: str = "text",
        split: str = "train",
        target_gb: float = 10.0,
    ) -> dict:
        datasets = _lazy_import("datasets")

        output_file.parent.mkdir(parents=True, exist_ok=True)
        target_bytes = int(target_gb * (1024**3))
        written = 0
        rows = 0

        with output_file.open("w", encoding="utf-8") as handle:
            stream = datasets.load_dataset(dataset_name, split=split, streaming=True)
            for item in stream:
                text = (item.get(text_field) or "").strip()
                if not text:
                    continue
                line = text.replace("\n", " ").strip() + "\n"
                encoded = line.encode("utf-8")
                handle.write(line)
                written += len(encoded)
                rows += 1
                if written >= target_bytes:
                    break

        return {
            "dataset": dataset_name,
            "split": split,
            "output": str(output_file),
            "rows": rows,
            "written_bytes": written,
            "target_bytes": target_bytes,
            "completed": written >= target_bytes,
        }

    @staticmethod
    def load_chunks_from_txt(data_dir: Path, chunk_size_chars: int = 1200, overlap_chars: int = 200) -> list[Chunk]:
        chunks: list[Chunk] = []
        txt_files = sorted(data_dir.rglob("*.txt"))
        for file_path in txt_files:
            raw_text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
            for idx, text in enumerate(chunk_text(raw_text, chunk_size_chars, overlap_chars)):
                chunks.append(
                    Chunk(
                        chunk_id=f"{file_path.as_posix()}::{idx}",
                        source=file_path.as_posix(),
                        text=text,
                    )
                )
        return chunks


def chunk_text(text: str, chunk_size_chars: int, overlap_chars: int) -> list[str]:
    if chunk_size_chars <= 0:
        raise ValueError("chunk_size_chars deve ser maior que zero")
    if overlap_chars < 0 or overlap_chars >= chunk_size_chars:
        raise ValueError("overlap_chars deve estar entre 0 e chunk_size_chars - 1")

    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size_chars, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap_chars
    return chunks


class SemanticSearcher:
    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("Nenhum chunk disponível para indexação")
        self.chunks = chunks
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.word2vec_model = None
        self.word2vec_chunk_vectors: list[list[float]] = []
        self.sentence_model = None
        self.sentence_vectors = None
        self.hnsw_index = None
        self.available_methods: set[str] = set()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [t.lower() for t in text.split() if t.strip()]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)

    def build_tfidf(self) -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.tfidf_vectorizer = TfidfVectorizer(max_features=50000)
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform([c.text for c in self.chunks])
        except ImportError:
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
            self._fallback_doc_bow = [Counter(self._tokenize(c.text)) for c in self.chunks]
            warnings.warn(
                "scikit-learn ausente; usando fallback simples bag-of-words para método cosine.",
                RuntimeWarning,
            )
        self.available_methods.add("cosine")

    def build_word2vec_average(self, vector_size: int = 200, epochs: int = 10) -> None:
        gensim_models = _lazy_import("gensim.models")
        tokenized = [self._tokenize(c.text) for c in self.chunks]
        self.word2vec_model = gensim_models.Word2Vec(
            sentences=tokenized,
            vector_size=vector_size,
            window=5,
            min_count=1,
            workers=1,
            epochs=epochs,
        )
        self.word2vec_chunk_vectors = [self._avg_w2v(tokens) for tokens in tokenized]
        self.available_methods.add("word2vec_average")

    def _avg_w2v(self, tokens: list[str]) -> list[float]:
        if self.word2vec_model is None:
            return []
        vectors = [self.word2vec_model.wv[t] for t in tokens if t in self.word2vec_model.wv]
        if not vectors:
            return [0.0] * self.word2vec_model.vector_size
        dim = len(vectors[0])
        return [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]

    def build_sentence_embeddings(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        sentence_transformers = _lazy_import("sentence_transformers")
        self.sentence_model = sentence_transformers.SentenceTransformer(model_name)
        self.sentence_vectors = self.sentence_model.encode(
            [c.text for c in self.chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.available_methods.add("sentence_embeddings")

    def build_hnsw(self, space: str = "cosine", ef_construction: int = 200, m: int = 16) -> None:
        if self.sentence_vectors is None:
            raise RuntimeError("Execute build_sentence_embeddings antes de build_hnsw")
        hnswlib = _lazy_import("hnswlib")
        dim = len(self.sentence_vectors[0])
        self.hnsw_index = hnswlib.Index(space=space, dim=dim)
        self.hnsw_index.init_index(max_elements=len(self.chunks), ef_construction=ef_construction, M=m)
        self.hnsw_index.add_items(self.sentence_vectors, list(range(len(self.chunks))))
        self.hnsw_index.set_ef(100)
        self.available_methods.add("hnsw")

    def search(self, query: str, method: str, top_k: int = 5) -> list[RetrievalHit]:
        if self.available_methods and method not in self.available_methods:
            raise RuntimeError(
                f"Método '{method}' indisponível neste índice. Disponíveis: {sorted(self.available_methods)}"
            )
        if method == "cosine":
            return self._search_tfidf(query, top_k)
        if method == "word2vec_average":
            return self._search_word2vec(query, top_k)
        if method == "sentence_embeddings":
            return self._search_sentence_embeddings(query, top_k)
        if method == "hnsw":
            return self._search_hnsw(query, top_k)
        raise ValueError(f"Método não suportado: {method}")

    def _search_tfidf(self, query: str, top_k: int) -> list[RetrievalHit]:
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            self.build_tfidf()
        if self.tfidf_vectorizer is not None and self.tfidf_matrix is not None:
            qv = self.tfidf_vectorizer.transform([query])
            scores = (self.tfidf_matrix @ qv.T).toarray().ravel()
            best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            return [RetrievalHit(self.chunks[i], float(scores[i])) for i in best]

        qbow = Counter(self._tokenize(query))
        scores = [self._counter_cosine(qbow, doc_counter) for doc_counter in self._fallback_doc_bow]
        best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [RetrievalHit(self.chunks[i], float(scores[i])) for i in best]

    @staticmethod
    def _counter_cosine(a: Counter, b: Counter) -> float:
        keys = set(a) | set(b)
        av = [float(a.get(k, 0.0)) for k in keys]
        bv = [float(b.get(k, 0.0)) for k in keys]
        return SemanticSearcher._cosine(av, bv)

    def _search_word2vec(self, query: str, top_k: int) -> list[RetrievalHit]:
        if self.word2vec_model is None or not self.word2vec_chunk_vectors:
            self.build_word2vec_average()
        qv = self._avg_w2v(self._tokenize(query))
        scores = [self._cosine(qv, cv) for cv in self.word2vec_chunk_vectors]
        best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [RetrievalHit(self.chunks[i], float(scores[i])) for i in best]

    def _search_sentence_embeddings(self, query: str, top_k: int) -> list[RetrievalHit]:
        if self.sentence_model is None or self.sentence_vectors is None:
            self.build_sentence_embeddings()
        qv = self.sentence_model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = [float(sum(a * b for a, b in zip(vec, qv))) for vec in self.sentence_vectors]
        best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [RetrievalHit(self.chunks[i], float(scores[i])) for i in best]

    def _search_hnsw(self, query: str, top_k: int) -> list[RetrievalHit]:
        if self.hnsw_index is None:
            if self.sentence_model is None or self.sentence_vectors is None:
                self.build_sentence_embeddings()
            self.build_hnsw()
        qv = self.sentence_model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        labels, distances = self.hnsw_index.knn_query(qv, k=top_k)
        result = []
        for idx, dist in zip(labels[0], distances[0]):
            score = 1.0 - float(dist)
            result.append(RetrievalHit(self.chunks[int(idx)], score))
        return result


class LocalFormatterLLM:
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self._generator = None

    def _load_generator(self):
        if not self.model_path:
            return None
        transformers = _lazy_import("transformers")
        torch = _lazy_import("torch")

        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        self._generator = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=300,
            do_sample=False,
        )
        return self._generator

    def format_answer(self, question: str, hits: list[RetrievalHit]) -> str:
        if not hits:
            return "Não encontrei contexto relevante para responder."

        contexts = "\n\n".join(
            f"[{idx+1}] Fonte: {hit.chunk.source}\nTrecho: {hit.chunk.text[:700]}"
            for idx, hit in enumerate(hits)
        )

        prompt = (
            "Responda à pergunta com base apenas nos trechos abaixo. "
            "A resposta deve ser curta, factual e citar as fontes no formato [n].\n\n"
            f"Pergunta: {question}\n\n"
            f"Trechos:\n{contexts}\n\n"
            "Resposta:"
        )

        generator = self._generator or self._load_generator()
        if generator is None:
            citations = " ".join(f"[{i+1}]" for i in range(len(hits)))
            return f"Resumo baseado nos trechos recuperados {citations}:\n" + "\n".join(
                f"- {hit.chunk.text[:200].strip()}" for hit in hits[:3]
            )

        output = generator(prompt)[0]["generated_text"]
        return output[len(prompt) :].strip()


def benchmark_methods(searcher: SemanticSearcher, queries: Iterable[str], top_k: int = 5) -> dict:
    methods = ["cosine", "word2vec_average", "sentence_embeddings", "hnsw"]
    report = {}
    for method in methods:
        if searcher.available_methods and method not in searcher.available_methods:
            report[method] = {"skipped": True, "reason": "método não indexado/disponível"}
            continue
        latencies = []
        for q in queries:
            start = time.perf_counter()
            _ = searcher.search(q, method=method, top_k=top_k)
            latencies.append((time.perf_counter() - start) * 1000)
        report[method] = {
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
            "min_latency_ms": min(latencies) if latencies else 0.0,
            "max_latency_ms": max(latencies) if latencies else 0.0,
            "queries": len(latencies),
        }
    return report


def run_ragas_evaluation(eval_file: Path) -> dict:
    datasets = _lazy_import("datasets")
    ragas = _lazy_import("ragas")
    ragas_metrics = _lazy_import("ragas.metrics")

    payload = json.loads(eval_file.read_text(encoding="utf-8"))
    dataset = datasets.Dataset.from_list(payload)

    metrics = [
        ragas_metrics.answer_relevancy,
        ragas_metrics.faithfulness,
        ragas_metrics.context_precision,
        ragas_metrics.context_recall,
    ]
    result = ragas.evaluate(dataset=dataset, metrics=metrics)
    return result.to_pandas().mean(numeric_only=True).to_dict()


def save_index(path: Path, searcher: SemanticSearcher) -> None:
    with path.open("wb") as handle:
        pickle.dump(searcher, handle)


def load_index(path: Path) -> SemanticSearcher:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _cmd_download(args: argparse.Namespace) -> None:
    result = CorpusManager.download_hf_text_dataset(
        dataset_name=args.dataset,
        output_file=Path(args.output),
        text_field=args.text_field,
        split=args.split,
        target_gb=args.target_gb,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _cmd_build_index(args: argparse.Namespace) -> None:
    chunks = CorpusManager.load_chunks_from_txt(Path(args.data_dir), args.chunk_size, args.overlap)
    searcher = SemanticSearcher(chunks)
    searcher.build_tfidf()
    for method_name, builder in (
        ("word2vec_average", lambda: searcher.build_word2vec_average()),
        ("sentence_embeddings", lambda: searcher.build_sentence_embeddings(args.embedding_model)),
        ("hnsw", lambda: searcher.build_hnsw()),
    ):
        try:
            builder()
        except RuntimeError as exc:
            warnings.warn(f"Não foi possível construir índice {method_name}: {exc}", RuntimeWarning)
    save_index(Path(args.index_file), searcher)
    print(
        f"Index salvo em: {args.index_file} ({len(chunks)} chunks). "
        f"Métodos disponíveis: {sorted(searcher.available_methods)}"
    )


def _cmd_ask(args: argparse.Namespace) -> None:
    searcher = load_index(Path(args.index_file))
    hits = searcher.search(args.question, method=args.method, top_k=args.top_k)
    formatter = LocalFormatterLLM(args.local_llm_path)
    answer = formatter.format_answer(args.question, hits)
    print("\n=== RESPOSTA ===")
    print(answer)
    print("\n=== CITAÇÕES ===")
    for i, hit in enumerate(hits, start=1):
        print(f"[{i}] {hit.chunk.source} | score={hit.score:.4f}")


def _cmd_benchmark(args: argparse.Namespace) -> None:
    searcher = load_index(Path(args.index_file))
    queries = [q.strip() for q in Path(args.queries_file).read_text(encoding="utf-8").splitlines() if q.strip()]
    report = benchmark_methods(searcher, queries, top_k=args.top_k)
    _write_json(Path(args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _cmd_evaluate(args: argparse.Namespace) -> None:
    result = run_ragas_evaluation(Path(args.eval_file))
    _write_json(Path(args.output), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Buscador semântico com citação e formatação via LLM local")
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download", help="Baixa texto em massa de um dataset do Hugging Face")
    download.add_argument("--dataset", required=True)
    download.add_argument("--split", default="train")
    download.add_argument("--text-field", default="text")
    download.add_argument("--target-gb", type=float, default=10.0)
    download.add_argument("--output", required=True)
    download.set_defaults(func=_cmd_download)

    build_index = sub.add_parser("build-index", help="Indexa textos e prepara todos os métodos de busca")
    build_index.add_argument("--data-dir", required=True)
    build_index.add_argument("--index-file", default="artifacts/index.pkl")
    build_index.add_argument("--chunk-size", type=int, default=1200)
    build_index.add_argument("--overlap", type=int, default=200)
    build_index.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    build_index.set_defaults(func=_cmd_build_index)

    ask = sub.add_parser("ask", help="Responde pergunta com base na busca semântica e mostra citação")
    ask.add_argument("--index-file", default="artifacts/index.pkl")
    ask.add_argument("--question", required=True)
    ask.add_argument("--method", choices=["cosine", "word2vec_average", "sentence_embeddings", "hnsw"], default="hnsw")
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--local-llm-path", default=None)
    ask.set_defaults(func=_cmd_ask)

    benchmark = sub.add_parser("benchmark", help="Compara latência dos quatro métodos")
    benchmark.add_argument("--index-file", default="artifacts/index.pkl")
    benchmark.add_argument("--queries-file", required=True)
    benchmark.add_argument("--top-k", type=int, default=5)
    benchmark.add_argument("--output", default="artifacts/benchmark.json")
    benchmark.set_defaults(func=_cmd_benchmark)

    evaluate = sub.add_parser("evaluate", help="Avalia com métricas RAGAS")
    evaluate.add_argument("--eval-file", required=True)
    evaluate.add_argument("--output", default="artifacts/ragas_metrics.json")
    evaluate.set_defaults(func=_cmd_evaluate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

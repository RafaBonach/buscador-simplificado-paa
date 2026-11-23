"""Linha de comando: download, build-index, ask, chat, benchmark, evaluate, sample, stats, download-models."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

from . import __version__
from .corpus import DEFAULT_SOURCES, GIB, download_corpus, iter_documents
from .encoders import DEFAULT_EMBEDDING_MODEL, MissingDependency
from .index import METHODS, BuildConfig, build_index, load_meta


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------- comandos
def cmd_download(args: argparse.Namespace) -> None:
    result = download_corpus(
        sources=args.source or DEFAULT_SOURCES, out_dir=Path(args.output), target_gb=args.target_gb,
        split=args.split, text_field=args.text_field, shard_mb=args.shard_mb,
    )
    _print_json(result)
    if not result["completed"]:
        sys.exit(1)


def cmd_build_index(args: argparse.Namespace) -> None:
    cfg = BuildConfig(
        chunk_size=args.chunk_size, overlap=args.overlap, methods=tuple(args.methods), max_chunks=args.max_chunks,
        max_docs=args.max_docs, embedding_model=args.embedding_model, embed_batch=args.embed_batch,
        max_seq_length=args.max_seq_length, device=args.device, w2v_dim=args.w2v_dim, w2v_epochs=args.w2v_epochs,
        w2v_train_chunks=args.w2v_train_chunks, w2v_workers=args.w2v_workers, hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction, hnsw_ef_search=args.hnsw_ef_search,
        hnsw_backend=args.hnsw_backend, min_doc_chars=args.min_doc_chars, force=args.force,
    )
    meta = build_index([Path(p) for p in args.data], Path(args.index_dir), cfg)
    print(f"\nÍndice em {args.index_dir}: {meta['ingest']['docs']} documentos, {meta['ingest']['chunks']} chunks.")
    print(f"Métodos disponíveis: {meta['available_methods']}")
    if meta.get("skipped"):
        print(f"Métodos ignorados: {json.dumps(meta['skipped'], ensure_ascii=False)}")


def _load_qa(args: argparse.Namespace):
    from .qa import QASystem

    return QASystem.load(Path(args.index_dir), llm=args.llm, max_new_tokens=args.max_new_tokens, device=args.device)


def cmd_ask(args: argparse.Namespace) -> None:
    qa = _load_qa(args)
    answer = qa.ask(args.question, method=args.method, top_k=args.top_k)
    if args.json:
        _print_json(answer.to_dict())
    else:
        print(answer.render())


def cmd_chat(args: argparse.Namespace) -> None:
    qa = _load_qa(args)
    method = args.method or qa.default_method()
    print(f"Carregando o método {method}...")
    qa.searcher.warmup(method)  # a 1ª consulta paga o carregamento; os tempos exibidos depois são de regime
    print(f"Método: {method} | gerador: {qa.formatter.name}. Linha vazia ou Ctrl-D encerra.")
    while True:
        try:
            question = input("\nPergunta> ").strip()
        except EOFError:
            break
        if not question:
            break
        print(qa.ask(question, method=args.method, top_k=args.top_k).render())


def _load_items(args: argparse.Namespace) -> list[dict]:
    from .evaluate import load_questions

    if args.eval_file:
        return load_questions(Path(args.eval_file))
    if args.queries_file:
        lines = Path(args.queries_file).read_text(encoding="utf-8").splitlines()
        return [{"question": line.strip()} for line in lines if line.strip()]
    raise SystemExit("Informe --eval-file (JSON) ou --queries-file (uma pergunta por linha).")


def cmd_benchmark(args: argparse.Namespace) -> None:
    from .benchmark import render_markdown, run_benchmark

    report = run_benchmark(Path(args.index_dir), _load_items(args), methods=args.methods, top_k=args.top_k,
                           repeats=args.repeats, device=args.device)
    _write_json(Path(args.output), report)
    markdown = render_markdown(report)
    Path(args.output).with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"JSON: {args.output} | tabela: {Path(args.output).with_suffix('.md')}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    from .evaluate import evaluate_system, load_questions

    qa = _load_qa(args)
    result = evaluate_system(qa, load_questions(Path(args.questions)), method=args.method, top_k=args.top_k)
    if args.ragas:
        from .ragas_eval import run_ragas

        result["ragas"] = run_ragas(
            result["rows"], judge_model_path=args.judge_model_path, judge_base_url=args.judge_base_url,
            judge_model=args.judge_model, embedding_model=args.ragas_embedding_model,
            max_workers=args.ragas_workers, timeout=args.ragas_timeout,
        )
    _write_json(Path(args.output), result)
    _print_json({"method": result["method"], "generator": result["generator"], "summary": result["summary"],
                 **({"ragas": result["ragas"]["metrics"]} if "ragas" in result else {})})
    print(f"\nResultado completo: {args.output}")


def cmd_sample(args: argparse.Namespace) -> None:
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    limit = int(args.max_mb * 1024 * 1024)
    written = docs = 0
    with out.open("w", encoding="utf-8") as handle:
        for doc in iter_documents(Path(args.corpus)):
            handle.write(json.dumps({"id": doc.doc_id, "title": doc.title, "url": doc.url, "source": doc.source,
                                     "text": doc.text}, ensure_ascii=False) + "\n")
            written += len(doc.text.encode("utf-8"))
            docs += 1
            if written >= limit:
                break
    print(f"Amostra: {docs} documentos, {written/1e6:.1f} MB de texto em {out}")


def cmd_stats(args: argparse.Namespace) -> None:
    meta = load_meta(Path(args.index_dir))
    ing = meta["ingest"]
    print(f"Documentos: {ing['docs']} | chunks: {ing['chunks']} | caracteres: {ing['chars']:,} | ingestão: {ing['seconds']}s")
    print(f"{'método':22} {'build (s)':>10} {'tamanho (MB)':>13} {'s / 1M chunks':>14}")
    for method in METHODS:
        info = meta["methods"].get(method)
        if info:
            per_m = info["build_seconds"] / ing["chunks"] * 1e6
            print(f"{method:22} {info['build_seconds']:>10.1f} {info['bytes']/1e6:>13.1f} {per_m:>14.1f}")
        elif method in meta.get("skipped", {}):
            print(f"{method:22} ignorado: {meta['skipped'][method]}")
    if args.target_gb:
        avg_chunk = ing["chars"] / ing["chunks"]
        projected = int(args.target_gb * GIB / max(avg_chunk, 1))
        print(f"\nProjeção linear para {args.target_gb} GiB de texto (~{avg_chunk:.0f} caracteres/chunk): ~{projected:,} chunks")
        for method, info in meta["methods"].items():
            hours = info["build_seconds"] / ing["chunks"] * projected / 3600
            gb = info["bytes"] / ing["chunks"] * projected / 1e9
            print(f"  {method:22} ~{hours:8.1f} h   ~{gb:7.1f} GB em disco")
        print("(Extrapolação linear da máquina atual; HNSW é O(N log N) e o hardware muda os tempos.)")


def cmd_download_models(args: argparse.Namespace) -> None:
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError as exc:
        raise MissingDependency("Pacote 'huggingface_hub' ausente. Instale com: pip install -r requirements.txt") from exc
    from .llm import resolve_model

    for repo in [resolve_model(m) for m in [args.embedding_model, *args.llm] if m]:
        target = Path(args.output_dir) / repo.replace("/", "__") if args.output_dir else None
        path = snapshot_download(repo_id=repo, local_dir=str(target) if target else None)
        print(f"{repo} -> {path}")
    print("Para usar offline: export HF_HUB_OFFLINE=1 e passe o caminho acima em --llm / --embedding-model.")


# ---------------------------------------------------------------- parser
def _common_runtime(p: argparse.ArgumentParser) -> None:
    p.add_argument("--index-dir", default="artifacts/index")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--device", default=None, help="cpu, cuda ou mps (padrão: automático)")


def _llm_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--llm", default=None, help="LLM local: smollm, tinyllama, 'phi-3 mini', deepseek, mistral, um id do "
                                               "Hugging Face ou uma pasta. Sem isso usa o formatador extrativo local.")
    p.add_argument("--max-new-tokens", type=int, default=256)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paa_qa", description="Buscador semântico com citações e LLM local")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="Baixa texto (Hugging Face, streaming, retomável) até a meta em GiB")
    p.add_argument("--source", action="append", help="dataset[:config], repetível. Padrão: Wikipédia pt + en")
    p.add_argument("--split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--target-gb", type=float, default=10.0)
    p.add_argument("--shard-mb", type=int, default=128)
    p.add_argument("--output", default="data/corpus")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("build-index", help="Chunking + índices dos 4 métodos")
    p.add_argument("--data", "--data-dir", dest="data", nargs="+", required=True,
                   help="uma ou mais pastas/arquivos: .jsonl (download), wiki_* (WikiExtractor) ou .txt")
    p.add_argument("--min-doc-chars", type=int, default=200, help="ignora páginas menores que isso (padrão: 200)")
    p.add_argument("--hnsw-backend", choices=["auto", "hnswlib", "faiss"], default="auto",
                   help="auto = hnswlib se instalado, senão faiss-cpu")
    p.add_argument("--index-dir", default="artifacts/index")
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--chunk-size", type=int, default=800)
    p.add_argument("--overlap", type=int, default=120)
    p.add_argument("--max-chunks", type=int, default=None, help="orçamento de chunks (recursos limitados)")
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL,
                   help="modelo sentence-transformers; 'hashing' = codificador de teste, não semântico")
    p.add_argument("--embed-batch", type=int, default=64)
    p.add_argument("--max-seq-length", type=int, default=256)
    p.add_argument("--device", default=None)
    p.add_argument("--w2v-dim", type=int, default=100)
    p.add_argument("--w2v-epochs", type=int, default=5)
    p.add_argument("--w2v-train-chunks", type=int, default=500_000)
    p.add_argument("--w2v-workers", type=int, default=2)
    p.add_argument("--hnsw-m", type=int, default=16)
    p.add_argument("--hnsw-ef-construction", type=int, default=200)
    p.add_argument("--hnsw-ef-search", type=int, default=64)
    p.add_argument("--force", action="store_true", help="apaga o índice existente e refaz tudo")
    p.set_defaults(func=cmd_build_index)

    p = sub.add_parser("ask", help="Responde uma pergunta com citações")
    _common_runtime(p)
    _llm_args(p)
    p.add_argument("--question", required=True)
    p.add_argument("--method", choices=METHODS, default=None, help="padrão: hnsw (ou o primeiro disponível)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("chat", help="Modo interativo")
    _common_runtime(p)
    _llm_args(p)
    p.add_argument("--method", choices=METHODS, default=None)
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("benchmark", help="Compara os métodos: latência, qualidade, build, tamanho, complexidade")
    _common_runtime(p)
    p.add_argument("--queries-file", help="uma pergunta por linha")
    p.add_argument("--eval-file", help="JSON com question e relevant_titles (habilita hit@k/MRR)")
    p.add_argument("--methods", nargs="+", choices=METHODS, default=None)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", default="artifacts/benchmark.json")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("evaluate", help="Avalia o sistema (métricas próprias; --ragas liga o RAGAS com juiz local)")
    _common_runtime(p)
    _llm_args(p)
    p.add_argument("--questions", required=True)
    p.add_argument("--method", choices=METHODS, default=None)
    p.add_argument("--output", default="artifacts/evaluation.json")
    p.add_argument("--ragas", action="store_true")
    p.add_argument("--judge-model-path", default=None, help="modelo HF local usado como juiz do RAGAS")
    p.add_argument("--judge-base-url", default=None, help="servidor local OpenAI-compatível (ex.: http://localhost:11434/v1)")
    p.add_argument("--judge-model", default=None, help="nome do modelo no servidor local")
    p.add_argument("--ragas-embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    p.add_argument("--ragas-workers", type=int, default=1)
    p.add_argument("--ragas-timeout", type=int, default=600)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("sample", help="Extrai uma amostra do corpus (para o zip de entrega, < 100 MB)")
    p.add_argument("--corpus", default="data/corpus")
    p.add_argument("--output", default="data/sample/amostra_corpus.jsonl")
    p.add_argument("--max-mb", type=float, default=40.0)
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("stats", help="Estatísticas do índice e projeção de recursos")
    p.add_argument("--index-dir", default="artifacts/index")
    p.add_argument("--target-gb", type=float, default=10.0)
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("download-models", help="Baixa modelos para uso offline")
    p.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    p.add_argument("--llm", nargs="*", default=["smollm"],
                   help="nomes do app (smollm, tinyllama, phi-3 mini, deepseek, mistral) ou ids do Hugging Face")
    p.add_argument("--output-dir", default=None)
    p.set_defaults(func=cmd_download_models)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # evita UnicodeEncodeError em consoles/redirecionamentos não-UTF-8
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    warnings.simplefilter("default", RuntimeWarning)
    try:
        args.func(args)
    except (MissingDependency, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        sys.exit(2)

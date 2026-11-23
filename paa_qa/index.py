"""Construção e consulta dos quatro métodos de busca semântica.

Layout de um índice (diretório):

    meta.json              parâmetros, tempos de construção e tamanhos
    chunks.db              SQLite com documentos e chunks (texto + citação)
    tfidf_terms_x_chunks.npz   matriz esparsa termo x chunk (= índice invertido ponderado por TF-IDF)
    tfidf_vectorizer.joblib
    w2v.model, w2v_vectors.npy  modelo Word2Vec e média dos vetores por chunk (float32, memmap)
    st_vectors.npy         embeddings de sentenças por chunk (float16, memmap)
    hnsw.bin               grafo HNSW sobre st_vectors

Complexidades (N chunks, d dimensões, k resultados, V = |vocabulário|):
    cosine (TF-IDF)       build O(T)             consulta O(soma das postings dos termos da pergunta)
    word2vec_average      build O(T·épocas·...)  consulta O(N·d)  (varredura exata)
    sentence_embeddings   build O(N·custo do transformer)  consulta O(custo do encode + N·d) (varredura exata)
    hnsw                  build O(N·log N·M·efC·d)         consulta O(custo do encode + ef·log N·d) (aproximada)
"""

from __future__ import annotations

import json
import shutil
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from .corpus import chunk_text, iter_documents
from .encoders import DEFAULT_EMBEDDING_MODEL, Encoder, MissingDependency, get_encoder
from .store import ChunkRecord, ChunkStore
from .text import STOPWORDS_PT, fold_accents, word_tokens

METHODS = ("cosine", "word2vec_average", "sentence_embeddings", "hnsw")
BLOCK_ROWS = 262_144


# ====================================================================== configuração
@dataclass
class BuildConfig:
    chunk_size: int = 800
    overlap: int = 120
    methods: tuple[str, ...] = METHODS
    max_chunks: int | None = None          # orçamento: para de ingerir ao atingir este nº de chunks
    max_docs: int | None = None
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embed_batch: int = 64
    max_seq_length: int = 256
    device: str | None = None
    tfidf_max_features: int = 200_000
    tfidf_min_df: int | None = None        # None = automático (1 se N < 200k, senão 2)
    w2v_dim: int = 100
    w2v_epochs: int = 5
    w2v_window: int = 5
    w2v_min_count: int | None = None       # None = automático (1 se treino < 100k chunks, senão 5)
    w2v_train_chunks: int = 500_000
    w2v_workers: int = 2
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64
    hnsw_backend: str = "auto"             # "auto" (hnswlib, senão faiss) | "hnswlib" | "faiss"
    min_doc_chars: int = 200               # descarta páginas menores que isso (esboços, desambiguações)
    force: bool = False


@dataclass
class Hit:
    rank: int
    score: float
    chunk: ChunkRecord


class IndexPaths:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.meta = self.root / "meta.json"
        self.db = self.root / "chunks.db"
        self.tfidf_matrix = self.root / "tfidf_terms_x_chunks.npz"
        self.tfidf_vectorizer = self.root / "tfidf_vectorizer.joblib"
        self.w2v_model = self.root / "w2v.model"
        self.w2v_vectors = self.root / "w2v_vectors.npy"
        self.st_vectors = self.root / "st_vectors.npy"
        self.st_progress = self.root / "st_progress.json"
        self.hnsw = self.root / "hnsw.bin"

    def files_for(self, method: str) -> list[Path]:
        return {
            "cosine": [self.tfidf_matrix, self.tfidf_vectorizer],
            "word2vec_average": [self.w2v_model, self.w2v_vectors],
            "sentence_embeddings": [self.st_vectors],
            "hnsw": [self.hnsw],
        }[method]


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_meta(index_dir: Path) -> dict:
    path = IndexPaths(index_dir).meta
    if not path.exists():
        raise FileNotFoundError(f"Índice não encontrado em {index_dir}. Rode 'build-index' primeiro.")
    return json.loads(path.read_text(encoding="utf-8"))


def _size(files: list[Path]) -> int:
    return sum(f.stat().st_size for f in files if f.exists())


# ====================================================================== utilidades numéricas
def top_k_dense(matrix: np.ndarray, query: np.ndarray, k: int, block: int = BLOCK_ROWS) -> tuple[np.ndarray, np.ndarray]:
    """Top-k por produto interno, varrendo `matrix` (pode ser memmap) em blocos. O(N·d)."""
    n = matrix.shape[0]
    k = min(k, n)
    if k <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    best_ids = np.empty(0, dtype=np.int64)
    best_scores = np.empty(0, dtype=np.float32)
    q = np.asarray(query, dtype=np.float32)
    for start in range(0, n, block):
        part = np.asarray(matrix[start : start + block], dtype=np.float32) @ q
        kk = min(k, part.shape[0])
        idx = np.argpartition(-part, kk - 1)[:kk]
        best_ids = np.concatenate([best_ids, idx.astype(np.int64) + start])
        best_scores = np.concatenate([best_scores, part[idx]])
        if best_ids.size > k:
            keep = np.argpartition(-best_scores, k - 1)[:k]
            best_ids, best_scores = best_ids[keep], best_scores[keep]
    order = np.argsort(-best_scores, kind="stable")
    return best_ids[order], best_scores[order]


def average_vectors(token_lists: list[list[str]], key_to_index: dict, matrix: np.ndarray, dim: int) -> np.ndarray:
    """Média dos vetores de palavras (Word2Vec Average), normalizada em L2. Vazio -> vetor nulo."""
    out = np.zeros((len(token_lists), dim), dtype=np.float32)
    for row, tokens in enumerate(token_lists):
        idxs = [key_to_index[t] for t in tokens if t in key_to_index]
        if idxs:
            out[row] = matrix[idxs].mean(axis=0)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


def _top_k_sparse(row, k: int) -> tuple[np.ndarray, np.ndarray]:
    row = row.tocsr()
    ids, vals = row.indices, row.data
    if ids.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    kk = min(k, ids.size)
    top = np.argpartition(-vals, kk - 1)[:kk]
    order = top[np.argsort(-vals[top], kind="stable")]
    return ids[order].astype(np.int64), vals[order].astype(np.float32)


# ====================================================================== construção
def build_index(data_path: "Path | list[Path]", index_dir: Path, cfg: BuildConfig,
                log: Callable[[str], None] = print) -> dict:
    """`data_path`: uma pasta/arquivo ou uma lista delas (ex.: pt e en do WikiExtractor), lidas na ordem dada."""
    index_dir = Path(index_dir)
    paths = IndexPaths(index_dir)
    if cfg.force and index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads(paths.meta.read_text(encoding="utf-8")) if paths.meta.exists() else {
        "version": 1, "params": {}, "ingest": None, "methods": {}, "skipped": {},
    }

    # ---- 1) ingestão: documentos -> chunks no SQLite
    if meta.get("ingest"):
        log(f"[build] ingestão já concluída ({meta['ingest']['chunks']} chunks) — reutilizando. Use --force para refazer.")
    else:
        t0 = time.perf_counter()
        if paths.db.exists():
            paths.db.unlink()
        store = ChunkStore(paths.db)
        n_chunks = n_docs = 0
        for doc in iter_documents(data_path, limit_docs=cfg.max_docs, min_chars=cfg.min_doc_chars):
            pieces = chunk_text(doc.text, cfg.chunk_size, cfg.overlap)
            if not pieces:
                continue
            n_chunks += store.add_document(doc.title, doc.url, doc.source, doc.doc_id, pieces)
            n_docs += 1
            if n_docs % 2000 == 0:
                store.commit()
                log(f"[build] ingestão: {n_docs} docs, {n_chunks} chunks")
            if cfg.max_chunks and n_chunks >= cfg.max_chunks:
                log(f"[build] orçamento --max-chunks atingido ({n_chunks}).")
                break
        store.commit()
        chars = store.total_chars()
        store.close()
        if n_chunks == 0:
            raise ValueError(f"Nenhum texto encontrado em {data_path}")
        meta["ingest"] = {"docs": n_docs, "chunks": n_chunks, "chars": chars,
                          "seconds": round(time.perf_counter() - t0, 2), "bytes": paths.db.stat().st_size}
        meta["params"] = {k: v for k, v in asdict(cfg).items() if k in ("chunk_size", "overlap", "max_chunks", "max_docs", "min_doc_chars")}
        _write_json(paths.meta, meta)
        log(f"[build] ingestão: {n_docs} docs, {n_chunks} chunks em {meta['ingest']['seconds']}s")

    n = int(meta["ingest"]["chunks"])
    store = ChunkStore(paths.db, readonly=True)
    requested = list(dict.fromkeys(cfg.methods))
    if "hnsw" in requested and "sentence_embeddings" not in requested:
        requested.append("sentence_embeddings")
    order = [m for m in METHODS if m in requested]

    builders = {
        "cosine": _build_tfidf,
        "word2vec_average": _build_word2vec,
        "sentence_embeddings": _build_sentence_embeddings,
        "hnsw": _build_hnsw,
    }
    for method in order:
        if method in meta["methods"] and all(f.exists() for f in paths.files_for(method)) and not cfg.force:
            log(f"[build] {method}: já construído — pulando.")
            continue
        if method == "hnsw" and "sentence_embeddings" not in meta["methods"]:
            meta["skipped"][method] = "sentence_embeddings indisponível"
            warnings.warn("hnsw ignorado: depende dos embeddings de sentenças.", RuntimeWarning)
            continue
        log(f"[build] construindo {method} ...")
        t0 = time.perf_counter()
        try:
            info = builders[method](store, paths, cfg, n, meta, log)
        except MissingDependency as exc:
            meta["skipped"][method] = str(exc)
            warnings.warn(f"Método '{method}' não construído: {exc}", RuntimeWarning)
            _write_json(paths.meta, meta)
            continue
        info["build_seconds"] = round(time.perf_counter() - t0, 2)
        info["bytes"] = _size(paths.files_for(method))
        meta["methods"][method] = info
        meta["skipped"].pop(method, None)
        _write_json(paths.meta, meta)
        log(f"[build] {method}: {info['build_seconds']}s, {info['bytes']/1e6:.1f} MB")
    store.close()
    meta["available_methods"] = [m for m in METHODS if m in meta["methods"]]
    _write_json(paths.meta, meta)
    return meta


def _build_tfidf(store: ChunkStore, paths: IndexPaths, cfg: BuildConfig, n: int, meta: dict, log) -> dict:
    import joblib
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import TfidfVectorizer

    min_df = cfg.tfidf_min_df if cfg.tfidf_min_df is not None else (1 if n < 200_000 else 2)
    stop = sorted({fold_accents(w) for w in STOPWORDS_PT if len(fold_accents(w)) > 1})
    vec = TfidfVectorizer(lowercase=True, strip_accents="unicode", stop_words=stop, sublinear_tf=True,
                          min_df=min_df, max_features=cfg.tfidf_max_features, dtype=np.float32)
    texts = (t for _, batch in store.iter_texts(0, n) for t in batch)
    matrix = vec.fit_transform(texts)  # N x V, linhas normalizadas em L2
    inverted = matrix.T.tocsr()        # V x N: linha = lista de postings do termo
    sp.save_npz(paths.tfidf_matrix, inverted, compressed=False)
    joblib.dump(vec, paths.tfidf_vectorizer)
    return {"vocab": int(len(vec.vocabulary_)), "nnz": int(matrix.nnz), "min_df": min_df}


class _ChunkTokens:
    """Iterável reutilizável de listas de tokens (o gensim itera o corpus várias vezes)."""

    def __init__(self, db_path: Path, limit: int):
        self.db_path, self.limit = db_path, limit

    def __iter__(self) -> Iterator[list[str]]:
        store = ChunkStore(self.db_path, readonly=True)
        try:
            for _, batch in store.iter_texts(0, self.limit):
                for text in batch:
                    yield word_tokens(text)
        finally:
            store.close()


def _build_word2vec(store: ChunkStore, paths: IndexPaths, cfg: BuildConfig, n: int, meta: dict, log) -> dict:
    try:
        from gensim.models import Word2Vec  # type: ignore
    except ImportError as exc:
        raise MissingDependency("Pacote 'gensim' ausente. Instale com: pip install -r requirements.txt") from exc

    train_n = min(n, cfg.w2v_train_chunks)
    min_count = cfg.w2v_min_count if cfg.w2v_min_count is not None else (1 if train_n < 100_000 else 5)
    log(f"[build] word2vec: treinando em {train_n} chunks (dim={cfg.w2v_dim}, épocas={cfg.w2v_epochs}, min_count={min_count})")
    model = Word2Vec(sentences=_ChunkTokens(paths.db, train_n), vector_size=cfg.w2v_dim, window=cfg.w2v_window,
                     min_count=min_count, workers=cfg.w2v_workers, epochs=cfg.w2v_epochs, seed=42)
    model.save(str(paths.w2v_model))
    wv = model.wv
    vectors = np.lib.format.open_memmap(paths.w2v_vectors, mode="w+", dtype=np.float32, shape=(n, cfg.w2v_dim))
    for start, batch in store.iter_texts(0, n, batch=5000):
        tokens = [word_tokens(t) for t in batch]
        vectors[start : start + len(batch)] = average_vectors(tokens, wv.key_to_index, wv.vectors, cfg.w2v_dim)
    vectors.flush()
    return {"dim": cfg.w2v_dim, "vocab": int(len(wv.key_to_index)), "train_chunks": train_n,
            "epochs": cfg.w2v_epochs, "min_count": min_count}


def _build_sentence_embeddings(store: ChunkStore, paths: IndexPaths, cfg: BuildConfig, n: int, meta: dict, log) -> dict:
    encoder = get_encoder(cfg.embedding_model, cfg.device, cfg.embed_batch, cfg.max_seq_length)
    dim = encoder.dim
    done = 0
    if paths.st_progress.exists() and paths.st_vectors.exists():
        prog = json.loads(paths.st_progress.read_text(encoding="utf-8"))
        if prog.get("model") == encoder.name and prog.get("dim") == dim and prog.get("n") == n:
            done = int(prog["done"])
            log(f"[build] embeddings: retomando de {done}/{n}")
    mode = "r+" if done else "w+"
    vectors = np.lib.format.open_memmap(paths.st_vectors, mode=mode, dtype=np.float16, shape=(n, dim))
    step = cfg.embed_batch * 16
    t0 = time.perf_counter()
    for start in range(done, n, step):
        stop = min(start + step, n)
        texts = [t for _, batch in store.iter_texts(start, stop, batch=stop - start) for t in batch]
        vectors[start:stop] = encoder.encode(texts).astype(np.float16)
        vectors.flush()
        _write_json(paths.st_progress, {"model": encoder.name, "dim": dim, "n": n, "done": stop})
        rate = (stop - done) / max(time.perf_counter() - t0, 1e-9)
        log(f"[build] embeddings: {stop}/{n} ({rate:.1f} chunks/s)")
    paths.st_progress.unlink(missing_ok=True)
    return {"model": encoder.name, "dim": dim, "dtype": "float16", "max_seq_length": cfg.max_seq_length}


def _pick_hnsw_backend(preferred: str) -> str:
    """hnswlib (precisa compilar no Windows) ou faiss (pacote pronto). Ambos implementam HNSW."""
    if preferred not in ("auto", "hnswlib", "faiss"):
        raise ValueError(f"hnsw_backend inválido: {preferred!r}. Use auto, hnswlib ou faiss.")
    candidates = ["hnswlib", "faiss"] if preferred == "auto" else [preferred]
    for name in candidates:
        try:
            __import__(name)
            return name
        except ImportError:
            continue
    raise MissingDependency(
        "HNSW precisa de 'faiss-cpu' (pip install faiss-cpu; tem pacote pronto para Windows) ou de 'hnswlib' "
        "(precisa de compilador C++ no Windows)."
    )


def _normalized(block: np.ndarray) -> np.ndarray:
    block = np.ascontiguousarray(block, dtype=np.float32)
    norms = np.linalg.norm(block, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return block / norms


def _build_hnsw(store: ChunkStore, paths: IndexPaths, cfg: BuildConfig, n: int, meta: dict, log) -> dict:
    backend = _pick_hnsw_backend(cfg.hnsw_backend)
    dim = int(meta["methods"]["sentence_embeddings"]["dim"])
    vectors = np.load(paths.st_vectors, mmap_mode="r")
    step = 50_000
    if backend == "hnswlib":
        import hnswlib  # type: ignore

        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(max_elements=n, ef_construction=cfg.hnsw_ef_construction, M=cfg.hnsw_m)
        for start in range(0, n, step):
            stop = min(start + step, n)
            index.add_items(np.asarray(vectors[start:stop], dtype=np.float32), np.arange(start, stop))
            log(f"[build] hnsw (hnswlib): {stop}/{n}")
        index.save_index(str(paths.hnsw))
    else:
        import faiss  # type: ignore

        # Vetores normalizados: produto interno = cosseno. Os ids são a ordem de inserção (= chunk_id).
        index = faiss.IndexHNSWFlat(dim, cfg.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = cfg.hnsw_ef_construction
        for start in range(0, n, step):
            stop = min(start + step, n)
            index.add(_normalized(vectors[start:stop]))
            log(f"[build] hnsw (faiss): {stop}/{n}")
        faiss.write_index(index, str(paths.hnsw))
    return {"backend": backend, "M": cfg.hnsw_m, "ef_construction": cfg.hnsw_ef_construction,
            "ef_search": cfg.hnsw_ef_search, "space": "cosine"}


# ====================================================================== consulta
class IndexSearcher:
    """Carrega um índice e consulta por qualquer um dos quatro métodos (componentes carregados sob demanda)."""

    def __init__(self, index_dir: Path, encoder: Encoder | None = None, device: str | None = None):
        self.paths = IndexPaths(index_dir)
        self.meta = load_meta(index_dir)
        self.store = ChunkStore(self.paths.db, readonly=True)
        self.n = int(self.meta["ingest"]["chunks"])
        self.device = device
        self._encoder = encoder
        self._tfidf = None
        self._w2v = None
        self._st_vectors = None
        self._hnsw = None

    @property
    def methods(self) -> list[str]:
        return [m for m in METHODS if m in self.meta["methods"]]

    def close(self) -> None:
        self.store.close()

    # -- carregamento preguiçoso
    def _get_encoder(self) -> Encoder:
        if self._encoder is None:
            info = self.meta["methods"]["sentence_embeddings"]
            self._encoder = get_encoder(info["model"], self.device, 32, info.get("max_seq_length", 256))
        return self._encoder

    def warmup(self, method: str) -> None:
        """Carrega modelos/matrizes do método para que a 1ª consulta não inclua o custo de carregamento."""
        self.search("aquecimento", method, 1)

    # -- busca
    def search(self, query: str, method: str, top_k: int = 5) -> list[Hit]:
        if method not in METHODS:
            raise ValueError(f"Método não suportado: {method}. Use um de {METHODS}")
        if method not in self.meta["methods"]:
            raise RuntimeError(f"Método '{method}' não está neste índice. Disponíveis: {self.methods}. "
                               f"Ignorados: {self.meta.get('skipped', {})}")
        ids, scores = getattr(self, f"_search_{method}")(query, top_k)
        records = self.store.get_chunks(ids.tolist())
        by_id = {r.chunk_id: r for r in records}
        hits = []
        for i, s in zip(ids.tolist(), scores.tolist()):
            if i in by_id:
                hits.append(Hit(rank=len(hits) + 1, score=float(s), chunk=by_id[i]))
        return hits

    def _search_cosine(self, query: str, k: int):
        if self._tfidf is None:
            import joblib
            import scipy.sparse as sp
            self._tfidf = (joblib.load(self.paths.tfidf_vectorizer), sp.load_npz(self.paths.tfidf_matrix).tocsr())
        vec, inverted = self._tfidf
        q = vec.transform([query])                 # 1 x V
        return _top_k_sparse(q @ inverted, k)      # custo ~ postings dos termos da pergunta

    def _search_word2vec_average(self, query: str, k: int):
        if self._w2v is None:
            from gensim.models import Word2Vec  # type: ignore
            model = Word2Vec.load(str(self.paths.w2v_model))
            self._w2v = (model.wv, np.load(self.paths.w2v_vectors, mmap_mode="r"))
        wv, vectors = self._w2v
        q = average_vectors([word_tokens(query)], wv.key_to_index, wv.vectors, vectors.shape[1])[0]
        if not q.any():
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        return top_k_dense(vectors, q, k)

    def _load_st(self):
        if self._st_vectors is None:
            self._st_vectors = np.load(self.paths.st_vectors, mmap_mode="r")
        return self._st_vectors

    def _search_sentence_embeddings(self, query: str, k: int):
        q = self._get_encoder().encode([query])[0]
        return top_k_dense(self._load_st(), q, k)

    def _search_hnsw(self, query: str, k: int):
        info = self.meta["methods"]["hnsw"]
        backend = info.get("backend", "hnswlib")  # índices antigos não tinham o campo
        ef = max(int(info.get("ef_search", 64)), k)
        k = min(k, self.n)
        if self._hnsw is None:
            if backend == "hnswlib":
                import hnswlib  # type: ignore

                index = hnswlib.Index(space="cosine", dim=int(self.meta["methods"]["sentence_embeddings"]["dim"]))
                index.load_index(str(self.paths.hnsw), max_elements=self.n)
            else:
                import faiss  # type: ignore

                index = faiss.read_index(str(self.paths.hnsw))
            self._hnsw = index
        q = self._get_encoder().encode([query])
        if backend == "hnswlib":
            self._hnsw.set_ef(ef)
            labels, distances = self._hnsw.knn_query(q, k=k)
            return labels[0].astype(np.int64), (1.0 - distances[0]).astype(np.float32)
        self._hnsw.hnsw.efSearch = ef
        scores, labels = self._hnsw.search(_normalized(q), k)
        keep = labels[0] >= 0  # o faiss completa com -1 quando há menos resultados que k
        return labels[0][keep].astype(np.int64), scores[0][keep].astype(np.float32)

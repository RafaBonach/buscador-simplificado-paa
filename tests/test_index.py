from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from paa_qa import index as index_mod
from paa_qa.encoders import HashingEncoder
from paa_qa.evaluate import retrieval_metrics
from paa_qa.index import (BuildConfig, IndexSearcher, _top_k_sparse, average_vectors, build_index, load_meta,
                          top_k_dense)

from .conftest import DEMO_CORPUS, write_corpus

quiet = lambda _msg: None  # noqa: E731


# ------------------------------------------------------------------ funções numéricas
def test_top_k_dense_igual_a_forca_bruta_em_blocos():
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(103, 8)).astype(np.float32)
    query = rng.normal(size=8).astype(np.float32)
    ids, scores = top_k_dense(matrix, query, 5, block=10)
    expected = np.argsort(-(matrix @ query))[:5]
    assert ids.tolist() == expected.tolist()
    assert np.allclose(scores, (matrix @ query)[expected], atol=1e-5)


def test_top_k_dense_k_maior_que_n_e_k_zero():
    matrix = np.eye(3, dtype=np.float32)
    ids, _ = top_k_dense(matrix, np.array([1, 0, 0], dtype=np.float32), 10)
    assert sorted(ids.tolist()) == [0, 1, 2] and ids[0] == 0
    assert top_k_dense(matrix, np.ones(3, dtype=np.float32), 0)[0].size == 0


def test_average_vectors_media_normaliza_e_ignora_oov():
    key_to_index = {"a": 0, "b": 1}
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    out = average_vectors([["a", "b"], ["a", "zzz"], ["zzz"]], key_to_index, matrix, 2)
    assert np.allclose(out[0], [2**-0.5, 2**-0.5])
    assert np.allclose(out[1], [1.0, 0.0])
    assert not out[2].any()  # sem palavras conhecidas -> vetor nulo


def test_top_k_sparse_vazio_e_ordenado():
    import scipy.sparse as sp

    row = sp.csr_matrix(np.array([[0.0, 0.2, 0.0, 0.9, 0.5]], dtype=np.float32))
    ids, vals = _top_k_sparse(row, 2)
    assert ids.tolist() == [3, 4] and vals.tolist() == pytest.approx([0.9, 0.5])
    assert _top_k_sparse(sp.csr_matrix((1, 5), dtype=np.float32), 3)[0].size == 0


# ------------------------------------------------------------------ build + busca (TF-IDF e embeddings)
def test_build_gera_metadados_e_citacao_por_documento(built_index):
    meta = load_meta(built_index)
    assert meta["ingest"]["docs"] == 12
    assert meta["available_methods"] == ["cosine", "sentence_embeddings"]
    assert meta["methods"]["cosine"]["build_seconds"] >= 0 and meta["methods"]["cosine"]["bytes"] > 0
    searcher = IndexSearcher(built_index)
    hit = searcher.search("Como o TF-IDF calcula o peso de um termo?", "cosine", 3)[0]
    assert hit.chunk.title == "Busca lexical e TF-IDF" and hit.rank == 1 and hit.score > 0
    searcher.close()


@pytest.mark.parametrize("method,min_hit,min_mrr", [("cosine", 0.9, 0.8), ("sentence_embeddings", 0.8, 0.6)])
def test_qualidade_minima_no_corpus_de_demonstracao(built_index, demo_questions, method, min_hit, min_mrr):
    searcher = IndexSearcher(built_index)
    hits_, mrrs = [], []
    for item in demo_questions:
        found = searcher.search(item["question"], method, 5)
        m = retrieval_metrics([h.chunk.title for h in found], item["relevant_titles"])
        hits_.append(m["hit"])
        mrrs.append(m["mrr"])
    searcher.close()
    assert np.mean(hits_) >= min_hit and np.mean(mrrs) >= min_mrr


def test_metodo_nao_indexado_e_invalido(built_index):
    searcher = IndexSearcher(built_index)
    with pytest.raises(RuntimeError, match="hnsw"):
        searcher.search("x", "hnsw")
    with pytest.raises(ValueError):
        searcher.search("x", "inexistente")
    searcher.close()


def test_indice_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError, match="build-index"):
        IndexSearcher(tmp_path / "nada")


def test_dependencias_ausentes_sao_registradas_sem_quebrar_o_build(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "gensim", None)
    monkeypatch.setitem(sys.modules, "gensim.models", None)
    monkeypatch.setitem(sys.modules, "hnswlib", None)
    meta = build_index(DEMO_CORPUS, tmp_path / "i", BuildConfig(embedding_model="hashing"), log=quiet)
    assert meta["available_methods"] == ["cosine", "sentence_embeddings"]
    assert set(meta["skipped"]) == {"word2vec_average", "hnsw"}
    assert "gensim" in meta["skipped"]["word2vec_average"]


def test_max_chunks_limita_a_ingestao(tmp_path):
    corpus = write_corpus(tmp_path / "c.jsonl", 30)
    cfg = BuildConfig(methods=("cosine",), max_chunks=10, chunk_size=300, overlap=50)
    meta = build_index(corpus, tmp_path / "i", cfg, log=quiet)
    assert 10 <= meta["ingest"]["chunks"] < 30 * 3  # parou cedo, no limite de um documento


def test_segunda_execucao_reaproveita_o_que_ja_foi_construido(tmp_path):
    cfg = BuildConfig(methods=("cosine",), embedding_model="hashing")
    messages: list[str] = []
    build_index(DEMO_CORPUS, tmp_path / "i", cfg, log=quiet)
    build_index(DEMO_CORPUS, tmp_path / "i", cfg, log=messages.append)
    assert any("reutilizando" in m for m in messages) and any("pulando" in m for m in messages)


class FlakyEncoder(HashingEncoder):
    """Falha na N-ésima chamada, simulando queda no meio de horas de indexação."""

    def __init__(self, fail_on_call=None):
        super().__init__(64)
        self.calls, self.fail_on_call = 0, fail_on_call

    def encode(self, texts):
        self.calls += 1
        if self.fail_on_call and self.calls == self.fail_on_call:
            raise KeyboardInterrupt
        return super().encode(texts)


def test_embeddings_retomam_de_onde_pararam_e_ficam_iguais(tmp_path, monkeypatch):
    corpus = write_corpus(tmp_path / "c.jsonl", 40)
    cfg = BuildConfig(methods=("sentence_embeddings",), embedding_model="hashing", embed_batch=1,
                      chunk_size=200, overlap=20)

    monkeypatch.setattr(index_mod, "get_encoder", lambda *a, **k: FlakyEncoder())
    build_index(corpus, tmp_path / "clean", cfg, log=quiet)
    clean = np.load(tmp_path / "clean" / "st_vectors.npy")
    assert clean.shape[0] > 32  # mais de dois passos (embed_batch*16 = 16)

    monkeypatch.setattr(index_mod, "get_encoder", lambda *a, **k: FlakyEncoder(fail_on_call=3))
    with pytest.raises(KeyboardInterrupt):
        build_index(corpus, tmp_path / "resumed", cfg, log=quiet)
    progress = json.loads((tmp_path / "resumed" / "st_progress.json").read_text())
    assert progress["done"] == 32  # 2 passos de 16 concluídos antes da queda

    messages: list[str] = []
    monkeypatch.setattr(index_mod, "get_encoder", lambda *a, **k: FlakyEncoder())
    build_index(corpus, tmp_path / "resumed", cfg, log=messages.append)
    assert any("retomando de 32" in m for m in messages)
    assert np.array_equal(np.load(tmp_path / "resumed" / "st_vectors.npy"), clean)
    assert not (tmp_path / "resumed" / "st_progress.json").exists()


# ------------------------------------------------------------------ métodos que dependem de bibliotecas externas
def test_word2vec_average_com_gensim(tmp_path):
    pytest.importorskip("gensim")
    cfg = BuildConfig(methods=("word2vec_average",), w2v_epochs=20, w2v_dim=32, w2v_workers=1)
    meta = build_index(DEMO_CORPUS, tmp_path / "i", cfg, log=quiet)
    assert "word2vec_average" in meta["available_methods"]
    searcher = IndexSearcher(tmp_path / "i")
    hits = searcher.search("Quais são as duas arquiteturas do Word2Vec?", "word2vec_average", 5)
    assert len(hits) == 5
    vectors = np.load(tmp_path / "i" / "w2v_vectors.npy")
    assert vectors.shape == (meta["ingest"]["chunks"], 32)
    searcher.close()


def test_hnsw_com_hnswlib_tem_recall_alto_contra_a_busca_exata(tmp_path, demo_questions):
    pytest.importorskip("hnswlib")
    cfg = BuildConfig(methods=("hnsw",), embedding_model="hashing")
    meta = build_index(DEMO_CORPUS, tmp_path / "i", cfg, log=quiet)
    assert {"sentence_embeddings", "hnsw"} <= set(meta["available_methods"])
    searcher = IndexSearcher(tmp_path / "i")
    overlaps = []
    for item in demo_questions:
        exact = {h.chunk.chunk_id for h in searcher.search(item["question"], "sentence_embeddings", 5)}
        approx = {h.chunk.chunk_id for h in searcher.search(item["question"], "hnsw", 5)}
        overlaps.append(len(exact & approx) / 5)
    searcher.close()
    assert np.mean(overlaps) >= 0.9

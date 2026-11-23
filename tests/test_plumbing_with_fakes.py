"""Exercita a lógica de integração com simuladores das bibliotecas pesadas (ver tests/fakes.py)."""

from __future__ import annotations

import numpy as np
import pytest

from paa_qa.encoders import SentenceTransformerEncoder, get_encoder
from paa_qa.index import BuildConfig, IndexSearcher, build_index
from paa_qa.llm import NO_ANSWER, LocalLLM

from . import fakes
from .conftest import DEMO_CORPUS
from .test_qa_eval_cli import make_hit

quiet = lambda _m: None  # noqa: E731


# ------------------------------------------------------------------ Word2Vec
def test_word2vec_pipeline_com_gensim_simulado(tmp_path, monkeypatch):
    fakes.install_gensim(monkeypatch)
    cfg = BuildConfig(methods=("word2vec_average",), w2v_dim=16, w2v_epochs=3, w2v_min_count=1)
    meta = build_index(DEMO_CORPUS, tmp_path / "i", cfg, log=quiet)
    assert meta["available_methods"] == ["word2vec_average"]
    assert meta["methods"]["word2vec_average"]["dim"] == 16 and meta["methods"]["word2vec_average"]["vocab"] > 50
    assert fakes.FakeWord2Vec.train_passes >= 3 * meta["ingest"]["chunks"]  # o corpus foi reiterado por época
    vectors = np.load(tmp_path / "i" / "w2v_vectors.npy")
    assert vectors.shape == (meta["ingest"]["chunks"], 16)
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms[norms > 0], 1.0, atol=1e-5)

    searcher = IndexSearcher(tmp_path / "i")
    hits = searcher.search("busca semântica com vetores de palavras", "word2vec_average", 4)
    assert len(hits) == 4 and hits[0].score >= hits[-1].score
    assert searcher.search("zzzz qqqq", "word2vec_average", 4) == []  # nenhuma palavra conhecida
    searcher.close()


# ------------------------------------------------------------------ HNSW
def test_hnsw_pipeline_com_hnswlib_simulado(tmp_path, monkeypatch, demo_questions):
    fakes.install_hnswlib(monkeypatch)
    cfg = BuildConfig(methods=("hnsw",), embedding_model="hashing")
    meta = build_index(DEMO_CORPUS, tmp_path / "i", cfg, log=quiet)
    assert {"sentence_embeddings", "hnsw"} <= set(meta["available_methods"])  # hnsw puxa os embeddings
    searcher = IndexSearcher(tmp_path / "i")
    for item in demo_questions:
        exact = searcher.search(item["question"], "sentence_embeddings", 5)
        approx = searcher.search(item["question"], "hnsw", 5)
        assert [h.chunk.chunk_id for h in exact] == [h.chunk.chunk_id for h in approx]
        assert approx[0].score == pytest.approx(exact[0].score, abs=1e-2)  # 1 - distância de cosseno
    # k maior que o nº de chunks não pode quebrar (o hnswlib real lança RuntimeError)
    assert len(searcher.search("qualquer coisa", "hnsw", 1000)) == meta["ingest"]["chunks"]
    searcher.close()


def test_hnsw_sem_embeddings_e_ignorado(tmp_path, monkeypatch):
    fakes.install_hnswlib(monkeypatch)
    import sys
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    meta = build_index(DEMO_CORPUS, tmp_path / "i", BuildConfig(methods=("hnsw",)), log=quiet)
    assert meta["available_methods"] == []
    assert set(meta["skipped"]) == {"sentence_embeddings", "hnsw"}


# ------------------------------------------------------------------ sentence-transformers
def test_encoder_sentence_transformers_simulado(monkeypatch):
    fakes.install_sentence_transformers(monkeypatch)
    encoder = get_encoder("algum/modelo", device="cpu", batch_size=8, max_seq_length=256)
    assert isinstance(encoder, SentenceTransformerEncoder) and encoder.dim == 8
    assert fakes.FakeSentenceTransformer.instances[-1].max_seq_length == 256  # evita truncar em 128 tokens
    out = encoder.encode(["a", "b"])
    assert out.dtype == np.float32 and out.shape == (2, 8) and np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_build_completo_com_embedding_simulado(tmp_path, monkeypatch):
    fakes.install_sentence_transformers(monkeypatch)
    fakes.install_hnswlib(monkeypatch)
    cfg = BuildConfig(methods=("sentence_embeddings", "hnsw"), embedding_model="algum/modelo")
    meta = build_index(DEMO_CORPUS, tmp_path / "i", cfg, log=quiet)
    assert meta["methods"]["sentence_embeddings"]["model"] == "algum/modelo"
    searcher = IndexSearcher(tmp_path / "i")  # recarrega o encoder a partir do meta.json
    assert len(searcher.search("pergunta", "hnsw", 3)) == 3
    searcher.close()


# ------------------------------------------------------------------ LLM local
def test_llm_local_gera_so_os_tokens_novos(monkeypatch):
    fakes.install_transformers(monkeypatch)
    llm = LocalLLM("modelo/local", max_new_tokens=16)
    answer = llm.generate("Pergunta?", [make_hit(1, "texto um", "Alfa"), make_hit(2, "texto dois", "Beta")])
    assert answer == "tok1000 tok1001 tok1002"  # o prompt não vaza na resposta
    prompt = fakes.FakeTokenizer.last_prompt
    assert "<system>" in prompt and "[1] Documento: Alfa 1" in prompt and "Pergunta?" in prompt
    assert llm.device == "cpu" and fakes.FakeCausalLM.loaded_with["dtype"] == "float32"


def test_llm_local_cai_para_torch_dtype_em_transformers_antigo(monkeypatch):
    fakes.install_transformers(monkeypatch, reject_dtype_kwarg=True)
    LocalLLM("m").generate("P?", [make_hit(1, "t")])
    assert "torch_dtype" in fakes.FakeCausalLM.loaded_with


def test_llm_local_funde_system_quando_o_template_recusa(monkeypatch):
    fakes.install_transformers(monkeypatch, reject_system=True)
    LocalLLM("m").generate("P?", [make_hit(1, "t")])
    assert "<system>" not in fakes.FakeTokenizer.last_prompt and "<user>" in fakes.FakeTokenizer.last_prompt


def test_llm_local_sem_hits_nao_carrega_o_modelo(monkeypatch):
    fakes.install_transformers(monkeypatch)
    llm = LocalLLM("m")
    assert llm.generate("P?", []) == NO_ANSWER and llm._model is None

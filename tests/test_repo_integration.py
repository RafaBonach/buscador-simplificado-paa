"""Integração com o repositório: saída do WikiExtractor, HNSW via faiss, modelos do app, LLMs pequenos."""

from __future__ import annotations

import json
import sys
import threading

import numpy as np
import pytest

from paa_qa.corpus import iter_documents
from paa_qa.encoders import MissingDependency
from paa_qa.index import BuildConfig, IndexSearcher, _pick_hnsw_backend, build_index, load_meta
from paa_qa.llm import LocalLLM, resolve_model, strip_reasoning
from paa_qa.store import ChunkStore

from . import fakes
from .conftest import DEMO_CORPUS
from .test_qa_eval_cli import make_hit

quiet = lambda _m: None  # noqa: E731
LONGO = "Esta é uma frase de exemplo com conteúdo suficiente para passar do filtro de páginas curtas. " * 4


def write_wikiextractor(root, articles, shard="AA", filename="wiki_00"):
    """Reproduz a saída de `wikiextractor --json`: pasta AA/wiki_00 com um JSON por linha."""
    folder = root / shard
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / filename).open("w", encoding="utf-8") as handle:
        for i, (title, text) in enumerate(articles, start=1):
            handle.write(json.dumps({"id": str(i), "revid": "99", "url": f"https://pt.wikipedia.org/wiki?curid={i}",
                                     "title": title, "text": f"{title}\n\n{text}"}, ensure_ascii=False) + "\n")
    return root


# ------------------------------------------------------------------ saída do WikiExtractor
def test_le_arquivos_wiki_sem_extensao_com_titulo_e_url(tmp_path):
    root = write_wikiextractor(tmp_path / "pt", [("Santos Dumont", LONGO), ("Dijkstra", LONGO)])
    docs = list(iter_documents(root))
    assert [d.title for d in docs] == ["Santos Dumont", "Dijkstra"]
    assert docs[0].url == "https://pt.wikipedia.org/wiki?curid=1" and docs[0].doc_id == "1"
    assert docs[0].text.startswith("Santos Dumont")


def test_filtra_paginas_curtas_e_ignora_linha_corrompida(tmp_path):
    root = write_wikiextractor(tmp_path / "pt", [("Esboço", "curto"), ("Artigo", LONGO)])
    with (root / "AA" / "wiki_00").open("a", encoding="utf-8") as handle:
        handle.write("{isto não é json\n")
    assert [d.title for d in iter_documents(root, min_chars=200)] == ["Artigo"]
    assert len(list(iter_documents(root))) == 2  # sem filtro mantém o esboço; a linha corrompida some


def test_varias_pastas_na_ordem_dada_e_limite_global(tmp_path):
    pt = write_wikiextractor(tmp_path / "pt", [("A", LONGO), ("B", LONGO)])
    en = write_wikiextractor(tmp_path / "en", [("C", LONGO), ("D", LONGO)], shard="AB", filename="wiki_07")
    assert [d.title for d in iter_documents([pt, en])] == ["A", "B", "C", "D"]
    assert [d.title for d in iter_documents([en, pt], limit_docs=3)] == ["C", "D", "A"]


def test_build_index_a_partir_de_duas_pastas_do_wikiextractor(tmp_path):
    pt = write_wikiextractor(tmp_path / "pt", [
        ("Santos Dumont", "Alberto Santos Dumont foi um inventor e aeronauta brasileiro, pioneiro da aviação. " * 5),
        ("Algoritmo de Dijkstra", "O algoritmo de Dijkstra calcula o caminho mínimo entre vértices de um grafo. " * 5),
    ])
    en = write_wikiextractor(tmp_path / "en", [
        ("Photosynthesis", "Photosynthesis is the process by which plants convert light into chemical energy. " * 5)])
    cfg = BuildConfig(methods=("cosine", "sentence_embeddings"), embedding_model="hashing")
    meta = build_index([pt, en], tmp_path / "idx", cfg, log=quiet)
    assert meta["ingest"]["docs"] == 3
    searcher = IndexSearcher(tmp_path / "idx")
    hit = searcher.search("caminho mínimo em um grafo", "cosine", 3)[0]
    assert hit.chunk.title == "Algoritmo de Dijkstra" and hit.chunk.url.endswith("curid=2")
    searcher.close()


def test_pasta_sem_arquivos_reconhecidos_da_erro_claro(tmp_path):
    (tmp_path / "vazia").mkdir()
    with pytest.raises(FileNotFoundError, match="wiki_"):
        list(iter_documents(tmp_path / "vazia"))


# ------------------------------------------------------------------ HNSW: hnswlib ou faiss
def test_hnsw_com_faiss_quando_nao_ha_hnswlib(tmp_path, monkeypatch, demo_questions):
    fakes.install_faiss(monkeypatch)
    monkeypatch.setitem(sys.modules, "hnswlib", None)
    meta = build_index(DEMO_CORPUS, tmp_path / "i", BuildConfig(methods=("hnsw",), embedding_model="hashing"), log=quiet)
    assert meta["methods"]["hnsw"]["backend"] == "faiss"
    searcher = IndexSearcher(tmp_path / "i")
    for item in demo_questions:
        exact = searcher.search(item["question"], "sentence_embeddings", 5)
        approx = searcher.search(item["question"], "hnsw", 5)
        assert [h.chunk.chunk_id for h in exact] == [h.chunk.chunk_id for h in approx]
        assert approx[0].score == pytest.approx(exact[0].score, abs=1e-2)  # produto interno = cosseno
    assert len(searcher.search("qualquer coisa", "hnsw", 1000)) == meta["ingest"]["chunks"]  # k > N não quebra
    searcher.close()


def test_auto_prefere_hnswlib_e_permite_forcar_faiss(monkeypatch):
    fakes.install_faiss(monkeypatch)
    fakes.install_hnswlib(monkeypatch)
    assert _pick_hnsw_backend("auto") == "hnswlib"
    assert _pick_hnsw_backend("faiss") == "faiss"
    with pytest.raises(ValueError, match="hnsw_backend"):
        _pick_hnsw_backend("annoy")


def test_sem_hnswlib_nem_faiss_o_build_continua_e_explica(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "hnswlib", None)
    monkeypatch.setitem(sys.modules, "faiss", None)
    meta = build_index(DEMO_CORPUS, tmp_path / "i", BuildConfig(methods=("hnsw",), embedding_model="hashing"), log=quiet)
    assert meta["available_methods"] == ["sentence_embeddings"] and "faiss-cpu" in meta["skipped"]["hnsw"]
    with pytest.raises(MissingDependency, match="faiss-cpu"):
        _pick_hnsw_backend("auto")


def test_indice_antigo_sem_campo_backend_continua_usando_hnswlib(tmp_path, monkeypatch):
    fakes.install_hnswlib(monkeypatch)
    build_index(DEMO_CORPUS, tmp_path / "i", BuildConfig(methods=("hnsw",), embedding_model="hashing"), log=quiet)
    meta = load_meta(tmp_path / "i")
    del meta["methods"]["hnsw"]["backend"]
    (tmp_path / "i" / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    searcher = IndexSearcher(tmp_path / "i")
    assert len(searcher.search("busca semântica", "hnsw", 3)) == 3
    searcher.close()


def test_hnsw_com_faiss_real(tmp_path, demo_questions):
    pytest.importorskip("faiss")
    cfg = BuildConfig(methods=("hnsw",), embedding_model="hashing", hnsw_backend="faiss")
    meta = build_index(DEMO_CORPUS, tmp_path / "i", cfg, log=quiet)
    assert meta["methods"]["hnsw"]["backend"] == "faiss"
    searcher = IndexSearcher(tmp_path / "i")
    overlaps = []
    for item in demo_questions:
        exact = {h.chunk.chunk_id for h in searcher.search(item["question"], "sentence_embeddings", 5)}
        approx = {h.chunk.chunk_id for h in searcher.search(item["question"], "hnsw", 5)}
        overlaps.append(len(exact & approx) / 5)
    searcher.close()
    assert np.mean(overlaps) >= 0.9


# ------------------------------------------------------------------ nomes de modelos do app
@pytest.mark.parametrize("nome,esperado", [
    ("DeepSeek", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
    ("SmolLM", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
    ("TinyLlama", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    ("Phi-3 Mini", "microsoft/Phi-3-mini-4k-instruct"),
    ("  mistral ", "mistralai/Mistral-7B-Instruct-v0.3"),
    ("minha/pasta-local", "minha/pasta-local"),
])
def test_resolve_model(nome, esperado):
    assert resolve_model(nome) == esperado


def test_strip_reasoning():
    assert strip_reasoning("<think>vou pensar</think>\nResposta final [1]") == "Resposta final [1]"
    assert strip_reasoning("raciocínio sem abertura</think> Resposta [2]") == "Resposta [2]"  # template já abre <think>
    assert strip_reasoning("<think>cortado pelo limite de tokens") == ""
    assert strip_reasoning("  Sem bloco nenhum [1]  ") == "Sem bloco nenhum [1]"


# ------------------------------------------------------------------ LLMs pequenos
def test_deepseek_sem_prompt_de_sistema_e_com_mais_tokens(monkeypatch):
    fakes.install_transformers(monkeypatch)
    llm = LocalLLM("DeepSeek", max_new_tokens=256)
    assert llm.model_id == "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" and llm.max_new_tokens == 768
    llm.generate("Pergunta?", [make_hit(1, "texto", "Alfa")])
    assert [m["role"] for m in fakes.FakeTokenizer.last_messages] == ["user"]
    assert "responder perguntas" in fakes.FakeTokenizer.last_messages[0]["content"].lower() or \
           "trechos" in fakes.FakeTokenizer.last_messages[0]["content"].lower()  # instruções foram para o prompt do usuário


def test_resposta_do_deepseek_perde_o_bloco_de_raciocinio(monkeypatch):
    fakes.install_transformers(monkeypatch)
    monkeypatch.setattr(fakes.FakeTokenizer, "decode",
                        lambda self, ids, skip_special_tokens=True: "<think>hum...</think> A resposta é 42 [1]")
    assert LocalLLM("deepseek").generate("P?", [make_hit(1, "t")]) == "A resposta é 42 [1]"


def test_prompt_maior_que_a_janela_do_modelo_descarta_os_trechos_menos_relevantes(monkeypatch):
    fakes.install_transformers(monkeypatch)
    # janela de 320 - 16 de resposta = 304 "tokens" (o fake conta palavras): 2 trechos ocupam 297, 3 ocupam 415
    monkeypatch.setattr(fakes.FakeTokenizer, "model_max_length", 320, raising=False)
    hits = [make_hit(i, "palavra " * 120, f"Doc{i}") for i in range(1, 6)]
    LocalLLM("tinyllama", max_new_tokens=16).generate("Pergunta?", hits)
    prompt = fakes.FakeTokenizer.last_prompt
    assert "[1] Documento: Doc1" in prompt and "[2] Documento: Doc2" in prompt
    assert "Doc3" not in prompt and "Doc5" not in prompt  # os menos relevantes saíram para caber


def test_um_unico_trecho_gigante_ainda_gera_resposta(monkeypatch):
    fakes.install_transformers(monkeypatch)
    monkeypatch.setattr(fakes.FakeTokenizer, "model_max_length", 100, raising=False)
    answer = LocalLLM("tinyllama", max_new_tokens=16).generate("P?", [make_hit(1, "palavra " * 500)])
    assert answer == "tok1000 tok1001 tok1002"


def test_tokenizer_com_limite_infinito_usa_janela_padrao(monkeypatch):
    fakes.install_transformers(monkeypatch)
    monkeypatch.setattr(fakes.FakeTokenizer, "model_max_length", int(1e30), raising=False)  # como no Mistral
    llm = LocalLLM("mistral", max_new_tokens=256)
    llm.load()
    assert llm._prompt_budget() == 4096 - 256


# ------------------------------------------------------------------ SQLite entre threads
def test_store_somente_leitura_funciona_em_outra_thread(tmp_path):
    store = ChunkStore(tmp_path / "c.db")
    store.add_document("A", "", "s", "1", ["texto"])
    store.commit()
    store.close()
    ro = ChunkStore(tmp_path / "c.db", readonly=True)
    out: list = []
    errors: list = []

    def worker():
        try:
            out.append(ro.get_chunks([0])[0].text)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    ro.close()
    assert out == ["texto"] and not errors


# ------------------------------------------------------------------ tempo de resposta sem o carregamento
def test_carregar_o_llm_nao_entra_no_tempo_de_geracao(built_index):
    import time

    from paa_qa.qa import QASystem

    class LLMLento:
        name = "lento"

        def load(self):
            time.sleep(0.3)  # simula download/carregamento do modelo

        def generate(self, question, hits):
            return "resposta [1]"

    qa = QASystem(IndexSearcher(built_index), LLMLento())
    answer = qa.ask("índice invertido", method="cosine")
    assert answer.timings_ms["generation"] < 150 and answer.timings_ms["total"] < 150
    qa.searcher.close()


def test_encoder_usa_o_nome_novo_da_dimensao_sem_aviso(monkeypatch):
    import warnings

    from paa_qa.encoders import SentenceTransformerEncoder

    class ModeloNovo(fakes.FakeSentenceTransformer):
        def get_sentence_embedding_dimension(self):  # o nome antigo emite aviso nas versões recentes
            warnings.warn("renamed", FutureWarning)
            return 8

        def get_embedding_dimension(self):
            return 8

    fakes.install_sentence_transformers(monkeypatch)
    sys.modules["sentence_transformers"].SentenceTransformer = ModeloNovo
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert SentenceTransformerEncoder("qualquer").dim == 8


def test_encoder_antigo_sem_o_nome_novo_continua_funcionando(monkeypatch):
    from paa_qa.encoders import SentenceTransformerEncoder

    fakes.install_sentence_transformers(monkeypatch)  # o simulador só tem o nome antigo
    assert SentenceTransformerEncoder("qualquer").dim == 8


# ------------------------------------------------------------------ memória do LLM
TINY = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # ~2,2 GB em 16 bits, ~4,4 GB em float32


def test_ram_sobrando_usa_float32_que_e_mais_rapido_em_cpu():
    from paa_qa.llm import choose_dtype

    assert choose_dtype(TINY, "cpu", 16.0) == "float32"


def test_ram_apertada_usa_bfloat16_que_ocupa_metade():
    from paa_qa.llm import choose_dtype

    assert choose_dtype(TINY, "cpu", 4.0) == "bfloat16"  # float32 pediria ~5,5 GB; 16 bits pede ~2,75 GB


def test_ram_insuficiente_avisa_em_vez_de_deixar_o_processo_morrer():
    from paa_qa.llm import InsufficientMemory, choose_dtype

    with pytest.raises(InsufficientMemory, match="Memória insuficiente.*TinyLlama.*1.5 GB"):
        choose_dtype(TINY, "cpu", 1.5)


def test_gpu_usa_float16_e_ram_desconhecida_ou_modelo_desconhecido_mantem_o_padrao_antigo():
    from paa_qa.llm import choose_dtype

    assert choose_dtype(TINY, "cuda", 1.0) == "float16"
    assert choose_dtype(TINY, "cpu", None) == "float32"
    assert choose_dtype("minha/pasta-local", "cpu", 0.5) == "float32"


def test_dtype_pode_ser_forcado_e_valor_invalido_e_recusado():
    from paa_qa.llm import choose_dtype

    assert choose_dtype(TINY, "cpu", 1.0, forced="bfloat16") == "bfloat16"
    with pytest.raises(ValueError, match="PAA_LLM_DTYPE"):
        choose_dtype(TINY, "cpu", 64.0, forced="int4")


def test_load_passa_a_precisao_escolhida_ao_transformers(monkeypatch):
    from paa_qa import llm as llm_module

    fakes.install_transformers(monkeypatch)
    monkeypatch.delenv("PAA_LLM_DTYPE", raising=False)
    monkeypatch.setattr(llm_module, "available_ram_gb", lambda: 4.0)
    llm = LocalLLM("tinyllama")
    llm.load()
    assert fakes.FakeCausalLM.loaded_with["dtype"] == "bfloat16" and llm.dtype_name == "bfloat16"


def test_load_com_pouca_ram_levanta_erro_claro_sem_carregar_o_modelo(monkeypatch):
    from paa_qa import llm as llm_module

    fakes.install_transformers(monkeypatch)
    monkeypatch.setattr(llm_module, "available_ram_gb", lambda: 1.0)
    fakes.FakeCausalLM.loaded_with = {"marca": "nao carregou"}
    llm = LocalLLM("tinyllama")
    with pytest.raises(llm_module.InsufficientMemory):
        llm.load()
    assert fakes.FakeCausalLM.loaded_with == {"marca": "nao carregou"} and llm._model is None


def test_available_ram_gb_devolve_numero_positivo_ou_none():
    from paa_qa.llm import available_ram_gb

    value = available_ram_gb()
    assert value is None or value > 0

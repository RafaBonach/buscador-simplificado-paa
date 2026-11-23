from __future__ import annotations

import runpy
import sys
import time
from pathlib import Path

import pytest

from paa_qa import app_backend as backend
from paa_qa.qa import Answer, Citation

from . import fakes
from .fake_streamlit import FakeStreamlit, Stop

ROOT = Path(__file__).resolve().parent.parent
METODO = "Método de Similaridade:"
LLM = "Modelo LLM Local:"
PASTA = "Pasta do índice:"


@pytest.fixture(autouse=True)
def _limpa_cache():
    yield
    backend.reset_cache()


@pytest.fixture
def st(monkeypatch):
    fake = FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    return fake


def run_app(fake: FakeStreamlit):
    try:
        runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
    except Stop:
        pass


# ------------------------------------------------------------------ app.py com o backend real
def test_sem_indice_mostra_como_construir_e_para(st, tmp_path):
    st.inputs[PASTA] = str(tmp_path / "nada")
    run_app(st)
    assert "Índice não encontrado." in st.texts("error")
    assert any("Não há índice" in t for t in st.texts("warning"))
    assert any("build-index" in t for t in st.texts("code"))
    assert st.session_state.messages == []


def test_pergunta_gera_resposta_com_documento_de_origem(st, built_index):
    st.inputs[PASTA] = str(built_index)
    st.choices[METODO] = "Cosseno (Cosine Similarity)"
    st.prompt = "O que é um índice invertido?"
    run_app(st)

    assert st.options[METODO] == ["Cosseno (Cosine Similarity)", "Sentence Embeddings"]  # sem HNSW/Word2Vec: não foram indexados
    assert st.options[LLM][0] == backend.LLM_EXTRACTIVE
    user, assistant = st.session_state.messages
    assert user == {"role": "user", "content": "O que é um índice invertido?"}
    assert "lista dos documentos" in assistant["content"]
    assert "Busca lexical e TF-IDF" in assistant["citations"] and "similaridade" in assistant["citations"]
    assert "busca (cosine)" in assistant["timings"]
    assert any("Busca lexical e TF-IDF" in t for t in st.texts("markdown"))
    assert any(t.startswith("Índice: 12 documentos") for t in st.texts("caption"))  # resumo do índice na lateral
    assert st.session_state.historico_perguntas[0].endswith("O que é um índice invertido?")


def test_segunda_pergunta_reexibe_o_historico_do_chat(st, built_index):
    st.inputs[PASTA] = str(built_index)
    st.choices[METODO] = "Sentence Embeddings"
    st.prompt = "Qual a fórmula da similaridade de cosseno?"
    run_app(st)
    st.log.clear()
    st.prompt = "Quais são as duas arquiteturas do Word2Vec?"
    run_app(st)
    assert len(st.session_state.messages) == 4
    # a resposta antiga foi redesenhada antes da nova pergunta
    markdowns = st.texts("markdown")
    assert any("Similaridade de cosseno" in t for t in markdowns)
    assert len(st.session_state.historico_perguntas) == 2


def test_metodo_e_carregado_uma_vez_antes_da_primeira_pergunta(st, built_index, monkeypatch):
    from paa_qa.index import IndexSearcher

    chamadas = []
    original = IndexSearcher.warmup
    monkeypatch.setattr(IndexSearcher, "warmup", lambda self, metodo: (chamadas.append(metodo), original(self, metodo))[1])
    st.inputs[PASTA] = str(built_index)
    st.choices[METODO] = "Cosseno (Cosine Similarity)"
    st.prompt = "O que é um índice invertido?"
    run_app(st)
    assert chamadas == ["cosine"]
    st.prompt = "Qual a fórmula da similaridade de cosseno?"
    run_app(st)
    assert chamadas == ["cosine"]  # a 2ª execução do script não recarrega
    st.choices[METODO] = "Sentence Embeddings"
    run_app(st)
    assert chamadas == ["cosine", "sentence_embeddings"]  # método novo: aquece de novo


def test_falha_ao_preparar_o_metodo_vira_aviso_e_o_erro_aparece_ao_perguntar(st, built_index, monkeypatch):
    from paa_qa.index import IndexSearcher

    def quebra(self, metodo):
        raise RuntimeError("falha de teste")

    monkeypatch.setattr(IndexSearcher, "warmup", quebra)
    st.inputs[PASTA] = str(built_index)
    st.choices[METODO] = "Cosseno (Cosine Similarity)"
    run_app(st)
    assert any("Não consegui preparar o método de busca: RuntimeError: falha de teste" in t for t in st.texts("warning"))


def test_erro_do_llm_vira_mensagem_no_chat(st, built_index, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # simula torch não instalado
    st.inputs[PASTA] = str(built_index)
    st.choices[METODO] = "Cosseno (Cosine Similarity)"
    st.choices[LLM] = "SmolLM"
    st.prompt = "O que é um índice invertido?"
    run_app(st)
    erro = st.texts("error")[-1]
    assert erro.startswith("Não consegui responder: MissingDependency")
    assert st.session_state.messages[-1]["content"] == erro


def test_app_com_llm_simulado(st, built_index, monkeypatch):
    fakes.install_transformers(monkeypatch)
    st.inputs[PASTA] = str(built_index)
    st.choices[METODO] = "Cosseno (Cosine Similarity)"
    st.choices[LLM] = "TinyLlama"
    st.prompt = "O que é um índice invertido?"
    run_app(st)
    assistant = st.session_state.messages[-1]
    assert assistant["content"] == "tok1000 tok1001 tok1002"
    assert "geração (TinyLlama)" in assistant["timings"]
    assert "Busca lexical e TF-IDF" in assistant["citations"]  # as citações vêm da busca, não do texto gerado


# ------------------------------------------------------------------ app_backend
def test_index_info_e_opcoes_de_metodo(built_index):
    info = backend.index_info(str(built_index))
    assert info["docs"] == 12 and info["chunks"] >= 12 and info["methods"] == ["cosine", "sentence_embeddings"]
    assert backend.method_options(info) == ["Cosseno (Cosine Similarity)", "Sentence Embeddings"]
    with pytest.raises(FileNotFoundError):
        backend.index_info(str(built_index / "nao_existe"))


def test_so_um_llm_fica_em_memoria_e_o_mesmo_e_reaproveitado():
    a = backend._llm("SmolLM", 256)
    assert backend._llm("SmolLM", 256) is a
    assert a.model_id == "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    b = backend._llm("TinyLlama", 256)
    assert b is not a and backend._llm_cache["obj"] is b
    assert backend._llm(backend.LLM_EXTRACTIVE, 256) is None


def test_busca_roda_em_outra_thread(built_index):
    """O Streamlit atende cada interação numa thread diferente; o SQLite não pode reclamar."""
    import threading

    backend.ask(str(built_index), "Cosseno (Cosine Similarity)", backend.LLM_EXTRACTIVE, "índice invertido")
    result: list = []
    thread = threading.Thread(target=lambda: result.append(
        backend.ask(str(built_index), "Cosseno (Cosine Similarity)", backend.LLM_EXTRACTIVE, "cosseno")))
    thread.start()
    thread.join()
    assert result and result[0].citations


def make_answer(**overrides) -> Answer:
    citation = Citation(n=1, title="Preço em $ [teste]", url="https://x/y", doc_id=0, chunk_id=0, position=0,
                        score=0.5, snippet="custa $5 e $10")
    base = dict(question="q", method="cosine", answer="r", generator="extractive", citations=[citation], contexts=["c"],
                timings_ms={"retrieval": 1.0, "generation": 2.0, "total": 3.0})
    base.update(overrides)
    return Answer(**base)


def test_citacoes_escapam_cifrao_e_colchetes_e_linkam_o_documento():
    text = backend.render_citations(make_answer())
    assert "[Preço em \\$ \\[teste\\]](https://x/y)" in text and "custa \\$5 e \\$10" in text
    assert "similaridade 0.50" in text


def test_citacao_sem_url_e_sem_resultados():
    sem_url = make_answer()
    sem_url.citations[0].url = ""
    assert "**[1] Preço" in backend.render_citations(sem_url)
    assert "Nenhum documento" in backend.render_citations(make_answer(citations=[]))
    assert "total: 3 ms" in backend.render_timings(make_answer())

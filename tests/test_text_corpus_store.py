from __future__ import annotations

import json

import pytest

from paa_qa import corpus
from paa_qa.corpus import chunk_text, iter_documents
from paa_qa.store import ChunkStore
from paa_qa.text import content_tokens, split_sentences, word_tokens


# ------------------------------------------------------------------ texto
def test_content_tokens_remove_stopwords_com_e_sem_acento():
    tokens = content_tokens("Eles são rápidos e até mais fortes")
    assert "sao" not in tokens and "ate" not in tokens
    assert "rapidos" in tokens and "fortes" in tokens


def test_content_tokens_remove_stopwords_digitadas_sem_acento():
    # usuários frequentemente digitam "sao", "ate", "voce" sem acento
    tokens = content_tokens("eles sao rapidos ate la")
    assert "sao" not in tokens and "ate" not in tokens and "rapidos" in tokens


def test_word_tokens_mantem_acentos_e_remove_stopwords():
    assert word_tokens("A busca é rápida") == ["busca", "rápida"]


def test_split_sentences():
    parts = split_sentences("Primeira frase. Segunda frase! Terceira?\nOutra linha.")
    assert len(parts) == 4


# ------------------------------------------------------------------ chunking
def _long_text(n=40):
    return " ".join(f"Esta é a sentença número {i} do texto de teste." for i in range(n))


def test_chunks_respeitam_limite_e_cobrem_o_texto():
    text = _long_text()
    chunks = chunk_text(text, 200, 50)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    joined = " ".join(chunks)
    assert all(f"número {i} " in joined for i in range(40))


def test_overlap_nao_estoura_o_limite_quando_a_proxima_sentenca_e_grande():
    s1 = "A" + "a" * 48 + "."   # 50 caracteres
    s2 = "B" + "b" * 43 + "."   # 45
    s3 = "C" + "c" * 88 + "."   # 90: com o overlap de s2 (46) passaria de 100
    chunks = chunk_text(" ".join([s1, s2, s3]), 100, 60)
    assert all(len(c) <= 100 for c in chunks), [len(c) for c in chunks]
    assert chunks[-1] == s3  # o overlap foi descartado em vez de estourar


def test_overlap_repete_final_do_chunk_anterior():
    chunks = chunk_text(_long_text(), 200, 80)
    first_sentences = split_sentences(chunks[0])
    assert first_sentences[-1] in chunks[1]


def test_sentenca_gigante_e_dividida_em_espacos():
    chunks = chunk_text("palavra " * 500, 100, 10)
    assert all(len(c) <= 100 for c in chunks)
    assert sum(len(c.split()) for c in chunks) >= 500


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_chunk_text_valida_parametros(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("texto", size, overlap)


def test_chunk_text_vazio():
    assert chunk_text("   ", 100, 10) == []


# ------------------------------------------------------------------ store
def test_store_ida_e_volta(tmp_path):
    store = ChunkStore(tmp_path / "c.db")
    assert store.add_document("A", "http://a", "s", "1", ["a0", "a1"]) == 2
    assert store.add_document("B", "", "s", "2", ["b0"]) == 1
    store.commit()
    assert store.count_chunks() == 3 and store.count_docs() == 2
    recs = store.get_chunks([2, 0])
    assert [r.text for r in recs] == ["b0", "a0"]  # mantém a ordem pedida
    assert recs[0].title == "B" and recs[1].url == "http://a"
    batches = list(store.iter_texts(batch=2))
    assert batches == [(0, ["a0", "a1"]), (2, ["b0"])]
    store.close()


def test_store_readonly_com_caminho_especial(tmp_path):
    folder = tmp_path / "minha pasta #1 (cópia)"
    store = ChunkStore(folder / "c.db")
    store.add_document("A", "", "s", "1", ["texto"])
    store.commit()
    store.close()
    ro = ChunkStore(folder / "c.db", readonly=True)
    assert ro.get_chunks([0])[0].text == "texto"
    with pytest.raises(Exception):  # modo somente leitura de verdade
        ro.conn.execute("DELETE FROM chunks")
    ro.close()


def test_store_readonly_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        ChunkStore(tmp_path / "nao_existe.db", readonly=True)


# ------------------------------------------------------------------ leitura de documentos
def test_iter_documents_jsonl_e_txt(tmp_path):
    (tmp_path / "a.jsonl").write_text(
        json.dumps({"id": "1", "title": "T1", "url": "u", "text": "conteúdo um"}) + "\n"
        + json.dumps({"id": "2", "title": "T2", "text": "   "}) + "\n",  # vazio: ignorado
        encoding="utf-8")
    (tmp_path / "livro_um.txt").write_text("Texto do livro.", encoding="utf-8")
    docs = list(iter_documents(tmp_path))
    assert [d.title for d in docs] == ["T1", "livro um"]
    assert docs[0].url == "u"


def test_iter_documents_limit_e_vazio(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(iter_documents(tmp_path))
    (tmp_path / "x.jsonl").write_text("\n".join(json.dumps({"id": str(i), "title": "t", "text": "x"}) for i in range(5)))
    assert len(list(iter_documents(tmp_path, limit_docs=2))) == 2


def test_txt_grande_demais_gera_erro_claro(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus, "MAX_TXT_BYTES", 10)
    (tmp_path / "grande.txt").write_text("x" * 50)
    with pytest.raises(ValueError, match="download"):
        list(iter_documents(tmp_path))

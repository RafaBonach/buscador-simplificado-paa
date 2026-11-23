"""Testa o download com um módulo `datasets` simulado (sem rede)."""

from __future__ import annotations

import sys
import types

import pytest

from paa_qa.corpus import GIB, download_corpus, iter_documents

DOC_BYTES = 1000


def _rows(prefix: str, n: int):
    return [{"id": f"{prefix}{i}", "title": f"{prefix} {i}", "url": f"http://{prefix}/{i}", "text": "x" * DOC_BYTES}
            for i in range(n)]


class FakeStream:
    def __init__(self, rows, skip=0, fail_after=None):
        self.rows, self._skip, self.fail_after = rows, skip, fail_after

    def skip(self, n):
        return FakeStream(self.rows, n, self.fail_after)

    def __iter__(self):
        for i, row in enumerate(self.rows[self._skip:]):
            if self.fail_after is not None and i >= self.fail_after:
                raise RuntimeError("rede caiu")
            yield row


def install_fake(monkeypatch, data: dict, fail_after: dict | None = None):
    module = types.ModuleType("datasets")

    def load_dataset(name, config=None, split="train", streaming=True):
        key = f"{name}:{config}" if config else name
        return FakeStream(data[key], fail_after=(fail_after or {}).get(key))

    module.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", module)


def gb(n_docs: float) -> float:
    return n_docs * DOC_BYTES / GIB


def test_para_ao_atingir_a_meta_e_grava_metadados(tmp_path, monkeypatch):
    install_fake(monkeypatch, {"d:pt": _rows("a", 50)})
    result = download_corpus(["d:pt"], tmp_path, target_gb=gb(10), shard_mb=0.002, log=lambda _m: None)
    assert result["completed"] and result["docs"] == 10
    docs = list(iter_documents(tmp_path))
    assert len(docs) == 10
    assert docs[0].title == "a 0" and docs[0].url == "http://a/0" and docs[0].source == "d:pt"
    assert len(list(tmp_path.glob("part-*.jsonl"))) > 1  # houve troca de shard


def test_varias_fontes_em_ordem_quando_a_primeira_acaba(tmp_path, monkeypatch):
    install_fake(monkeypatch, {"d:pt": _rows("a", 4), "d:en": _rows("b", 20)})
    result = download_corpus(["d:pt", "d:en"], tmp_path, target_gb=gb(10), log=lambda _m: None)
    assert result["completed"]
    ids = [d.doc_id for d in iter_documents(tmp_path)]
    assert ids[:4] == ["a0", "a1", "a2", "a3"] and ids[4] == "b0" and len(ids) == 10


def test_fontes_esgotadas_avisam_que_nao_chegou_na_meta(tmp_path, monkeypatch):
    install_fake(monkeypatch, {"d:pt": _rows("a", 3)})
    result = download_corpus(["d:pt"], tmp_path, target_gb=gb(100), log=lambda _m: None)
    assert not result["completed"] and "warning" in result and result["docs"] == 3


def test_retomada_nao_duplica_documentos(tmp_path, monkeypatch):
    data = {"d:pt": _rows("a", 50)}
    install_fake(monkeypatch, data)
    download_corpus(["d:pt"], tmp_path, target_gb=gb(4), shard_mb=0.002, log=lambda _m: None)
    result = download_corpus(["d:pt"], tmp_path, target_gb=gb(12), shard_mb=0.002, log=lambda _m: None)
    ids = [d.doc_id for d in iter_documents(tmp_path)]
    assert result["completed"] and ids == [f"a{i}" for i in range(12)]


def test_retomada_apos_falha_de_rede_no_meio(tmp_path, monkeypatch):
    data = {"d:pt": _rows("a", 50)}
    install_fake(monkeypatch, data, fail_after={"d:pt": 5})
    with pytest.raises(RuntimeError, match="rede caiu"):
        download_corpus(["d:pt"], tmp_path, target_gb=gb(12), shard_mb=0.002, log=lambda _m: None)
    install_fake(monkeypatch, data)  # rede voltou
    result = download_corpus(["d:pt"], tmp_path, target_gb=gb(12), shard_mb=0.002, log=lambda _m: None)
    ids = [d.doc_id for d in iter_documents(tmp_path)]
    assert result["completed"] and ids == [f"a{i}" for i in range(12)]


def test_sem_pacote_datasets_da_erro_claro(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)
    with pytest.raises(RuntimeError, match="datasets"):
        download_corpus(["d:pt"], tmp_path, target_gb=1)

"""Simuladores mínimos de gensim, hnswlib, sentence-transformers e transformers/torch.

Servem para exercitar a lógica de integração do projeto (fluxo de dados, shapes, formatos) quando as
bibliotecas reais não estão instaladas. Eles refletem a API documentada, mas NÃO provam que as bibliotecas
reais se comportam igual: isso só se confirma rodando o projeto com elas instaladas.
"""

from __future__ import annotations

import pickle
import types
from contextlib import contextmanager

import numpy as np


# ------------------------------------------------------------------ gensim
class _KeyedVectors:
    def __init__(self, key_to_index, vectors):
        self.key_to_index, self.vectors = key_to_index, vectors
        self.vector_size = vectors.shape[1]


class FakeWord2Vec:
    train_passes = 0

    def __init__(self, sentences=None, vector_size=100, window=5, min_count=5, workers=1, epochs=5, seed=1, **_):
        counts: dict[str, int] = {}
        for sentence in sentences:            # 1ª passada: vocabulário (o gensim exige iterável reutilizável)
            for token in sentence:
                counts[token] = counts.get(token, 0) + 1
        for _ in range(epochs):               # passadas de treino
            for _sentence in sentences:
                FakeWord2Vec.train_passes += 1
        vocab = sorted(t for t, c in counts.items() if c >= min_count)
        rng = np.random.default_rng(seed)
        self.wv = _KeyedVectors({t: i for i, t in enumerate(vocab)},
                                rng.normal(size=(len(vocab), vector_size)).astype(np.float32))

    def save(self, path):
        with open(path, "wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as handle:
            return pickle.load(handle)


def install_gensim(monkeypatch):
    import sys
    models = types.ModuleType("gensim.models")
    models.Word2Vec = FakeWord2Vec
    pkg = types.ModuleType("gensim")
    pkg.models = models
    monkeypatch.setitem(sys.modules, "gensim", pkg)
    monkeypatch.setitem(sys.modules, "gensim.models", models)


# ------------------------------------------------------------------ hnswlib
class FakeHnswIndex:
    def __init__(self, space, dim):
        assert space == "cosine"
        self.dim, self.data, self.count, self.ef = dim, None, 0, 10

    def init_index(self, max_elements, ef_construction=200, M=16):
        self.data = np.zeros((max_elements, self.dim), dtype=np.float32)

    def add_items(self, data, ids=None):
        data = np.asarray(data, dtype=np.float32)
        assert data.ndim == 2 and data.shape[1] == self.dim
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.data[np.asarray(ids)] = data / norms
        self.count += len(data)

    def set_ef(self, ef):
        self.ef = ef

    def knn_query(self, q, k=1):
        q = np.asarray(q, dtype=np.float32)
        assert q.ndim == 2
        if k > self.count:  # o hnswlib real também falha nesse caso
            raise RuntimeError("Cannot return the results in a contiguous 2D array. Probably ef or M is too small")
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        sims = q @ self.data[: self.count].T
        labels = np.argsort(-sims, axis=1)[:, :k]
        return labels.astype(np.uint64), (1.0 - np.take_along_axis(sims, labels, axis=1)).astype(np.float32)

    def save_index(self, path):
        with open(path, "wb") as handle:
            np.save(handle, self.data[: self.count])

    def load_index(self, path, max_elements=0):
        with open(path, "rb") as handle:
            self.data = np.load(handle)
        self.count = len(self.data)


def install_hnswlib(monkeypatch):
    import sys
    module = types.ModuleType("hnswlib")
    module.Index = FakeHnswIndex
    monkeypatch.setitem(sys.modules, "hnswlib", module)


# ------------------------------------------------------------------ faiss
class FakeFaissIndex:
    """IndexHNSWFlat com produto interno, como o faiss real (add/search/hnsw.efSearch)."""

    def __init__(self, d, M, metric):
        assert metric == 0  # METRIC_INNER_PRODUCT
        self.d = d
        self.hnsw = types.SimpleNamespace(efConstruction=40, efSearch=16)
        self.vectors = np.zeros((0, d), dtype=np.float32)

    @property
    def ntotal(self):
        return len(self.vectors)

    def add(self, x):
        assert x.dtype == np.float32 and x.ndim == 2 and x.flags["C_CONTIGUOUS"]  # exigências do faiss real
        self.vectors = np.vstack([self.vectors, x])

    def search(self, q, k):
        assert q.dtype == np.float32 and q.ndim == 2
        sims = q @ self.vectors.T
        order = np.argsort(-sims, axis=1)[:, :k]
        scores = np.take_along_axis(sims, order, axis=1)
        if order.shape[1] < k:  # com k > ntotal o faiss completa com -1
            pad = k - order.shape[1]
            order = np.pad(order, ((0, 0), (0, pad)), constant_values=-1)
            scores = np.pad(scores, ((0, 0), (0, pad)), constant_values=-3.4e38)
        return scores.astype(np.float32), order.astype(np.int64)


def install_faiss(monkeypatch):
    import sys
    module = types.ModuleType("faiss")
    module.METRIC_INNER_PRODUCT = 0
    module.IndexHNSWFlat = FakeFaissIndex

    def write_index(index, path):
        with open(path, "wb") as handle:
            pickle.dump(index, handle)

    def read_index(path):
        with open(path, "rb") as handle:
            return pickle.load(handle)

    module.write_index, module.read_index = write_index, read_index
    monkeypatch.setitem(sys.modules, "faiss", module)


# ------------------------------------------------------------------ sentence-transformers
class FakeSentenceTransformer:
    instances: list = []

    def __init__(self, model_name, device=None):
        self.model_name, self.device, self.max_seq_length = model_name, device, 128
        FakeSentenceTransformer.instances.append(self)

    def get_sentence_embedding_dimension(self):
        return 8

    def encode(self, texts, batch_size=32, normalize_embeddings=False, show_progress_bar=False, convert_to_numpy=True):
        assert isinstance(texts, list)
        rng_vectors = np.stack([np.random.default_rng(abs(hash(t)) % (2**32)).normal(size=8) for t in texts])
        if normalize_embeddings:
            rng_vectors = rng_vectors / np.linalg.norm(rng_vectors, axis=1, keepdims=True)
        return rng_vectors.astype(np.float64)


def install_sentence_transformers(monkeypatch):
    import sys
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


# ------------------------------------------------------------------ torch + transformers
class _Batch(dict):
    def to(self, device):
        self.device = device
        return self


class FakeTokenizer:
    chat_template = "fake"
    eos_token = "</s>"
    pad_token = None
    reject_system = False
    last_prompt = ""

    @property
    def pad_token_id(self):  # como nos tokenizers reais: derivado de pad_token
        return None if self.pad_token is None else 2

    @classmethod
    def from_pretrained(cls, model_id):
        return cls()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False and add_generation_prompt is True
        FakeTokenizer.last_messages = messages
        if self.reject_system and any(m["role"] == "system" for m in messages):
            raise ValueError("System role not supported")
        prompt = "\n".join(f"<{m['role']}> {m['content']}" for m in messages) + "\n<assistant>"
        FakeTokenizer.last_prompt = prompt
        return prompt

    def __call__(self, prompt, return_tensors="pt", add_special_tokens=True):
        assert add_special_tokens is False  # o template já inclui os tokens especiais
        ids = np.arange(len(prompt.split())).reshape(1, -1)
        return _Batch(input_ids=ids)

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"tok{int(i)}" for i in ids)


class FakeCausalLM:
    reject_dtype_kwarg = False
    loaded_with: dict = {}

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        if cls.reject_dtype_kwarg and "dtype" in kwargs:
            raise TypeError("__init__() got an unexpected keyword argument 'dtype'")
        FakeCausalLM.loaded_with = kwargs
        return cls()

    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, input_ids, max_new_tokens, do_sample, pad_token_id, **_):
        assert do_sample is False and pad_token_id is not None
        new = np.array([[1000, 1001, 1002]])
        return np.concatenate([input_ids, new], axis=1)


@contextmanager
def _noop():
    yield


def install_transformers(monkeypatch, reject_dtype_kwarg=False, reject_system=False):
    import sys
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.float16, torch.float32, torch.bfloat16 = "float16", "float32", "bfloat16"
    torch.no_grad = _noop
    tf = types.ModuleType("transformers")
    FakeCausalLM.reject_dtype_kwarg = reject_dtype_kwarg
    FakeTokenizer.reject_system = reject_system
    tf.AutoTokenizer, tf.AutoModelForCausalLM = FakeTokenizer, FakeCausalLM
    from paa_qa import llm as llm_module

    monkeypatch.setattr(llm_module, "available_ram_gb", lambda: 64.0)  # os testes não dependem da RAM da máquina
    monkeypatch.setattr(FakeTokenizer, "model_max_length", 2048, raising=False)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", tf)

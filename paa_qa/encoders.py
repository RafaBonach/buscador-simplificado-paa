"""Codificadores de sentenças (embeddings densos)."""

from __future__ import annotations

import zlib

import numpy as np

from .text import content_tokens

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class MissingDependency(RuntimeError):
    """Dependência opcional ausente (o build-index continua com os métodos possíveis)."""


class Encoder:
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class SentenceTransformerEncoder(Encoder):
    """Wrapper do sentence-transformers. Roda 100% local; use `download-models` para baixar uma vez."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, device: str | None = None,
                 batch_size: int = 64, max_seq_length: int = 256):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise MissingDependency(
                "Pacote 'sentence-transformers' ausente. Instale com: pip install -r requirements.txt"
            ) from exc
        self.name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        # O padrão de 128 tokens do modelo multilíngue truncaria chunks de ~800 caracteres.
        if max_seq_length:
            self.model.max_seq_length = max_seq_length
        # O nome do método mudou nas versões recentes (o antigo ainda existe, mas emite aviso).
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = int(get_dim())

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)


class HashingEncoder(Encoder):
    """Codificador determinístico por hashing de tokens (sem rede, sem torch).

    NÃO é um modelo semântico: serve só para testes e para exercitar o pipeline completo sem baixar
    modelos (`--embedding-model hashing`). Para resultados reais use um SentenceTransformer.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.name = f"hashing:{dim}"

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in content_tokens(text):
                h = zlib.crc32(tok.encode("utf-8"))
                out[row, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


def get_encoder(name: str, device: str | None = None, batch_size: int = 64, max_seq_length: int = 256) -> Encoder:
    if name == "hashing" or name.startswith("hashing:"):
        dim = int(name.split(":", 1)[1]) if ":" in name else 256
        return HashingEncoder(dim)
    return SentenceTransformerEncoder(name, device=device, batch_size=batch_size, max_seq_length=max_seq_length)

"""Utilitários de texto: tokenização, normalização e divisão em sentenças."""

from __future__ import annotations

import re
import unicodedata

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+(?=[\"“(\[]?[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ0-9])")

STOPWORDS_PT = frozenset(
    """a o as os um uma uns umas de do da dos das em no na nos nas num numa por para com sem sob
    sobre entre até ao aos à às e ou mas que se como qual quais quem quando onde é são foi foram
    ser era eram há mais menos muito pouco já também não sim seu sua seus suas ele ela eles elas
    isso isto este esta estes estas esse essa esses essas aquele aquela ter tem têm pelo pela
    pelos pelas lhe lhes me te nos vos meu minha teu tua nosso nossa dele dela deles delas
    the of and to in is are what who how""".split()
)


def fold_accents(text: str) -> str:
    """Remove acentos (NFKD) mantendo as letras base."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def tokenize(text: str) -> list[str]:
    """Tokens em minúsculas (letras/dígitos). Mantém acentos."""
    return [t.lower() for t in _WORD_RE.findall(text)]


STOPWORDS_FOLDED = frozenset(fold_accents(w) for w in STOPWORDS_PT)


def content_tokens(text: str) -> list[str]:
    """Tokens sem acento e sem stopwords (comparadas com e sem acento), para métricas e comparação lexical."""
    out = []
    for word in tokenize(text):
        folded = fold_accents(word)
        if len(folded) < 2 or word in STOPWORDS_PT or folded in STOPWORDS_FOLDED:
            continue
        out.append(folded)
    return out


def word_tokens(text: str) -> list[str]:
    """Tokens para Word2Vec: minúsculas, com acento, sem stopwords e sem tokens de 1 caractere."""
    return [t for t in tokenize(text) if len(t) > 1 and t not in STOPWORDS_PT]


def split_sentences(text: str) -> list[str]:
    """Divisão simples de sentenças (suficiente para extração de respostas)."""
    parts: list[str] = []
    for block in text.replace("\r", "").split("\n"):
        block = block.strip()
        if not block:
            continue
        parts.extend(s.strip() for s in _SENT_SPLIT_RE.split(block) if s.strip())
    return parts


def shorten(text: str, limit: int = 220) -> str:
    """Resume em uma linha, cortando em limite de palavras."""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    cut = flat[:limit].rsplit(" ", 1)[0]
    return cut + "…"

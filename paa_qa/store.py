"""Armazenamento de documentos e chunks em SQLite.

O texto nunca fica todo em memória: os chunks são lidos sob demanda por id (para montar a
resposta) ou em lotes sequenciais (para indexar). O id do chunk é sequencial a partir de 0 e
corresponde à linha dos vetores/matrizes de cada índice.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
    doc_id INTEGER PRIMARY KEY,
    ext_id TEXT,
    title  TEXT NOT NULL,
    url    TEXT,
    source TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id INTEGER PRIMARY KEY,
    doc_id   INTEGER NOT NULL,
    pos      INTEGER NOT NULL,
    text     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: int
    doc_id: int
    pos: int
    text: str
    title: str
    url: str
    source: str


class ChunkStore:
    def __init__(self, path: Path, readonly: bool = False):
        self.path = Path(path)
        if readonly:
            if not self.path.exists():
                raise FileNotFoundError(f"Base de chunks não encontrada: {self.path}")
            # as_uri() escapa espaços/#/? e trata letra de unidade no Windows (file:///C:/...)
            # check_same_thread=False: o Streamlit reutiliza o índice (cache) em threads diferentes a cada
            # interação; o acesso é só leitura e serializado pelo lock em get_chunks.
            self.conn = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, check_same_thread=False)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(self.path)
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.executescript(SCHEMA)
        self._lock = threading.Lock()
        self._next_chunk = 0
        self._next_doc = 0
        if not readonly:
            row = self.conn.execute("SELECT COALESCE(MAX(chunk_id)+1, 0) FROM chunks").fetchone()
            self._next_chunk = int(row[0])
            row = self.conn.execute("SELECT COALESCE(MAX(doc_id)+1, 0) FROM docs").fetchone()
            self._next_doc = int(row[0])

    # ---------------------------------------------------------------- escrita
    def add_document(self, title: str, url: str, source: str, ext_id: str, chunks: Sequence[str]) -> int:
        """Insere um documento e seus chunks. Retorna o número de chunks inseridos."""
        doc_id = self._next_doc
        self._next_doc += 1
        self.conn.execute(
            "INSERT INTO docs(doc_id, ext_id, title, url, source) VALUES (?,?,?,?,?)",
            (doc_id, ext_id, title, url, source),
        )
        rows = []
        for pos, text in enumerate(chunks):
            rows.append((self._next_chunk, doc_id, pos, text))
            self._next_chunk += 1
        self.conn.executemany("INSERT INTO chunks(chunk_id, doc_id, pos, text) VALUES (?,?,?,?)", rows)
        return len(rows)

    def commit(self) -> None:
        self.conn.commit()

    # ---------------------------------------------------------------- leitura
    def count_chunks(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def count_docs(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0])

    def total_chars(self) -> int:
        return int(self.conn.execute("SELECT COALESCE(SUM(LENGTH(text)),0) FROM chunks").fetchone()[0])

    def get_chunks(self, chunk_ids: Iterable[int]) -> list[ChunkRecord]:
        """Retorna os chunks na mesma ordem dos ids pedidos."""
        ids = [int(i) for i in chunk_ids]
        if not ids:
            return []
        found: dict[int, ChunkRecord] = {}
        for start in range(0, len(ids), 500):
            part = ids[start : start + 500]
            marks = ",".join("?" * len(part))
            with self._lock:
                rows = self.conn.execute(
                    f"""SELECT c.chunk_id, c.doc_id, c.pos, c.text, d.title, COALESCE(d.url,''), COALESCE(d.source,'')
                        FROM chunks c JOIN docs d ON d.doc_id = c.doc_id
                        WHERE c.chunk_id IN ({marks})""",
                    part,
                ).fetchall()
            for row in rows:
                found[row[0]] = ChunkRecord(*row)
        return [found[i] for i in ids if i in found]

    def iter_texts(self, start: int = 0, stop: int | None = None, batch: int = 2000) -> Iterator[tuple[int, list[str]]]:
        """Itera (primeiro_id, [textos]) em lotes sequenciais de chunk_id."""
        end = self.count_chunks() if stop is None else min(stop, self.count_chunks())
        cur_id = start
        while cur_id < end:
            hi = min(cur_id + batch, end)
            rows = self.conn.execute(
                "SELECT text FROM chunks WHERE chunk_id >= ? AND chunk_id < ? ORDER BY chunk_id", (cur_id, hi)
            ).fetchall()
            yield cur_id, [r[0] for r in rows]
            cur_id = hi

    def doc_titles(self) -> dict[int, str]:
        return {int(r[0]): r[1] for r in self.conn.execute("SELECT doc_id, title FROM docs")}

    def close(self) -> None:
        self.conn.close()

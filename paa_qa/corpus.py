"""Corpus: download em streaming (Hugging Face), leitura de documentos e divisão em chunks.

Formato do corpus em disco: shards JSONL (`part-00000.jsonl`, ...), um documento por linha com
os campos `id`, `title`, `url`, `source`, `text`. Assim cada trecho recuperado sabe de qual
documento veio (exigência de citação do enunciado). Arquivos `.txt` soltos também são aceitos
(um documento por arquivo, título = nome do arquivo).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .text import split_sentences

GIB = 1024**3
DEFAULT_SOURCES = ["wikimedia/wikipedia:20231101.pt", "wikimedia/wikipedia:20231101.en"]
MAX_TXT_BYTES = 256 * 1024 * 1024


@dataclass
class Document:
    doc_id: str
    title: str
    url: str
    source: str
    text: str


# ------------------------------------------------------------------ download
def parse_source(spec: str) -> tuple[str, str | None]:
    """'wikimedia/wikipedia:20231101.pt' -> ('wikimedia/wikipedia', '20231101.pt')."""
    if ":" in spec:
        name, config = spec.split(":", 1)
        return name, config or None
    return spec, None


def _manifest_path(out_dir: Path) -> Path:
    return out_dir / "manifest.json"


def _load_manifest(out_dir: Path) -> dict:
    path = _manifest_path(out_dir)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"sources": {}, "shards": 0, "text_bytes": 0, "docs": 0}


def _save_manifest(out_dir: Path, manifest: dict) -> None:
    tmp = _manifest_path(out_dir).with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(_manifest_path(out_dir))


def download_corpus(
    sources: list[str],
    out_dir: Path,
    target_gb: float = 10.0,
    split: str = "train",
    text_field: str = "text",
    title_field: str = "title",
    url_field: str = "url",
    id_field: str = "id",
    shard_mb: float = 128,
    log=print,
) -> dict:
    """Baixa texto em streaming até atingir `target_gb` GiB de texto puro (bytes UTF-8 do campo de texto).

    As fontes são consumidas em ordem. O download é retomável: o `manifest.json` guarda quantos
    documentos de cada fonte já foram gravados e, ao reexecutar, eles são pulados.
    """
    try:
        import datasets  # type: ignore
    except ImportError as exc:  # pragma: no cover - depende do ambiente
        raise RuntimeError("Pacote 'datasets' ausente. Instale com: pip install -r requirements.txt") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    target_bytes = int(target_gb * GIB)
    manifest = _load_manifest(out_dir)
    shard_limit = int(shard_mb * 1024 * 1024)
    started = time.time()
    shard_idx = manifest["shards"]  # próximo shard a escrever (shards antigos ficam intactos)
    handle = None
    shard_bytes = 0

    def open_shard():
        nonlocal handle, shard_idx, shard_bytes
        if handle:
            # Checkpoint só na troca de shard: o manifesto sempre descreve apenas shards fechados,
            # então uma queda no meio de um shard não deixa documentos duplicados ao retomar.
            handle.close()
            manifest["shards"] = shard_idx
            _save_manifest(out_dir, manifest)
        path = out_dir / f"part-{shard_idx:05d}.jsonl"
        handle = path.open("w", encoding="utf-8")
        shard_idx += 1
        shard_bytes = 0

    try:
        for spec in sources:
            if manifest["text_bytes"] >= target_bytes:
                break
            name, config = parse_source(spec)
            state = manifest["sources"].setdefault(spec, {"docs": 0, "text_bytes": 0, "exhausted": False})
            if state["exhausted"]:
                continue
            log(f"[download] fonte {spec}: já gravado {state['docs']} docs, {state['text_bytes']/GIB:.2f} GiB")
            stream = datasets.load_dataset(name, config, split=split, streaming=True)
            if state["docs"]:
                stream = stream.skip(state["docs"])
            exhausted = True
            for item in stream:
                text = (item.get(text_field) or "").strip()
                if not text:
                    state["docs"] += 1
                    continue
                if handle is None or shard_bytes >= shard_limit:
                    open_shard()
                record = {
                    "id": str(item.get(id_field, state["docs"])),
                    "title": (item.get(title_field) or f"doc-{state['docs']}").strip(),
                    "url": item.get(url_field) or "",
                    "source": spec,
                    "text": text,
                }
                line = json.dumps(record, ensure_ascii=False) + "\n"
                handle.write(line)
                shard_bytes += len(line.encode("utf-8"))
                nbytes = len(text.encode("utf-8"))
                state["docs"] += 1
                state["text_bytes"] += nbytes
                manifest["docs"] += 1
                manifest["text_bytes"] += nbytes
                if manifest["docs"] % 20000 == 0:
                    rate = manifest["text_bytes"] / max(time.time() - started, 1e-9) / (1024**2)
                    log(f"[download] {manifest['text_bytes']/GIB:.2f}/{target_gb:.2f} GiB ({rate:.1f} MiB/s)")
                if manifest["text_bytes"] >= target_bytes:
                    exhausted = False
                    break
            if exhausted:
                state["exhausted"] = True
    finally:
        if handle:
            handle.close()
        manifest["shards"] = shard_idx
        _save_manifest(out_dir, manifest)

    completed = manifest["text_bytes"] >= target_bytes
    result = {
        "output_dir": str(out_dir),
        "sources": sources,
        "docs": manifest["docs"],
        "text_bytes": manifest["text_bytes"],
        "text_gib": round(manifest["text_bytes"] / GIB, 3),
        "target_gib": target_gb,
        "completed": completed,
        "elapsed_s": round(time.time() - started, 1),
    }
    if not completed:
        result["warning"] = (
            "As fontes acabaram antes da meta. Adicione outra --source (ex.: wikimedia/wikipedia:20231101.es) "
            "e execute novamente: o download continua de onde parou."
        )
    return result


# ------------------------------------------------------------------ leitura
def _is_json_lines(path: Path) -> bool:
    """.jsonl do `download` ou arquivos `wiki_00`, `wiki_01`... (saída do WikiExtractor com --json)."""
    return path.suffix == ".jsonl" or path.name.startswith("wiki_")


def _iter_one(path: Path, limit_docs: int | None = None) -> Iterator[Document]:
    path = Path(path)
    files = [path] if path.is_file() else sorted(
        p for p in path.rglob("*") if p.is_file() and (p.suffix in (".jsonl", ".txt") or p.name.startswith("wiki_"))
    )
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo .jsonl, .txt ou wiki_* (WikiExtractor) em {path}")
    count = 0
    for file in files:
        if _is_json_lines(file):
            with file.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # uma linha corrompida não deve derrubar horas de indexação
                    text = (obj.get("text") or "").strip()
                    if not text:
                        continue
                    yield Document(
                        doc_id=str(obj.get("id", f"{file.name}:{line_no}")),
                        title=(obj.get("title") or file.stem).strip(),
                        url=obj.get("url") or "",
                        source=obj.get("source") or file.name,
                        text=text,
                    )
                    count += 1
                    if limit_docs and count >= limit_docs:
                        return
        else:
            if file.stat().st_size > MAX_TXT_BYTES:
                raise ValueError(
                    f"{file} tem mais de {MAX_TXT_BYTES // (1024**2)} MB. Use o comando 'download' (JSONL em shards) "
                    "ou divida o arquivo em documentos menores."
                )
            text = file.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue
            yield Document(doc_id=file.stem, title=file.stem.replace("_", " "), url=file.as_posix(), source=file.name, text=text)
            count += 1
            if limit_docs and count >= limit_docs:
                return


def iter_documents(paths: "Path | str | list[Path | str]", limit_docs: int | None = None,
                   min_chars: int = 0) -> Iterator[Document]:
    """Lê documentos de uma ou várias pastas/arquivos, na ordem dada, sem carregar tudo em memória.

    Formatos: `.jsonl` (comando `download`), `wiki_*` (WikiExtractor --json: campos id, url, title, text) e `.txt`.
    `min_chars` descarta páginas muito curtas (esboços, desambiguações), que só geram ruído na busca.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    count = 0
    for path in paths:
        for doc in _iter_one(Path(path)):
            if len(doc.text) < min_chars:
                continue
            yield doc
            count += 1
            if limit_docs and count >= limit_docs:
                return


# ------------------------------------------------------------------ chunking
def _hard_split(piece: str, size: int) -> list[str]:
    """Divide um trecho sem pontuação em fatias <= size, cortando em espaço."""
    out = []
    while len(piece) > size:
        cut = piece.rfind(" ", 0, size)
        if cut <= 0:
            cut = size
        out.append(piece[:cut].strip())
        piece = piece[cut:].strip()
    if piece:
        out.append(piece)
    return out


def chunk_text(text: str, chunk_size_chars: int = 1000, overlap_chars: int = 150) -> list[str]:
    """Agrupa sentenças em chunks de até `chunk_size_chars` caracteres.

    O overlap repete as últimas sentenças do chunk anterior (até `overlap_chars`), evitando cortar
    uma ideia ao meio. Custo O(n) no tamanho do texto.
    """
    if chunk_size_chars <= 0:
        raise ValueError("chunk_size_chars deve ser maior que zero")
    if overlap_chars < 0 or overlap_chars >= chunk_size_chars:
        raise ValueError("overlap_chars deve estar entre 0 e chunk_size_chars - 1")

    sentences: list[str] = []
    for sent in split_sentences(text):
        sentences.extend(_hard_split(sent, chunk_size_chars) if len(sent) > chunk_size_chars else [sent])
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for sent in sentences:
        add = len(sent) + (1 if current else 0)
        if current and length + add > chunk_size_chars:
            chunks.append(" ".join(current))
            carry: list[str] = []
            carry_len = 0
            for prev in reversed(current):
                if carry_len + len(prev) + 1 > overlap_chars:
                    break
                carry.insert(0, prev)
                carry_len += len(prev) + 1
            if carry and carry_len + len(sent) > chunk_size_chars:
                carry = []  # overlap + nova sentença estouraria o limite do chunk
            current = carry
            length = sum(len(s) for s in current) + max(len(current) - 1, 0)
            add = len(sent) + (1 if current else 0)
        current.append(sent)
        length += add
    if current:
        last = " ".join(current)
        if not chunks or last != chunks[-1]:
            chunks.append(last)
    return chunks

#!/usr/bin/env python3
"""Gera o zip de entrega do enunciado: PAA_<PrimeiroNome>_<matricula>_Proj.zip, com no máximo 100 MB.

Inclui código, testes, documentação, amostra dos dados e perguntas de avaliação. Não inclui o corpus
completo nem índices (o enunciado pede um LINK para download deles; ele vai em LINK_DOWNLOAD.txt).

Uso:
    python tools/package_submission.py --name Jose --matricula 12345 \
        --download-link "https://..." [--extra slides.pptx slides.pdf]
"""

from __future__ import annotations

import argparse
import re
import unicodedata
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INCLUDE_DIRS = ["paa_qa", "webscrapper", "tests", "tools", "docs", "data/sample", "data/eval"]
INCLUDE_FILES = ["README.md", "requirements.txt", "requirements-eval.txt", "requirements-dev.txt", "pytest.ini",
                 "semantic_qa.py", "data/queries.txt", ".gitignore", "app.py", "pyproject.toml", "uv.lock", ".python-version"]
EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".venv", "venv", "artifacts", "dist", ".git"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}
MAX_MB_DEFAULT = 100


def slug(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", ascii_value)
    if not cleaned:
        raise ValueError(f"Valor inválido para o nome do arquivo: {value!r}")
    return cleaned


def collect_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for rel in INCLUDE_DIRS:
        base = root / rel
        if base.exists():
            files.extend(p for p in sorted(base.rglob("*")) if p.is_file())
    files.extend(root / rel for rel in INCLUDE_FILES if (root / rel).is_file())
    return [p for p in files
            if not (set(p.relative_to(root).parts) & EXCLUDE_PARTS) and p.suffix not in EXCLUDE_SUFFIXES]


def build_package(root: Path, name: str, matricula: str, out_dir: Path, download_link: str | None = None,
                  extra: list[Path] | None = None, max_mb: float = MAX_MB_DEFAULT) -> Path:
    zip_name = f"PAA_{slug(name)}_{slug(matricula)}_Proj.zip"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / zip_name
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in collect_files(root):
            archive.write(path, path.relative_to(root).as_posix())
        for path in extra or []:
            archive.write(path, f"extras/{Path(path).name}")
        link = download_link or "PREENCHER: link para download do corpus completo (10 GB), índices e demais arquivos."
        archive.writestr(
            "LINK_DOWNLOAD.txt",
            f"Banco de dados completo e arquivos complementares:\n{link}\n\n"
            "Reproduzir o corpus: python -m paa_qa download --target-gb 10 --output data/corpus\n",
        )
    size_mb = target.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        target.unlink()
        raise ValueError(f"O zip ficou com {size_mb:.1f} MB, acima do limite de {max_mb} MB. "
                         "Reduza a amostra em data/sample (python -m paa_qa sample --max-mb ...).")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="primeiro nome de quem envia")
    parser.add_argument("--matricula", required=True)
    parser.add_argument("--download-link", default=None)
    parser.add_argument("--extra", nargs="*", type=Path, default=[], help="arquivos extras (ex.: slides .pptx e .pdf)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--max-mb", type=float, default=MAX_MB_DEFAULT)
    args = parser.parse_args()
    target = build_package(ROOT, args.name, args.matricula, args.output_dir, args.download_link, args.extra, args.max_mb)
    print(f"Gerado: {target} ({target.stat().st_size / (1024 * 1024):.2f} MB)")
    if not args.download_link:
        print("Aviso: informe --download-link (o enunciado exige o link do banco de dados completo).")


if __name__ == "__main__":
    main()

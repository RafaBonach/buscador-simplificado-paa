"""Integração do WikiExtractor com o projeto."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def extract_dump(
    dump_path: Path,
    output_dir: Path,
    *,
    processes: int = 4,
    bytes_per_file: str = "100M",
) -> None:
    """Extrai um dump XML da Wikipédia em arquivos JSON."""
    dump_path = dump_path.resolve()
    output_dir = output_dir.resolve()

    if not dump_path.is_file():
        raise FileNotFoundError(f"Dump não encontrado: {dump_path}")
    if processes < 1:
        raise ValueError("O número de processos deve ser maior que zero.")

    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "wikiextractor.WikiExtractor",
        str(dump_path),
        "--output",
        str(output_dir),
        "--json",
        "--no-templates",
        "--processes",
        str(processes),
        "--bytes",
        bytes_per_file,
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extrai um dump da Wikipédia usando WikiExtractor."
    )
    parser.add_argument("dump", type=Path, help="Caminho do dump XML(.bz2).")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/extracted"),
        help="Diretório dos arquivos JSON (padrão: data/extracted).",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=4,
        help="Número de processos paralelos (padrão: 4).",
    )
    parser.add_argument(
        "--bytes",
        dest="bytes_per_file",
        default="100M",
        help="Tamanho máximo de cada arquivo (padrão: 100M).",
    )
    args = parser.parse_args()

    extract_dump(
        args.dump,
        args.output,
        processes=args.processes,
        bytes_per_file=args.bytes_per_file,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("package_submission", ROOT / "tools" / "package_submission.py")
pkg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pkg)


def test_nome_do_arquivo_segue_o_enunciado(tmp_path):
    target = pkg.build_package(ROOT, "José", "12/345", tmp_path, download_link="http://link")
    assert target.name == "PAA_Jose_12345_Proj.zip"


def test_conteudo_do_zip(tmp_path):
    (ROOT / "paa_qa" / "__pycache__").mkdir(exist_ok=True)
    target = pkg.build_package(ROOT, "Ana", "999", tmp_path, download_link="http://link")
    with zipfile.ZipFile(target) as archive:
        names = set(archive.namelist())
        link = archive.read("LINK_DOWNLOAD.txt").decode()
    assert {"README.md", "paa_qa/index.py", "data/sample/demo_corpus.jsonl", "data/eval/questions.json",
            "LINK_DOWNLOAD.txt", "semantic_qa.py"} <= names
    assert not any("__pycache__" in n or n.endswith(".pyc") for n in names)
    assert not any(n.startswith(("artifacts/", "data/corpus/", "dist/")) for n in names)
    assert "http://link" in link


def test_zip_leva_app_dependencias_e_extracao_do_grupo(tmp_path):
    root = tmp_path / "repo"
    for rel in ("app.py", "pyproject.toml", ".python-version", "webscrapper/wikipedia.py", "paa_qa/x.py",
                "data/wikipedia_pt/extracted/AA/wiki_00", "models/pesado.bin"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("x", encoding="utf-8")
    target = pkg.build_package(root, "Ana", "1", tmp_path / "out", download_link="http://link")
    with zipfile.ZipFile(target) as archive:
        names = set(archive.namelist())
    assert {"app.py", "pyproject.toml", ".python-version", "webscrapper/wikipedia.py", "paa_qa/x.py"} <= names
    assert not any(n.startswith(("data/wikipedia_pt", "models/")) for n in names)  # dados e modelos ficam fora do zip


def test_limite_de_tamanho_remove_o_zip(tmp_path):
    with pytest.raises(ValueError, match="acima do limite"):
        pkg.build_package(ROOT, "Ana", "1", tmp_path, max_mb=0.0001)
    assert not list(tmp_path.glob("*.zip"))


def test_nome_vazio_e_rejeitado(tmp_path):
    with pytest.raises(ValueError):
        pkg.build_package(ROOT, "!!!", "1", tmp_path)

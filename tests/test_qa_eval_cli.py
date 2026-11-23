from __future__ import annotations

import json
import sys

import pytest

from paa_qa.benchmark import render_markdown, run_benchmark
from paa_qa.cli import main
from paa_qa.encoders import MissingDependency
from paa_qa.evaluate import (context_recall_lexical, evaluate_system, latency_stats, retrieval_metrics, token_f1)
from paa_qa.index import Hit
from paa_qa.llm import NO_ANSWER, build_messages, cited_indices, extractive_answer
from paa_qa.qa import QASystem
from paa_qa.ragas_eval import run_ragas
from paa_qa.store import ChunkRecord


def make_hit(rank: int, text: str, title: str = "Doc") -> Hit:
    return Hit(rank=rank, score=1.0 / rank, chunk=ChunkRecord(rank, rank, 0, text, f"{title} {rank}", "", ""))


# ------------------------------------------------------------------ formatação da resposta
def test_extrativo_escolhe_a_frase_relevante_e_cita_a_origem():
    hits = [make_hit(1, "O céu é azul durante o dia. Gatos dormem bastante."),
            make_hit(2, "O algoritmo HNSW usa um grafo em camadas para buscar vizinhos próximos.")]
    answer = extractive_answer("Como o HNSW busca vizinhos?", hits)
    assert "HNSW" in answer and answer.endswith("[2]") and "Gatos" not in answer


def test_extrativo_prefere_cobertura_da_pergunta_a_frases_curtas():
    hits = [make_hit(1, "O esquema TF-IDF atribui a cada termo um peso igual à frequência no documento vezes o IDF. "
                        "O TF-IDF não reconhece sinônimos.")]
    answer = extractive_answer("Como o TF-IDF calcula o peso de um termo?", hits)
    assert "atribui a cada termo um peso" in answer and "sinônimos" not in answer


def test_extrativo_sem_contexto_util():
    assert extractive_answer("pergunta qualquer", []) == NO_ANSWER
    assert extractive_answer("zebra quântica", [make_hit(1, "Texto totalmente diferente sobre cozinha caseira.")]) == NO_ANSWER


def test_extrativo_nao_repete_sentenca_do_overlap():
    sentence = "O algoritmo HNSW usa um grafo em camadas para buscar vizinhos."
    answer = extractive_answer("HNSW grafo camadas", [make_hit(1, sentence), make_hit(2, sentence)])
    assert answer.count("HNSW usa") == 1


def test_prompt_numera_trechos_e_cita_o_documento():
    messages = build_messages("Pergunta?", [make_hit(1, "texto um", "Alfa"), make_hit(2, "texto dois", "Beta")])
    user = messages[1]["content"]
    assert "[1] Documento: Alfa 1" in user and "[2] Documento: Beta 2" in user and "Pergunta?" in user
    assert messages[0]["role"] == "system"
    assert cited_indices("Fato [2] e outro [1][2].") == [1, 2]


# ------------------------------------------------------------------ métricas
def test_token_f1():
    assert token_f1("a resposta correta", "a resposta correta") == pytest.approx(1.0)
    assert token_f1("totalmente diferente", "nada parecido aqui") == 0.0
    assert token_f1("resposta correta [1]", "resposta correta") == pytest.approx(1.0)  # ignora marcador de citação


def test_context_recall_lexical():
    assert context_recall_lexical(["gatos dormem muito"], "gatos dormem") == 1.0
    assert context_recall_lexical(["cachorros"], "gatos dormem") == 0.0


def test_retrieval_metrics():
    m = retrieval_metrics(["B", "Á", "C"], ["a"])  # case/acento ignorados
    assert m == {"hit": 1.0, "mrr": 0.5, "recall": 1.0}
    assert retrieval_metrics(["X"], ["A"]) == {"hit": 0.0, "mrr": 0.0, "recall": 0.0}


def test_latency_stats():
    stats = latency_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert stats["mean"] == 3.0 and stats["p50"] == 3.0 and stats["min"] == 1.0 and stats["max"] == 5.0
    assert stats["p95"] == pytest.approx(4.8)


# ------------------------------------------------------------------ pipeline completo
def test_ask_devolve_resposta_com_citacoes_e_tempos(built_index):
    qa = QASystem.load(built_index)
    answer = qa.ask("O que é um índice invertido?", method="cosine", top_k=3)
    assert answer.citations[0].title == "Busca lexical e TF-IDF"
    assert answer.cited() and set(answer.timings_ms) == {"retrieval", "generation", "total"}
    rendered = answer.render()
    assert "=== CITAÇÕES" in rendered and "Busca lexical e TF-IDF" in rendered
    assert json.dumps(answer.to_dict(), ensure_ascii=False)  # serializável
    qa.searcher.close()


def test_metodo_padrao_cai_para_o_primeiro_disponivel(built_index):
    qa = QASystem.load(built_index)
    assert qa.default_method() == "cosine"  # hnsw não foi indexado
    qa.searcher.close()


def test_evaluate_system_resume_metricas(built_index, demo_questions):
    qa = QASystem.load(built_index)
    result = evaluate_system(qa, demo_questions, method="cosine", top_k=5)
    summary = result["summary"]
    assert summary["questions"] == len(demo_questions) and summary["hit_at_k"] >= 0.9
    assert summary["answers_with_citation"] > 0.9 and summary["latency_ms"]["total"]["p95"] >= 0
    assert result["rows"][0]["contexts"] and result["generator"] == "extractive"
    qa.searcher.close()


def test_benchmark_compara_metodos_e_gera_tabela(built_index, demo_questions):
    report = run_benchmark(built_index, demo_questions, top_k=5, repeats=2)
    assert report["methods"]["cosine"]["quality"]["hit_at_k"] >= 0.9
    assert report["methods"]["sentence_embeddings"]["latency_ms"]["mean"] > 0
    assert report["methods"]["hnsw"]["skipped"] and report["methods"]["word2vec_average"]["skipped"]
    assert {"fastest_query", "best_quality_mrr", "best_tradeoff"} <= set(report["ranking"])
    table = render_markdown(report)
    assert "| cosine |" in table and "Complexidade teórica" in table
    assert "Latência p50 (ms) | Média (ms) | p95 (ms)" in table
    assert "não são significativas" in table  # amostra pequena: o relatório avisa


def test_ranking_usa_a_mediana_e_nao_a_media():
    from paa_qa.benchmark import _rank

    def entry(mean, p50, mrr):
        return {"latency_ms": {"mean": mean, "p50": p50, "p95": p50}, "quality": {"mrr": mrr}}

    # "a" tem uma medição atípica (média alta) mas é a mais rápida na maioria das consultas
    ranking = _rank({"a": entry(mean=30.0, p50=1.0, mrr=0.9), "b": entry(mean=5.0, p50=4.0, mrr=0.9)})
    assert ranking["fastest_query"] == "a" and ranking["best_tradeoff"] == "a"


def test_ragas_sem_pacote_da_erro_com_instrucao(monkeypatch):
    monkeypatch.setitem(sys.modules, "ragas", None)
    with pytest.raises(MissingDependency, match="pip install"):
        run_ragas([{"question": "q", "answer": "a", "contexts": ["c"], "reference": "r"}], judge_base_url="http://x/v1")


# ------------------------------------------------------------------ CLI
def test_cli_ask_json(built_index, capsys):
    main(["ask", "--index-dir", str(built_index), "--question", "Qual a fórmula da similaridade de cosseno?",
          "--method", "cosine", "--top-k", "2", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["citations"][0]["title"] == "Similaridade de cosseno" and payload["generator"] == "extractive"


def test_cli_evaluate_e_stats(built_index, tmp_path, capsys):
    out = tmp_path / "eval.json"
    main(["evaluate", "--index-dir", str(built_index), "--questions", "data/eval/questions.json",
          "--method", "cosine", "--output", str(out)])
    assert json.loads(out.read_text(encoding="utf-8"))["summary"]["hit_at_k"] >= 0.9
    main(["stats", "--index-dir", str(built_index), "--target-gb", "10"])
    assert "Projeção linear" in capsys.readouterr().out


def test_cli_erro_amigavel_sem_indice(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["ask", "--index-dir", str(tmp_path / "nada"), "--question", "x"])
    assert exc.value.code == 2 and "build-index" in capsys.readouterr().err


def test_cli_sample(tmp_path, capsys):
    out = tmp_path / "s.jsonl"
    main(["sample", "--corpus", "data/sample/demo_corpus.jsonl", "--output", str(out), "--max-mb", "0.001"])
    lines = out.read_text(encoding="utf-8").splitlines()
    assert 1 <= len(lines) < 12 and json.loads(lines[0])["title"]


def test_arquivos_de_perguntas_do_projeto_tem_o_formato_esperado():
    from pathlib import Path

    from paa_qa.evaluate import load_questions

    root = Path(__file__).resolve().parent.parent
    for name in ("questions.json", "questions_real.json"):
        questions = load_questions(root / "data" / "eval" / name)
        assert len(questions) >= 14
        for item in questions:
            assert item["question"].strip().endswith("?")
            assert item["reference"].strip()
            assert item["relevant_titles"] and all(isinstance(t, str) and t for t in item["relevant_titles"])

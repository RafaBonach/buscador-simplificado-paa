"""Avaliação com o framework RAGAS usando juiz LOCAL (sem OpenAI).

Alvo de API: ragas 0.2.x / 0.3.x (`EvaluationDataset`, `evaluate`, `LangchainLLMWrapper`).
O juiz pode ser:
  * um modelo local do Hugging Face (`--judge-model-path`), via langchain-huggingface; ou
  * um servidor local compatível com a API OpenAI (Ollama, llama.cpp, vLLM) em `--judge-base-url`.

Atenção: modelos pequenos (< 7B) como juiz costumam falhar em produzir o JSON que o RAGAS exige. Linhas
que falham ficam como NaN e são EXCLUÍDAS da média (e contadas em `n_failed`), nunca tratadas como zero.
"""

from __future__ import annotations

import math

from .encoders import DEFAULT_EMBEDDING_MODEL, MissingDependency

INSTALL_HINT = "pip install 'ragas>=0.2,<0.4' langchain-huggingface langchain-openai"


def _need(module: str, package: str):
    try:
        return __import__(module, fromlist=["_"])
    except ImportError as exc:
        raise MissingDependency(f"Pacote '{package}' ausente para o RAGAS. Instale com: {INSTALL_HINT}") from exc


def build_judge_llm(judge_model_path: str | None, judge_base_url: str | None, judge_model: str | None,
                    max_new_tokens: int = 512, timeout: int = 600):
    if judge_base_url:
        chat_openai = _need("langchain_openai", "langchain-openai").ChatOpenAI
        return chat_openai(model=judge_model or "local-model", base_url=judge_base_url, api_key="not-needed",
                           temperature=0, timeout=timeout)
    if judge_model_path:
        lc_hf = _need("langchain_huggingface", "langchain-huggingface")
        pipe = lc_hf.HuggingFacePipeline.from_model_id(
            model_id=judge_model_path, task="text-generation",
            pipeline_kwargs={"max_new_tokens": max_new_tokens, "do_sample": False, "return_full_text": False},
        )
        return lc_hf.ChatHuggingFace(llm=pipe)
    raise ValueError("Informe --judge-model-path (modelo local) ou --judge-base-url (servidor local OpenAI-compatível).")


def run_ragas(rows: list[dict], judge_model_path: str | None = None, judge_base_url: str | None = None,
              judge_model: str | None = None, embedding_model: str = DEFAULT_EMBEDDING_MODEL,
              max_workers: int = 1, timeout: int = 600) -> dict:
    """`rows` vem de `evaluate_system(...)['rows']`; só entram linhas com `reference`."""
    ragas = _need("ragas", "ragas")
    metrics_mod = _need("ragas.metrics", "ragas")
    llms_mod = _need("ragas.llms", "ragas")
    emb_mod = _need("ragas.embeddings", "ragas")
    run_cfg_mod = _need("ragas.run_config", "ragas")
    lc_hf = _need("langchain_huggingface", "langchain-huggingface")

    samples = [
        {"user_input": r["question"], "retrieved_contexts": r["contexts"], "response": r["answer"],
         "reference": r["reference"]}
        for r in rows if r.get("reference")
    ]
    if not samples:
        raise ValueError("Nenhuma pergunta com 'reference' — o RAGAS precisa da resposta esperada.")

    judge = llms_mod.LangchainLLMWrapper(build_judge_llm(judge_model_path, judge_base_url, judge_model, timeout=timeout))
    embeddings = emb_mod.LangchainEmbeddingsWrapper(lc_hf.HuggingFaceEmbeddings(model_name=embedding_model))
    metrics = [
        metrics_mod.Faithfulness(),
        metrics_mod.ResponseRelevancy(),
        metrics_mod.LLMContextPrecisionWithReference(),
        metrics_mod.LLMContextRecall(),
    ]
    dataset = ragas.EvaluationDataset.from_list(samples)
    result = ragas.evaluate(
        dataset, metrics=metrics, llm=judge, embeddings=embeddings,
        run_config=run_cfg_mod.RunConfig(timeout=timeout, max_workers=max_workers), raise_exceptions=False,
    )
    frame = result.to_pandas()
    summary: dict[str, dict] = {}
    for column in frame.select_dtypes("number").columns:
        values = [float(v) for v in frame[column].tolist()]
        valid = [v for v in values if not math.isnan(v)]
        summary[column] = {
            "mean": sum(valid) / len(valid) if valid else None,
            "n_valid": len(valid),
            "n_failed": len(values) - len(valid),
        }
    per_sample = [
        {key: (None if isinstance(value, float) and math.isnan(value) else value) for key, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]  # NaN -> null: json.dumps escreveria NaN, que não é JSON válido
    return {"judge": judge_model_path or f"{judge_base_url} ({judge_model})", "embedding_model": embedding_model,
            "samples": len(samples), "metrics": summary, "per_sample": per_sample}

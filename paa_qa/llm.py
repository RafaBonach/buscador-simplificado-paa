"""Formatação da resposta: LLM local (transformers) ou fallback extrativo, ambos sem nuvem."""

from __future__ import annotations

import math
import os
import re
import sys

from .encoders import MissingDependency
from .index import Hit
from .text import content_tokens, fold_accents, shorten, split_sentences

NO_ANSWER = "Não encontrei nos trechos recuperados informação suficiente para responder."
CITATION_RE = re.compile(r"\[(\d+)\]")

SYSTEM_PROMPT = (
    "Você é um assistente que responde perguntas em português do Brasil usando APENAS os trechos numerados "
    "fornecidos. Seja objetivo (no máximo 4 frases). Ao final de cada afirmação cite a fonte no formato [n], "
    "onde n é o número do trecho. Se os trechos não contiverem a resposta, diga que não encontrou a informação."
)


# ---------------------------------------------------------------- memória
class InsufficientMemory(RuntimeError):
    """O modelo não cabe na RAM livre; falhar com uma mensagem clara é melhor que o Python fechar sozinho."""


# Tamanho dos pesos em 16 bits (GB), aproximado, dos modelos do enunciado. Em float32 ocupam o dobro.
WEIGHTS_GB_16BIT = {
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 2.2,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": 3.6,
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": 3.4,
    "microsoft/Phi-3-mini-4k-instruct": 7.6,
    "mistralai/Mistral-7B-Instruct-v0.3": 14.5,
}
RAM_OVERHEAD = 1.25  # ativações, cache de atenção e o próprio Python


def available_ram_gb() -> float | None:
    """RAM física livre em GiB, ou None se não der para saber."""
    try:
        import psutil  # type: ignore

        return psutil.virtual_memory().available / 1024**3
    except ImportError:
        pass
    try:
        if sys.platform == "win32":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))  # type: ignore[attr-defined]
            return status.ullAvailPhys / 1024**3
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024**2
    except (OSError, ValueError, AttributeError):
        pass
    return None


def choose_dtype(model_id: str, device: str, available_gb: float | None, forced: str | None = None) -> str:
    """Escolhe a precisão: float32 se couber folgado na RAM (mais rápido em CPU), senão bfloat16 (metade da
    memória). Se nem assim couber, avisa em vez de deixar o processo morrer. `forced` vem de PAA_LLM_DTYPE."""
    if forced:
        if forced not in ("float32", "bfloat16", "float16"):
            raise ValueError("PAA_LLM_DTYPE deve ser float32, bfloat16 ou float16")
        return forced
    gb16 = WEIGHTS_GB_16BIT.get(model_id)
    if device == "cuda":
        return "float16"
    if gb16 is None or available_gb is None:
        return "float32"  # modelo desconhecido ou RAM desconhecida: comportamento antigo
    if available_gb >= gb16 * 2 * RAM_OVERHEAD:
        return "float32"
    if available_gb >= gb16 * RAM_OVERHEAD:
        return "bfloat16"
    raise InsufficientMemory(
        f"Memória insuficiente para o modelo {model_id}: precisa de cerca de {gb16 * RAM_OVERHEAD:.1f} GB livres "
        f"(mesmo em precisão reduzida) e há só {available_gb:.1f} GB. Feche outros programas ou escolha um LLM "
        f"menor (TinyLlama ou 'Sem LLM')."
    )


def build_context_block(hits: list[Hit], max_chars: int = 900) -> str:
    return "\n\n".join(
        f"[{h.rank}] Documento: {h.chunk.title}\nTrecho: {' '.join(h.chunk.text.split())[:max_chars]}" for h in hits
    )


def build_messages(question: str, hits: list[Hit], max_chars: int = 900) -> list[dict]:
    user = f"Trechos:\n{build_context_block(hits, max_chars)}\n\nPergunta: {question}\n\nResposta:"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def cited_indices(answer: str) -> list[int]:
    return sorted({int(m) for m in CITATION_RE.findall(answer)})


# Nomes usados no seletor do app (e em --llm) -> modelos do Hugging Face.
MODEL_ALIASES = {
    "smollm": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "smollm2": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "phi-3 mini": "microsoft/Phi-3-mini-4k-instruct",
    "phi-3": "microsoft/Phi-3-mini-4k-instruct",
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}


def resolve_model(name: str) -> str:
    """'SmolLM' -> 'HuggingFaceTB/SmolLM2-1.7B-Instruct'. Qualquer outro valor (id HF ou pasta) passa direto."""
    return MODEL_ALIASES.get(name.strip().lower(), name)


def strip_reasoning(text: str) -> str:
    """Remove o bloco <think>...</think> de modelos de raciocínio (DeepSeek-R1). Sem o bloco, não muda nada."""
    if "</think>" in text:  # o template novo já abre <think>; só o fechamento aparece na saída
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:  # raciocínio cortado pelo limite de tokens: não há resposta
        text = text.split("<think>", 1)[0]
    return text.strip()


# ---------------------------------------------------------------- fallback extrativo
def extractive_answer(question: str, hits: list[Hit], max_sentences: int = 3) -> str:
    """Escolhe as sentenças dos trechos recuperados mais parecidas com a pergunta e cita a origem [n].

    Determinístico, roda sem modelo. Pontuação: cobertura dos termos da pergunta pela sentença (a
    medida que menos favorece frases curtas), com desempate por cosseno de tokens e pelo ranking da busca.
    """
    q_tokens = set(content_tokens(question))
    if not hits or not q_tokens:
        return NO_ANSWER
    candidates: list[tuple[float, int, int, str]] = []
    seen: set[str] = set()
    for hit in hits:
        for pos, sentence in enumerate(split_sentences(hit.chunk.text)):
            if len(sentence) < 20:
                continue
            key = fold_accents(sentence).lower()
            if key in seen:
                continue  # o overlap entre chunks repete sentenças
            seen.add(key)
            s_tokens = set(content_tokens(sentence))
            overlap = len(q_tokens & s_tokens)
            if overlap == 0:
                continue
            coverage = overlap / len(q_tokens)  # fração dos termos da pergunta presentes na sentença
            cosine = overlap / math.sqrt(len(q_tokens) * max(len(s_tokens), 1))
            score = coverage + 0.1 * cosine + 0.02 / hit.rank
            candidates.append((score, hit.rank, pos, sentence))
    if not candidates:
        return NO_ANSWER
    ranked = sorted(candidates, key=lambda c: -c[0])
    cutoff = 0.6 * ranked[0][0]  # descarta sentenças com sobreposição fraca frente à melhor
    best = [c for c in ranked if c[0] >= cutoff][:max_sentences]
    best.sort(key=lambda c: (c[1], c[2]))  # ordem de leitura: ranking do trecho e posição
    return " ".join(f"{shorten(sentence, 400)} [{rank}]" for _, rank, _, sentence in best)


class ExtractiveFormatter:
    name = "extractive"

    def generate(self, question: str, hits: list[Hit]) -> str:
        return extractive_answer(question, hits)


# ---------------------------------------------------------------- LLM local
class LocalLLM:
    """LLM local via transformers (SmolLM2, TinyLlama, Phi-3 Mini, Mistral, DeepSeek-distill, Qwen...).

    `model` é um diretório local ou um id do Hugging Face (baixado uma vez para o cache). Para impedir
    qualquer acesso à rede defina HF_HUB_OFFLINE=1 depois de rodar `download-models`.
    """

    def __init__(self, model: str, max_new_tokens: int = 256, device: str | None = None, max_context_chars: int = 900):
        self.model_id = resolve_model(model)
        self.name = model
        # O DeepSeek-R1 "pensa" antes de responder; com poucos tokens a resposta nem começa.
        self.max_new_tokens = max(max_new_tokens, 768) if "deepseek" in self.model_id.lower() else max_new_tokens
        self.device = device
        self.max_context_chars = max_context_chars
        self._tok = None
        self._model = None
        self._torch = None
        self.dtype_name = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except ImportError as exc:
            raise MissingDependency("Pacotes 'torch' e 'transformers' ausentes. Instale com: pip install -r requirements.txt") from exc
        self._torch = torch
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        dtype_name = choose_dtype(self.model_id, device, available_ram_gb(), os.environ.get("PAA_LLM_DTYPE"))
        dtype = getattr(torch, dtype_name)
        self.dtype_name = dtype_name
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        try:
            model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=dtype)
        except TypeError:  # versões antigas do transformers usam torch_dtype
            model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=dtype)
        self._model = model.to(device).eval()
        if self._tok.pad_token_id is None:
            self._tok.pad_token = self._tok.eos_token

    def _prompt(self, question: str, hits: list[Hit]) -> str:
        messages = build_messages(question, hits, self.max_context_chars)
        merged = [{"role": "user", "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}]
        if getattr(self._tok, "chat_template", None):
            if "deepseek" in self.model_id.lower():  # a recomendação do DeepSeek-R1 é não usar prompt de sistema
                return self._tok.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)
            try:
                return self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:  # alguns templates (ex.: Mistral) não aceitam o papel "system"
                return self._tok.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)
        return messages[0]["content"] + "\n\n" + messages[1]["content"] + "\n"

    def _prompt_budget(self) -> int:
        """Tokens disponíveis para o prompt: janela do modelo menos o espaço da resposta."""
        limit = getattr(self._tok, "model_max_length", None)
        if not isinstance(limit, int) or limit > 100_000:  # alguns tokenizers declaram "infinito"
            limit = 4096
        return max(limit - self.max_new_tokens, 256)

    def generate(self, question: str, hits: list[Hit]) -> str:
        if not hits:
            return NO_ANSWER
        self.load()
        torch = self._torch
        budget = self._prompt_budget()
        for keep in range(len(hits), 0, -1):  # TinyLlama tem só 2048 tokens: tira os trechos menos relevantes até caber
            inputs = self._tok(self._prompt(question, hits[:keep]), return_tensors="pt", add_special_tokens=False)
            if inputs["input_ids"].shape[1] <= budget:
                break
        inputs = inputs.to(self.device)
        with torch.no_grad():
            output = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                          pad_token_id=self._tok.pad_token_id)
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return strip_reasoning(self._tok.decode(new_tokens, skip_special_tokens=True)) or NO_ANSWER

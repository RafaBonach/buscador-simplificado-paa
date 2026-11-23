# Buscador Simplificado PAA

## Sobre

Buscador de perguntas e respostas em linguagem natural para a disciplina de *Projeto e Análise de Algoritmos 2/2026* (UnB).
O sistema pesquisa um corpus grande de texto (meta: **10 GB**) por **quatro métodos de busca semântica**, monta a
resposta com um **LLM local** e **cita o documento de origem**. A interface é um app Streamlit (`app.py`); o motor
fica no pacote `paa_qa/`, que também tem linha de comando para download, indexação, benchmark e avaliação (RAGAS).

```
pergunta ─► busca semântica (cosseno/TF-IDF · Word2Vec · Sentence Embeddings · HNSW)
        ─► trechos numerados ─► LLM local (ou resposta extrativa) ─► resposta [1][2] + documentos de origem + tempos
```

## O que o enunciado pede e onde está

| Exigência | Onde |
|---|---|
| Baixar ≥ 10 GB de texto puro de uma base pública | dumps da Wikipédia (seções abaixo) **ou** `download` (Hugging Face, retomável) |
| Processar o texto para busca semântica | `build-index` — chunks por sentença + índices em disco (SQLite, memmap) |
| **Cosseno (TF-IDF)**, **Word2Vec Average**, **Sentence Embeddings**, **HNSW** | `paa_qa/index.py` |
| Dizer qual é melhor em desempenho e complexidade | `benchmark` (tempo, qualidade, build, tamanho) + tabela de complexidade |
| Resposta formatada por LLM local (SmolLM, TinyLlama, Phi-3 Mini, Mistral, DeepSeek) | menu do app / `ask --llm <modelo>`; sem LLM, fallback extrativo |
| Mostrar o documento de onde veio a informação | citações `[n]`: título (com link), trecho e score |
| Framework de avaliação (RAGAS...) com acurácia e tempo de resposta | `evaluate` (métricas próprias + `--ragas` com juiz local) |
| Zip ≤ 100 MB com código e amostra; link para o banco completo | `tools/package_submission.py` + `sample` |
| Apresentação de 10 min com 14 itens | `docs/roteiro_apresentacao.md` indica de onde sai cada número |

## Instalação com `uv`

Pré-requisitos: [`uv`](https://docs.astral.sh/uv/) instalado. O `uv` baixa o Python certo sozinho: o arquivo
`.python-version` pede o **3.12** (use 3.12 ou 3.13; o 3.14+ ainda não tem pacotes prontos de `torch`/`gensim`).

Num Windows recém-instalado, instale antes (PowerShell):

1. **Microsoft Visual C++ Redistributable** (x64): https://aka.ms/vc14/vc_redist.x64.exe. Sem ele o `torch` falha ao importar com
   `WinError 126 ... shm.dll or one of its dependencies`.
2. **uv**: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"` (ou `winget install --id=astral-sh.uv -e`).
   Feche e reabra o PowerShell depois.

Na raiz do projeto:

```bash
uv sync                      # instala tudo que está no pyproject.toml (demora na primeira vez: torch é grande)
uv sync --extra eval         # + RAGAS, só se for rodar `evaluate --ragas`
```

> O `uv sync` cria/atualiza o `uv.lock`. Não é mais preciso rodar `uv init --bare` nem `uv add -r requirements.txt`:
> as dependências já estão no `pyproject.toml`. Sem `uv`: `python -m venv .venv`, ative, `pip install -r requirements.txt`
> e troque `uv run python` por `python` nos comandos abaixo.

Se uma biblioteca faltar, o `build-index` **continua** com os métodos possíveis e informa quais foram ignorados e por quê.

## Teste rápido do app (sem baixar nada)

O repositório traz um corpus de demonstração (12 documentos originais em português — **não** é Wikipédia) e 14 perguntas:

```bash
uv run python -m paa_qa build-index --data data/sample/demo_corpus.jsonl --index-dir artifacts/index
uv run streamlit run app.py
```

O app abre no navegador. Na barra lateral escolha o **método de similaridade** e o **modelo LLM** (a primeira opção,
*Sem LLM*, monta a resposta com frases dos trechos e não baixa nada). Pergunte, por exemplo, "Como funciona a busca no HNSW?".
Abaixo de cada resposta aparecem **Fontes e Citações** (documento de origem) e os **tempos** de busca e geração.

Só quer exercitar o pipeline sem baixar o modelo de embeddings? Acrescente `--embedding-model hashing` ao `build-index`.
É um codificador de teste **sem semântica real** — nunca use para medir qualidade.

Pela linha de comando:

```bash
uv run python -m paa_qa ask --index-dir artifacts/index --question "Como funciona a busca no HNSW?" --method hnsw
uv run python -m paa_qa benchmark --index-dir artifacts/index --eval-file data/eval/questions.json
uv run python -m paa_qa evaluate  --index-dir artifacts/index --questions data/eval/questions.json --method hnsw
```

O ~1 s de "busca" num `ask` isolado é a **partida a frio** (a primeira consulta carrega vetorizador, matriz e modelos).
No app, no `chat`, no `benchmark` e no `evaluate` o carregamento acontece antes de medir. Para comparar métodos use o `benchmark`.

## Fluxo completo com o corpus real

### 1. Obter o texto (≥ 10 GB) — duas opções

#### Opção A — dumps da Wikipédia (fluxo do grupo)

Na raiz do projeto, copie e execute o bloco abaixo. Ele cria as estruturas necessárias e baixa os dumps em português e inglês:

```bash
mkdir -p data/wikipedia_pt/raw data/wikipedia_pt/extracted
mkdir -p data/wikipedia_en/raw data/wikipedia_en/extracted

wget -c \
  https://dumps.wikimedia.org/ptwiki/latest/ptwiki-latest-pages-articles-multistream.xml.bz2 \
  -P data/wikipedia_pt/raw

wget -c \
  https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2 \
  -P data/wikipedia_en/raw
```

Depois de baixar, execute o WikiExtractor duas vezes. Ele não faz parte do `uv sync` (a versão do PyPI quebra no Python 3.11+ e a do GitHub
precisa ser compilada, o que exige o **Git** instalado e falha em alguns Windows), então entra só nesta execução, com `uv run --with`:

- **1º - Extração do dump em <u>português</u>**

```bash
uv run --with "wikiextractor @ git+https://github.com/attardi/wikiextractor.git" python -m webscrapper.wikipedia \
  data/wikipedia_pt/raw/ptwiki-latest-pages-articles-multistream.xml.bz2 \
  --output data/wikipedia_pt/extracted \
  --processes 4 \
  --bytes 100M
```

- **2ª - Extração do dump em <u>inglês</u>**

```bash
uv run --with "wikiextractor @ git+https://github.com/attardi/wikiextractor.git" python -m webscrapper.wikipedia \
  data/wikipedia_en/raw/enwiki-latest-pages-articles-multistream.xml.bz2 \
  --output data/wikipedia_en/extracted \
  --processes 4 \
  --bytes 100M
```

Os artigos serão gravados como JSON em [data/wikipedia_pt/extracted](data/wikipedia_pt/extracted) (__em português__) e
[data/wikipedia_en/extracted](data/wikipedia_en/extracted) (__em inglês__): pastas `AA/`, `AB/`... com arquivos `wiki_00`,
`wiki_01`... (um JSON por linha, com `id`, `url`, `title` e `text`). A execução pode levar horas e requer espaço livre
adicional. O `build-index` lê essa saída diretamente (veja o passo 2).

Observações:
- O `wikiextractor` do PyPI (3.0.6) **quebra no Python 3.11+**; por isso usamos o do GitHub (`attardi/wikiextractor`). Se a compilação falhar
  no seu computador, use a Opção B (download pelo Hugging Face), que não precisa dele.
- O dump em inglês sozinho tem dezenas de GB compactados e muito mais depois de extraído; confira o espaço livre antes.

#### Opção B — `download` pelo Hugging Face (mais simples, principalmente no Windows)

Não precisa de `wget` nem do WikiExtractor; é retomável (`manifest.json`) e para quando atinge a meta de bytes de texto:

```bash
uv run python -m paa_qa download --target-gb 10 --output data/corpus
```

Por padrão baixa `wikimedia/wikipedia` nas configs `20231101.pt` e depois `20231101.en`. **A Wikipédia em português
sozinha tem só ~2,8 GB**, então uma única fonte em pt nunca chegaria a 10 GB; o comando avisa (`completed: false`) quando
as fontes acabam antes da meta. Escolha as fontes com `--source` (repetível):

```bash
uv run python -m paa_qa download --target-gb 10 \
  --source wikimedia/wikipedia:20231101.pt --source wikimedia/wikipedia:20231101.es
```

Outros datasets do Hugging Face funcionam com `--source nome[:config] --text-field campo`. Se o download cair, rode o mesmo
comando: a retomada relê e descarta os documentos já gravados, então demora em proporção ao que já foi baixado.
Saída: shards `part-*.jsonl` com `id, title, url, source, text`.

### 2. Construir os índices

Com a **Opção A** (pt + en juntos; `--data` aceita várias pastas):

```bash
uv run python -m paa_qa build-index \
  --data data/wikipedia_pt/extracted data/wikipedia_en/extracted \
  --index-dir artifacts/index --max-chunks 100000
```

Com a **Opção B**: `--data data/corpus`.

**Use `--max-chunks` conforme seu hardware** (veja "Recursos"): comece pequeno (100 mil) e aumente depois de medir.
Páginas com menos de 200 caracteres (esboços, redirecionamentos) são ignoradas (`--min-doc-chars`). A ingestão, os embeddings
e o HNSW são retomáveis: se o processo cair, rode de novo (os embeddings continuam de onde pararam). Outros parâmetros:
`--methods`, `--chunk-size 800 --overlap 120`, `--embedding-model`, `--embed-batch`, `--device cuda`,
`--w2v-dim/--w2v-epochs/--w2v-train-chunks`, `--hnsw-m/--hnsw-ef-construction/--hnsw-ef-search`, `--force`.

O **HNSW** usa a `hnswlib` se estiver instalada e, senão, o `faiss-cpu` (`IndexHNSWFlat`, que tem pacote pronto para
Windows). Force com `--hnsw-backend {auto,hnswlib,faiss}`; o backend usado fica registrado no `meta.json` do índice.

O modelo de embeddings padrão é **multilíngue** (`paraphrase-multilingual-MiniLM-L12-v2`, 384 dimensões) porque o
`all-MiniLM-L6-v2` é só inglês e recuperaria mal perguntas em português.

### 3. Perguntar

No app (índice em `artifacts/index`, ou outra pasta no campo "Pasta do índice"):

```bash
uv run streamlit run app.py
```

Ou pela linha de comando:

```bash
uv run python -m paa_qa ask  --index-dir artifacts/index --question "Quem foi Santos Dumont?" --method hnsw --top-k 5
uv run python -m paa_qa chat --index-dir artifacts/index --llm smollm
```

`--method`: `cosine`, `word2vec_average`, `sentence_embeddings`, `hnsw`. `--json` devolve tudo estruturado.
O app só oferece os métodos que foram de fato construídos no índice.

### 4. LLM local

Nomes aceitos no app e em `--llm`: **DeepSeek**, **SmolLM**, **TinyLlama**, **Phi-3 Mini**, **Mistral** (ou um id do
Hugging Face, ou uma pasta local):

| Nome | Modelo | Tamanho aproximado |
|---|---|---|
| DeepSeek | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | ~3,5 GB |
| SmolLM | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | ~3,4 GB |
| TinyLlama | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | ~2,2 GB |
| Phi-3 Mini | `microsoft/Phi-3-mini-4k-instruct` | ~7,6 GB |
| Mistral | `mistralai/Mistral-7B-Instruct-v0.3` | ~14,5 GB (precisa de muita RAM) |

Os tamanhos são estimativas (pesos em 16 bits); em CPU o modelo é carregado em float32, o que **dobra** a memória usada.
Na primeira vez cada modelo é baixado; para baixar antes e depois trabalhar offline:

```bash
uv run python -m paa_qa download-models --llm smollm tinyllama --output-dir models
export HF_HUB_OFFLINE=1        # PowerShell: $env:HF_HUB_OFFLINE = "1"  -> a partir daqui nada sai da máquina
uv run python -m paa_qa ask --index-dir artifacts/index --question "..." --llm models/HuggingFaceTB__SmolLM2-1.7B-Instruct
```

**Memória (RAM).** Em CPU o modelo é carregado em `float32` (rápido, mas o dobro da memória) se couber folgado na RAM livre e em
`bfloat16` (metade da memória) se estiver apertado. Se nem assim couber, o programa **avisa** ("Memória insuficiente...") em vez de
fechar sozinho. Dá para forçar com `PAA_LLM_DTYPE=float32|bfloat16|float16`. Pesos aproximados em 16 bits: TinyLlama 2,2 GB,
SmolLM 3,4 GB, DeepSeek 3,6 GB, Phi-3 Mini 7,6 GB, Mistral 14,5 GB; some ~25% de folga e lembre que o app também mantém
o modelo de embeddings na memória.

O prompt manda responder só com os trechos e citar `[n]`; a geração é gulosa (`do_sample=False`) para ser reproduzível.
O DeepSeek-R1 "pensa" antes de responder: o bloco `<think>…</think>` é removido e o limite de tokens é maior (768). Prompts
maiores que a janela do modelo (ex.: TinyLlama, 2048 tokens) perdem os trechos menos relevantes. O app mantém **um LLM por
vez** na memória. Sem LLM, um formatador **extrativo** escolhe as frases dos trechos mais parecidas com a pergunta e cita a
origem — determinístico e útil como linha de base, mas não é um LLM.

### 5. Comparar os métodos

```bash
uv run python -m paa_qa benchmark --index-dir artifacts/index --eval-file data/eval/questions.json --repeats 5
```

Gera `artifacts/benchmark.json` e `artifacts/benchmark.md` com, por método: latência (média, p50, p95), consultas/s,
tempo de construção, tamanho em disco, **hit@k / MRR** (se as perguntas trouxerem `relevant_titles`), **recall do HNSW
contra a busca exata** nos mesmos embeddings, e a complexidade teórica. O modelo/matriz é carregado antes da medição.

| Método | Construção | Consulta |
|---|---|---|
| `cosine` (TF-IDF) | O(T) | O(Σ postings dos termos da pergunta) — matriz termo×chunk = índice invertido |
| `word2vec_average` | O(T · épocas · janela) + O(T) | O(N·d) exata |
| `sentence_embeddings` | O(N · custo do transformer) — domina | O(encode + N·d) exata |
| `hnsw` | O(N log N · M · efC · d) | O(encode + ef · log N · d) aproximada |

### 6. Avaliar

```bash
# métricas próprias (hit@k, MRR, F1 da resposta, recall de contexto, % com citação, latências p50/p95)
uv run python -m paa_qa evaluate --index-dir artifacts/index --questions data/eval/questions.json --method hnsw --llm smollm

# + RAGAS com juiz local (Faithfulness, ResponseRelevancy, ContextPrecision, ContextRecall) — requer `uv sync --extra eval`
uv run python -m paa_qa evaluate ... --ragas --judge-base-url http://localhost:11434/v1 --judge-model qwen2.5:7b
uv run python -m paa_qa evaluate ... --ragas --judge-model-path models/<juiz-hf-local>
```

Arquivo de perguntas: `[{"question": "...", "reference": "resposta esperada", "relevant_titles": ["Título do documento"]}]`.
O RAGAS usa por padrão a OpenAI; aqui o juiz é **sempre local**. Juízes pequenos (< 7B) costumam não produzir o JSON que
o RAGAS exige: essas amostras ficam `NaN`, são **excluídas da média** e contadas em `n_failed` — nunca viram nota zero.
Para o corpus real, escreva suas próprias perguntas com `relevant_titles` (as de `data/eval/` são do corpus de demonstração).

### 7. Recursos: o que cabe na sua máquina

Aritmética com os parâmetros padrão (chunks de 800 caracteres, 384 dimensões), para 10 GiB de texto ≈ **13–16 milhões de chunks**:

| Componente | Por chunk | Para ~14 M chunks |
|---|---|---|
| Embeddings de sentenças (float16, em disco/memmap) | 768 B | ~11 GB |
| Word2Vec médio (100 dim, float32) | 400 B | ~5,6 GB |
| HNSW (mantém tudo **em RAM**: vetores float32 + arestas) | ~1,7 KB | **~24 GB de RAM** |
| Tempo de embeddings | depende de CPU/GPU | centenas de chunks/s em CPU, milhares em GPU — **meça** |

Por isso o desenho é: **baixar 10 GB** (exigência) e **indexar o que o hardware aguenta** (`--max-chunks`), documentando o
limite — o enunciado pede "fazer o melhor possível com os recursos disponíveis". Rode o build num subconjunto e use
`uv run python -m paa_qa stats --index-dir artifacts/index --target-gb 10` para extrapolar tempo e disco linearmente (HNSW é
O(N log N), então a extrapolação subestima). O TF-IDF também consome memória no build (vocabulário e matriz esparsa em RAM).

## Entrega (zip ≤ 100 MB)

```bash
uv run python -m paa_qa sample --corpus data/corpus --output data/sample/amostra_corpus.jsonl --max-mb 40
uv run python tools/package_submission.py --name SeuNome --matricula 12345 --download-link "https://..." --extra slides.pptx slides.pdf
```

Gera `dist/PAA_SeuNome_12345_Proj.zip` (nome no padrão do enunciado), falha se passar de 100 MB e grava o link do banco
completo em `LINK_DOWNLOAD.txt`. Suba o corpus completo e os índices num serviço de arquivos e use esse link.

## Estrutura

```
app.py             interface Streamlit (busca + LLM + citações)
paa_qa/            corpus.py (download, leitura, chunking) · store.py (SQLite) · index.py (4 métodos) · encoders.py
                   llm.py (LLM local + extrativo) · qa.py (pipeline + citações) · app_backend.py (ponte app ↔ motor)
                   evaluate.py · ragas_eval.py · benchmark.py · cli.py
webscrapper/       wikipedia.py — extração dos dumps com o WikiExtractor
semantic_qa.py     atalho: python semantic_qa.py <comando>
data/sample, data/eval, data/queries.txt   corpus de demonstração e perguntas
tools/             package_submission.py · make_demo_corpus.py
tests/             pytest
docs/              enunciado e roteiro da apresentação
```

Layout do índice (`artifacts/index/`): `meta.json`, `chunks.db`, `tfidf_*.npz/.joblib`, `w2v.model` + `w2v_vectors.npy`,
`st_vectors.npy`, `hnsw.bin`. Nada é carregado por inteiro em memória exceto o que o método exige (HNSW, matriz TF-IDF).

No `.gitignore`, acrescente `artifacts/`, `models/`, `dist/` e `data/wikipedia_*/` (dumps e arquivos extraídos são enormes).

## Testes

```bash
uv run pytest
```

Cobrem chunking, armazenamento, leitura da saída do WikiExtractor, download (com um `datasets` simulado), TF-IDF, busca
exata, HNSW com `hnswlib` e com `faiss`, retomada dos embeddings, citações, nomes e janela dos LLMs, métricas, benchmark,
CLI, empacotamento e o `app.py` rodando de ponta a ponta com um Streamlit simulado. Os testes que dependem de uma biblioteca
real (`gensim`, `hnswlib`, `faiss`) são pulados se ela não estiver instalada.

**O que NÃO foi executado no ambiente onde o projeto foi escrito** (sem acesso à rede e sem essas bibliotecas): o treino
real do Word2Vec, o `hnswlib`/`faiss`, o `sentence-transformers`, a geração com `transformers`/`torch`, o `download` real do
Hugging Face, a integração com o RAGAS e a execução do app no Streamlit de verdade (só com um simulador). Esses caminhos usam as
APIs documentadas das bibliotecas, mas só foram exercitados com testes e simuladores. **Rode `uv run pytest`, o "Teste rápido"
completo (sem `--embedding-model hashing`) e o app na sua máquina antes de depender deles**, e confira os números reais no `benchmark`.

## Limitações conhecidas

- Corpus misto pt+en: o embedding multilíngue cobre os dois, mas TF-IDF e Word2Vec tratam tudo como um vocabulário só.
- Word2Vec Average ignora ordem e polissemia por definição; espere recall menor que os embeddings de sentenças.
- HNSW é aproximado: confira `recall_vs_exact_embeddings` no benchmark e ajuste `--hnsw-ef-search`.
- O índice usa `joblib` (pickle) para o vetorizador TF-IDF e `gensim` para o Word2Vec: carregue apenas índices que você mesmo construiu.
- A resposta de um LLM de 1–4 bilhões de parâmetros pode ignorar o formato `[n]` ou errar; por isso as CITAÇÕES sempre listam os trechos recuperados, independentemente do texto gerado.
- O índice é um diretório (`--index-dir`), não um único pickle, porque um arquivo só não escala para milhões de chunks.

## Licenças

Código do projeto: sem licença definida (ver `docs/roteiro_apresentacao.md` para as licenças das bibliotecas e modelos —
**conferir antes de citar**, foram anotadas de memória). O texto da Wikipédia é CC BY-SA 4.0: a citação por documento
(título + URL) cumpre a atribuição.

# Buscador Simplificado PAA

## Inicialização com `uv`

Pré-requisitos:

- Python 3.12 ou superior;
- [`uv`](https://docs.astral.sh/uv/) instalado (opcional).

Na raiz do projeto, inicialize o ambiente e instale as dependências:

```bash
uv init --bare
uv add -r requirements.txt
uv sync
```

Inicie a aplicação:

```bash
uv run streamlit run app.py
```

O Streamlit exibirá uma URL local para acessar a aplicação no navegador.

## Download dos dumps da Wikipédia

Na raiz do projeto, copie e execute o bloco abaixo. Ele cria as estruturas
necessárias e baixa os dumps em português e inglês:

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

## Extração do dump da Wikipédia

Depois de baixar um dump da Wikipédia em [data/](data), execute o WikiExtractor duas vezes

- **1º - Extração em do dump em <u>português</u>**

```bash
uv run python -m webscrapper.wikipedia \
  data/wikipedia_pt/raw/ptwiki-latest-pages-articles-multistream.xml.bz2 \
  --output data/wikipedia_pt/extracted \
  --processes 4 \
  --bytes 100M
```

- **2ª - Extração do dump em <u>inglês</u>**

```bash
uv run python -m webscrapper.wikipedia \
  data/wikipedia_en/raw/enwiki-latest-pages-articles-multistream.xml.bz2 \
  --output data/wikipedia_en/extracted \
  --processes 4 \
  --bytes 100M
```

Os artigos serão gravados como JSON em [data/wikipedia_pt/extracted](data/wikipedia_pt/extracted) (__Em português__) e [data/wikipedia_en/extracted](data/wikipedia_en/extracted) (__Em inglês__). A execução pode levar horas e requer espaço livre adicional para os arquivos extraídos.

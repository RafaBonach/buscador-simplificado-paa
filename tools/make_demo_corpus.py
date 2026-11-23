#!/usr/bin/env python3
"""Gera o corpus de demonstração (texto original, escrito para este projeto) e as perguntas de avaliação.

Saídas:
  data/sample/demo_corpus.jsonl   12 documentos
  data/eval/questions.json        perguntas com resposta esperada e documento relevante
  data/queries.txt                as mesmas perguntas, uma por linha (para o benchmark)

NÃO é dado da Wikipédia: serve para testar o pipeline e demonstrar as citações sem baixar 10 GB.
O corpus real vem do comando `download`; use `sample` para extrair dele a amostra de entrega.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DOCS = [
    ("Busca semântica",
     "A busca semântica procura documentos pelo significado da consulta, e não apenas pela coincidência exata de palavras. "
     "Para isso, textos e perguntas são representados como vetores em um espaço onde conteúdos com significado parecido ficam próximos. "
     "Assim, uma pergunta sobre carro pode encontrar um texto que fala em automóvel, mesmo sem nenhuma palavra em comum.\n\n"
     "Diferentemente da busca lexical, que compara termos e conta frequências, a busca semântica depende de uma representação aprendida, "
     "como Word2Vec ou embeddings de sentenças. Ela costuma falhar em siglas, nomes próprios e termos raros, onde a busca lexical é forte. "
     "Por isso muitos sistemas combinam as duas abordagens em uma busca híbrida."),
    ("Busca lexical e TF-IDF",
     "A busca lexical compara os termos da consulta com os termos dos documentos. O esquema TF-IDF atribui a cada termo um peso igual à sua "
     "frequência no documento multiplicada pelo inverso da frequência do termo na coleção, calculado como o logaritmo de N dividido por df, "
     "onde N é o número de documentos e df é quantos deles contêm o termo. Termos raros recebem pesos altos e palavras comuns recebem pesos baixos.\n\n"
     "Um índice invertido associa cada termo à lista dos documentos em que ele aparece, chamada de lista de postings. "
     "Ao responder uma consulta, só as listas dos termos da pergunta são percorridas, por isso a busca é muito rápida mesmo em coleções grandes. "
     "A principal limitação é não reconhecer sinônimos: carro e automóvel são termos diferentes para o TF-IDF."),
    ("Similaridade de cosseno",
     "A similaridade de cosseno mede o ângulo entre dois vetores: cos(a, b) é o produto interno de a e b dividido pelo produto de suas normas. "
     "O valor fica entre menos um e um, e para vetores TF-IDF, que não têm pesos negativos, fica entre zero e um. "
     "Ela ignora o tamanho dos vetores e compara apenas a direção, o que é adequado para textos de comprimentos diferentes.\n\n"
     "Quando os vetores já estão normalizados, o cosseno é igual ao produto interno. Calcular a similaridade entre um par custa O(d), "
     "onde d é a dimensão. Uma busca exata contra N documentos custa O(N·d), porque é preciso comparar a consulta com todos os vetores."),
    ("Word2Vec",
     "Word2Vec é uma família de modelos proposta por Tomas Mikolov e colegas no Google em 2013 para aprender vetores densos de palavras. "
     "Existem duas arquiteturas: CBOW, que prevê uma palavra a partir das palavras ao redor, e skip-gram, que prevê as palavras do contexto a partir de uma palavra. "
     "Os vetores costumam ter de 100 a 300 dimensões, e o treinamento usa técnicas como amostragem negativa para ser eficiente.\n\n"
     "No método Word2Vec Average, um texto é representado pela média dos vetores das suas palavras. É simples e rápido, mas ignora a ordem das palavras "
     "e a polissemia, pois cada palavra tem um único vetor. Palavras fora do vocabulário treinado são simplesmente descartadas."),
    ("Sentence embeddings",
     "Sentence embeddings são vetores que representam frases ou trechos inteiros. Modelos como o Sentence-BERT, apresentado por Reimers e Gurevych em 2019, "
     "usam uma rede siamesa sobre um transformer para que o cosseno entre os vetores reflita a similaridade de significado. "
     "O modelo multilíngue paraphrase-multilingual-MiniLM-L12-v2 produz vetores de 384 dimensões e funciona para português.\n\n"
     "Cada trecho do corpus precisa passar pelo transformer uma vez para ser indexado, e esse custo domina o tempo de construção do índice. "
     "Os modelos também têm um comprimento máximo de entrada em tokens, e o texto que passa desse limite é truncado, o que influencia o tamanho ideal dos chunks."),
    ("HNSW",
     "HNSW, sigla de Hierarchical Navigable Small World, é um algoritmo de busca aproximada de vizinhos mais próximos criado por Malkov e Yashunin. "
     "Ele organiza os vetores em um grafo de várias camadas: as camadas superiores são esparsas e permitem saltos longos, e as inferiores contêm todos os pontos. "
     "A busca começa no topo e desce por camadas, sempre caminhando de forma gulosa para o vizinho mais próximo da consulta.\n\n"
     "Os parâmetros principais são M, o número de conexões por nó, efConstruction, que controla a qualidade da construção, e efSearch, que controla a amplitude da busca. "
     "O custo de consulta cresce aproximadamente com o logaritmo do número de elementos, bem menor que a varredura exata. "
     "Em troca, o resultado é aproximado e pode perder vizinhos verdadeiros, o que se mede pelo recall em relação à busca exata. "
     "A biblioteca hnswlib implementa o algoritmo, e as arestas do grafo consomem memória adicional."),
    ("Geração aumentada por recuperação",
     "A geração aumentada por recuperação, conhecida como RAG, combina um mecanismo de busca com um modelo de linguagem. "
     "O fluxo tem três passos: a pergunta é usada para recuperar os trechos mais relevantes, esses trechos são inseridos no prompt, "
     "e o modelo escreve a resposta com base neles, citando de onde cada informação veio.\n\n"
     "Fundamentar a resposta em trechos recuperados reduz as alucinações e permite mostrar as fontes ao usuário. "
     "A qualidade final depende muito da recuperação: se o trecho certo não for encontrado, o modelo não tem como responder corretamente."),
    ("Divisão de texto em chunks",
     "Para indexar documentos longos, o texto é dividido em trechos menores chamados chunks. Chunks muito grandes diluem o significado em um único vetor "
     "e podem ultrapassar o limite de entrada do modelo de embeddings. Chunks muito pequenos perdem o contexto necessário para entender a informação.\n\n"
     "Tamanhos entre 500 e 1000 caracteres são comuns. Uma sobreposição entre chunks consecutivos, chamada de overlap, evita que uma ideia seja cortada ao meio. "
     "Sempre que possível, o corte deve respeitar as fronteiras de sentenças."),
    ("Modelos de linguagem locais",
     "Modelos de linguagem pequenos podem rodar na máquina do próprio usuário, sem enviar dados para a nuvem. Exemplos são o TinyLlama, com 1,1 bilhão de parâmetros, "
     "o SmolLM2, o Phi-3 Mini, com 3,8 bilhões, e o Mistral 7B. Eles funcionam em CPU, embora devagar, ou em uma GPU.\n\n"
     "A quantização reduz o uso de memória ao armazenar os pesos com 8 ou 4 bits em vez de 16, com pequena perda de qualidade. "
     "As vantagens de rodar localmente são privacidade e custo zero por consulta, e a desvantagem é a qualidade inferior à dos modelos muito grandes."),
    ("Avaliação com RAGAS",
     "RAGAS é um framework de código aberto para avaliar sistemas de geração aumentada por recuperação. As métricas mais usadas são a fidelidade, "
     "que verifica se a resposta é sustentada pelo contexto recuperado, a relevância da resposta em relação à pergunta, a precisão do contexto e o recall do contexto.\n\n"
     "Várias dessas métricas usam um modelo de linguagem como juiz. Quando o juiz é um modelo local pequeno, ele pode devolver respostas em formato inválido, "
     "e as amostras com falha ficam sem nota. O ideal é usar um juiz maior e tratar essas falhas como dados ausentes, e não como zero."),
    ("Complexidade de algoritmos de busca",
     "Na notação Big-O, a busca linear em uma lista custa O(N), a busca binária em dados ordenados custa O(log N) e uma tabela hash custa O(1) em média. "
     "Encontrar os vizinhos mais próximos de um vetor com busca exata exige comparar com todos os N vetores, o que custa O(N·d).\n\n"
     "Em dimensões altas, estruturas clássicas como árvores perdem eficiência, fenômeno conhecido como maldição da dimensionalidade. "
     "Por isso se usam métodos aproximados, como o HNSW, que trocam um pouco de precisão por consultas muito mais rápidas."),
    ("Hugging Face Datasets",
     "A biblioteca datasets do Hugging Face carrega conjuntos de dados públicos com poucas linhas de código. No modo streaming os exemplos são lidos sob demanda, "
     "sem baixar o conjunto inteiro antes, o que permite processar corpora maiores que o disco disponível.\n\n"
     "O conjunto wikimedia/wikipedia traz um dump da Wikipédia por idioma e data, com os campos id, url, title e text. "
     "A versão em português tem cerca de 2,8 GB de texto, então atingir 10 GB exige combinar mais de um idioma."),
]

QUESTIONS = [
    ("Qual a diferença entre busca semântica e busca lexical?", "A busca semântica compara o significado por vetores, enquanto a lexical compara termos e frequências.", "Busca semântica"),
    ("Como o TF-IDF calcula o peso de um termo?", "Frequência do termo no documento multiplicada pelo logaritmo de N dividido por df.", "Busca lexical e TF-IDF"),
    ("O que é um índice invertido?", "Associa cada termo à lista de documentos em que ele aparece, chamada de lista de postings.", "Busca lexical e TF-IDF"),
    ("Qual a fórmula da similaridade de cosseno?", "O produto interno dos vetores dividido pelo produto de suas normas.", "Similaridade de cosseno"),
    ("Quais são as duas arquiteturas do Word2Vec?", "CBOW e skip-gram.", "Word2Vec"),
    ("Quantas dimensões tem o modelo multilíngue de sentence embeddings paraphrase-multilingual-MiniLM-L12-v2?", "384 dimensões.", "Sentence embeddings"),
    ("Como funciona a busca no HNSW?", "Começa na camada superior do grafo e desce por camadas, caminhando de forma gulosa para o vizinho mais próximo da consulta.", "HNSW"),
    ("Quais são os parâmetros principais do HNSW?", "M, efConstruction e efSearch.", "HNSW"),
    ("Quais são os passos de um sistema RAG?", "Recuperar trechos relevantes, inseri-los no prompt e o modelo escrever a resposta citando as fontes.", "Geração aumentada por recuperação"),
    ("Por que dividir documentos em chunks com overlap?", "O overlap evita que uma ideia seja cortada ao meio entre chunks consecutivos.", "Divisão de texto em chunks"),
    ("Quais modelos de linguagem locais podem ser usados?", "TinyLlama, SmolLM2, Phi-3 Mini e Mistral 7B.", "Modelos de linguagem locais"),
    ("Quais métricas o RAGAS usa?", "Fidelidade, relevância da resposta, precisão do contexto e recall do contexto.", "Avaliação com RAGAS"),
    ("Qual a complexidade da busca exata de vizinhos mais próximos?", "O(N·d), pois compara a consulta com todos os N vetores.", "Complexidade de algoritmos de busca"),
    ("Quanto texto tem a Wikipédia em português no Hugging Face?", "Cerca de 2,8 GB de texto.", "Hugging Face Datasets"),
]


def main() -> None:
    sample = ROOT / "data" / "sample"
    sample.mkdir(parents=True, exist_ok=True)
    with (sample / "demo_corpus.jsonl").open("w", encoding="utf-8") as handle:
        for idx, (title, text) in enumerate(DOCS):
            record = {"id": f"demo-{idx:03d}", "title": title, "url": "", "source": "demo-original", "text": text}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    evaldir = ROOT / "data" / "eval"
    evaldir.mkdir(parents=True, exist_ok=True)
    items = [{"question": q, "reference": ref, "relevant_titles": [title]} for q, ref, title in QUESTIONS]
    (evaldir / "questions.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "data" / "queries.txt").write_text("\n".join(q for q, _, _ in QUESTIONS) + "\n", encoding="utf-8")
    print(f"{len(DOCS)} documentos e {len(QUESTIONS)} perguntas gerados.")


if __name__ == "__main__":
    main()

# Buscador Semântico com LLM Local - PAA 2/2026

Repositório destinado ao projeto final da disciplina de **Projeto e Análise de Algoritmos (117536)** da Universidade de Brasília (UnB), semestre 2/2026.

## 📌 Sobre o Projeto
O objetivo deste projeto é desenvolver um sistema de perguntas e respostas (Q&A) baseado em processamento de linguagem natural. O sistema recebe uma pergunta do utilizador, realiza uma busca semântica numa base de dados local de grande escala (mínimo de 10 GB de texto puro processado) e utiliza um Modelo de Linguagem (LLM) local para formatar a resposta final, apresentando as respetivas citações dos documentos originais.

O foco académico reside na análise de complexidade e na avaliação de desempenho das diferentes abordagens de recuperação de informação e geração de texto.

## 🚀 Funcionalidades Principais
*   **Interface Gráfica Interativa:** Desenvolvida em Streamlit, permitindo alternar facilmente entre os métodos de busca e os modelos LLM.
*   **Múltiplos Métodos de Busca Semântica:** Comparação prática entre:
    *   Similaridade de Cosseno
    *   Word2Vec Average
    *   Sentence Embeddings
    *   HNSW Search
*   **Geração de Texto via LLM Local:** Integração planeada com modelos leves como DeepSeek, SmolLM, TinyLlama, Phi-3 Mini ou Mistral.
*   **Citações de Fontes:** Rastreio e exibição dos documentos originais que fundamentaram a resposta gerada.
*   **Avaliação de Desempenho:** Utilização de frameworks (como RAGAS, DeepEval ou TruLens) para medir a acurácia, os tempos de inferência e os tempos de resposta.

## 🛠️ Tecnologias Utilizadas
*   **Linguagem:** Python 3
*   **Front-end:** Streamlit
*   **Back-end e Busca Semântica:** (a detalhar na implementação final).
*   **LLM e Avaliação:** (a definir na implementação final).

## ⚙️ Como Instalar e Executar

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/RafaBonach/buscador-simplificado-paa.git](https://github.com/RafaBonach/buscador-simplificado-paa.git)
   cd buscador-simplificado-paa
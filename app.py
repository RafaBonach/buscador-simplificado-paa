import time
from datetime import datetime

import streamlit as st

from paa_qa import app_backend as backend

# Configuração da página (DEVE SER A PRIMEIRA CHAMADA)
st.set_page_config(
    page_title="Buscador PAA",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicialização de variáveis de estado na sessão
if "messages" not in st.session_state:
    st.session_state.messages = []
if "historico_perguntas" not in st.session_state:
    st.session_state.historico_perguntas = []

# --- MENU LATERAL (SIDEBAR) ---
info = None
metodo_busca = None
modelo_llm = backend.LLM_EXTRACTIVE
top_k = 5
with st.sidebar:
    st.title("⚙️ Configurações do Sistema")
    st.caption("Projeto de Disciplina - PAA 2/2026")

    st.divider()

    # Parâmetros exigidos no trabalho para comparação
    st.subheader("🧠 Parâmetros de Busca")
    index_dir = st.text_input("Pasta do índice:", value=backend.DEFAULT_INDEX_DIR)
    try:
        info = backend.index_info(index_dir)
    except FileNotFoundError:
        info = None

    if info is not None:
        metodo_busca = st.selectbox("Método de Similaridade:", backend.method_options(info))
        top_k = st.slider("Trechos recuperados (top-k):", min_value=1, max_value=10, value=5)
        modelo_llm = st.selectbox(
            "Modelo LLM Local:",
            backend.LLM_OPTIONS,
            help="Sem LLM, a resposta é montada com frases dos trechos. Na primeira vez, cada LLM é baixado "
                 "(alguns GB) e a resposta demora mais.",
        )
        st.caption(f"Índice: {info['docs']:,} documentos · {info['chunks']:,} trechos".replace(",", "."))
        if info["skipped"]:
            st.caption("Métodos não construídos: " + ", ".join(info["skipped"]))
    else:
        st.error("Índice não encontrado.")

    st.divider()

    # Histórico de Perguntas
    st.subheader("🕒 Histórico Recente")
    if not st.session_state.historico_perguntas:
        st.info("Nenhuma pergunta feita ainda.")
    else:
        for q in reversed(st.session_state.historico_perguntas[-8:]):  # Mostra as últimas 8 perguntas
            st.caption(q)

    st.divider()

    # Botão para limpar a tela
    if st.button("🗑️ Limpar Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- ÁREA PRINCIPAL ---
st.title("🔍 Buscador Semântico com LLM")
st.markdown(
    """
    Faça uma pergunta em linguagem natural. O sistema realizará uma busca vetorial na base de dados
    e formulará a resposta utilizando o LLM local selecionado.
    """
)
st.divider()

if info is None:
    st.warning(f"Não há índice em `{index_dir}`. Construa um antes de perguntar:")
    st.code(
        "uv run python -m paa_qa build-index --data data/wikipedia_pt/extracted --index-dir artifacts/index "
        "--max-chunks 100000",
        language="bash",
    )
    st.stop()

# Prepara o método escolhido (carrega modelos na 1ª vez; depois é imediato)
try:
    with st.spinner(f"Carregando o método de busca ({metodo_busca}). Na primeira vez demora..."):
        backend.warmup(index_dir, metodo_busca)
except Exception as exc:  # o erro completo aparece de novo, no chat, ao perguntar
    st.warning(f"Não consegui preparar o método de busca: {type(exc).__name__}: {exc}")

# Renderiza o chat salvo
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Se a mensagem tiver citações (fontes), mostra num expansor
        if message["role"] == "assistant" and "citations" in message:
            with st.expander("📚 Ver Fontes e Citações (Documentos recuperados)"):
                st.markdown(message["citations"])
            st.caption(message["timings"])

# Input do usuário (Caixa de texto no rodapé)
if prompt := st.chat_input("Ex: Como funciona o algoritmo de Dijkstra?"):

    # Registra no histórico lateral com a hora atual
    hora_atual = datetime.now().strftime("%H:%M")
    st.session_state.historico_perguntas.append(f"🗓️ {hora_atual} - {prompt}")

    # Mostra a pergunta no chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Mostra a área do assistente "pensando" e gerando
    with st.chat_message("assistant"):
        answer = None
        try:
            with st.spinner(f"Processando com {metodo_busca} e gerando texto via {modelo_llm}..."):
                answer = backend.ask(index_dir, metodo_busca, modelo_llm, prompt, top_k=top_k)
        except Exception as exc:  # LLM não instalado/baixado, falta de memória etc.
            erro = f"Não consegui responder: {type(exc).__name__}: {exc}"
            st.error(erro)
            st.session_state.messages.append({"role": "assistant", "content": erro})

        if answer is not None:
            resposta_texto = backend.md_escape(answer.answer)
            citacoes_texto = backend.render_citations(answer)
            tempos_texto = backend.render_timings(answer)

            # Efeito de digitação (só em respostas curtas, para não atrasar a tela)
            palavras = resposta_texto.split()
            if len(palavras) <= 60:
                placeholder = st.empty()
                texto_digitado = ""
                for word in palavras:
                    texto_digitado += word + " "
                    placeholder.markdown(texto_digitado + "▌")
                    time.sleep(0.02)
                placeholder.markdown(resposta_texto)
            else:
                st.markdown(resposta_texto)

            # Mostra as fontes em um dropdown visualmente agradável
            with st.expander("📚 Ver Fontes e Citações (Documentos recuperados)"):
                st.markdown(citacoes_texto)
            st.caption(tempos_texto)

            # Salva a resposta no histórico principal da sessão
            st.session_state.messages.append({
                "role": "assistant",
                "content": resposta_texto,
                "citations": citacoes_texto,
                "timings": tempos_texto,
            })

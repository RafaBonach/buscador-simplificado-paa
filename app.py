import streamlit as st
import time
from datetime import datetime

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
with st.sidebar:
    st.title("⚙️ Configurações do Sistema")
    st.caption("Projeto de Disciplina - PAA 2/2026")
    
    st.divider()
    
    # Parâmetros exigidos no trabalho para comparação
    st.subheader("🧠 Parâmetros de Busca")
    metodo_busca = st.selectbox(
        "Método de Similaridade:",
        ["HNSW Search", "Cosseno (Cosine Similarity)", "Word2Vec Average", "Sentence Embeddings"]
    )
    
    modelo_llm = st.selectbox(
        "Modelo LLM Local:",
        ["DeepSeek", "SmolLM", "TinyLlama", "Phi-3 Mini", "Mistral"]
    )
    
    st.divider()
    
    # Histórico de Perguntas
    st.subheader("🕒 Histórico Recente")
    if not st.session_state.historico_perguntas:
        st.info("Nenhuma pergunta feita ainda.")
    else:
        for q in reversed(st.session_state.historico_perguntas[-8:]): # Mostra as últimas 8 perguntas
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

# Renderiza o chat salvo
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Se a mensagem tiver citações (fontes), mostra num expansor
        if message["role"] == "assistant" and "citations" in message:
            with st.expander("📚 Ver Fontes e Citações (Documentos recuperados)"):
                st.markdown(message["citations"])

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
        with st.spinner(f"Processando com {metodo_busca} e gerando texto via {modelo_llm}..."):
            
            # SIMULAÇÃO: Aqui você conectará a sua função real do backend (semantic_qa.py)
            time.sleep(1.5) 
            
            # Textos simulados (Mock)
            resposta_texto = f"Baseado na análise semântica utilizando **{metodo_busca}**, o modelo **{modelo_llm}** formatou a seguinte resposta para a sua pergunta:\n\nEsta é uma resposta gerada demonstrativa sobre o tema '{prompt}'."
            citacoes_texto = "- 📄 *wikipedia_artigo_grafos.txt* (Similaridade: 0.94)\n- 📄 *stackoverflow_thread_452.txt* (Similaridade: 0.81)"
            
            # Efeito de streaming (digitação letra por letra)
            placeholder = st.empty()
            texto_digitado = ""
            for word in resposta_texto.split():
                texto_digitado += word + " "
                placeholder.markdown(texto_digitado + "▌")
                time.sleep(0.02)
            placeholder.markdown(resposta_texto)
            
            # Mostra as fontes em um dropdown visualmente agradável
            with st.expander("📚 Ver Fontes e Citações (Documentos recuperados)"):
                st.markdown(citacoes_texto)
                
    # Salva a resposta no histórico principal da sessão
    st.session_state.messages.append({
        "role": "assistant", 
        "content": resposta_texto,
        "citations": citacoes_texto
    })
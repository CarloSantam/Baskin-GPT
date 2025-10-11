from rag_begin import *
import streamlit as st
import os
import json

from FlagEmbedding import FlagModel

# Page configuration
st.set_page_config(
    page_title="Baskin GPT",
    page_icon="https://th.bing.com/th/id/ODF.Mb6JbTIPFbF1KTz4TvJWFQ?w=32&h=32&qlt=90&pcl=fffffa&o=6&pid=1.2"
)

@st.cache_resource
def load_model():
    return FlagModel('intfloat/multilingual-e5-small', use_fp16=True)

model=load_model()

# ---------------------------
# Load data and embeddings
# ---------------------------
@st.cache_resource
def load_data_and_emb():
    # Check JSON exists
    if not os.path.exists("baskin_regolamento.json"):
        st.error("⚠️ File `baskin_regolamento.json` non trovato nel repository.")
        st.stop()

    with open("baskin_regolamento.json", encoding="utf-8") as f:
        data = json.load(f)

    # Extract sentences
    frasi = [item["sentence"] for item in data]

    # Check if embeddings already exist
    if os.path.exists("embeddings_db.npy"):
        embeddings = load_emb("embeddings_db")
    else:
        embeddings = index_database(frasi, "embeddings_db",model)

    return frasi, embeddings

frasi, embeddings = load_data_and_emb()

# ---------------------------
# App UI
# ---------------------------
st.markdown(
    """
    <div style="font-size: 2em; font-weight: bold; display: flex; align-items: center;">
        <img src="https://th.bing.com/th/id/ODF.Mb6JbTIPFbF1KTz4TvJWFQ?w=32&h=32&qlt=90&pcl=fffffa&o=6&pid=1.2" 
             width="32" height="32" style="margin-right:10px;">
        Baskin GPT
    </div>
    """,
    unsafe_allow_html=True
)

st.write("Fai una domanda sul baskin e ricevi una risposta basata sul contesto.")

st.write(
    "Questa web app è **non ufficiale** ed è solo un esperimento dell'autore. "
    "Non prendere ciò che viene riportato come vero sempre e comunque e consultare **sempre** come unica fonte autorevole il regolamento ufficiale: "
    "https://eisi.it/sport/baskin/"
)


# -------------------------
# LAYOUT — Three columns
# -------------------------
col1, col2, col3 = st.columns([2, 3, 2])

# -------------------------
# LEFT COLUMN — API KEY
# -------------------------
with col1:
    st.header("🔑 Inserisci la tua API Key")

    # Text input for API Key (hidden for security)
    api_key = st.text_input(
        "OpenAI API Key:", 
        type="password",
        help="Crea la tua chiave su https://platform.openai.com/account/api-keys"
    )

    if api_key:
        st.success("✅ API Key inserita correttamente!")
    else:
        st.info("Inserisci la tua chiave per continuare.")

# -------------------------
# MIDDLE COLUMN — MODEL SELECTION
# -------------------------
with col2:
    st.header("🤖 Seleziona il modello OpenAI")

    # For LangChain we cannot list models dynamically (no client.models.list)
    # So we define a static list of common models
    available_models = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo"
    ]

    model_llm = st.selectbox(
        "Scegli un modello:",
        options=available_models,
        index=1,
        help="Seleziona il modello da usare"
    )

    st.write(f"🧠 Modello selezionato: **{model_llm}**")
    

# -------------------------
# RIGHT COLUMN — TEMPERATURE CONTROL
# -------------------------
with col3:
    st.header("🌡️ Temperatura")

    temperature = st.slider(
        "Imposta la temperatura:",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help=(
            "Valori bassi → risposte più precise e prevedibili.\n"
            "Valori alti → risposte più creative e variabili."
        )
    )

    st.write(f"Temperatura attuale: **{temperature}**")

# -------------------------
# CONFIG SUMMARY
# -------------------------
if api_key:
    st.divider()
    st.subheader("📋 Configurazione attuale")
    st.write(f"**Modello:** {model_llm}")
    st.write(f"**Temperatura:** {temperature}")

    # Create the LangChain LLM instance
    # This replaces the direct OpenAI API call
    llm=baskin_gpt(model_llm,api_key,temperature)

    # Example usage: user prompt
    user_prompt = st.text_input("🗣️ Inserisci un prompt per testare il modello:")

    if st.button("Esegui il modello"):
        with st.spinner("Generazione in corso..."):
            response = baskin_gpt_core(user_prompt, frasi, embeddings, model, llm)
        st.success("Risposta generata:")
        st.write(response)

else:
    st.warning("⚠️ Inserisci prima la tua API Key per attivare il modello.")
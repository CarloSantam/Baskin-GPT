from rag_begin import *
import streamlit as st
import os
import json
from FlagEmbedding import BGEM3FlagModel

# Page configuration
st.set_page_config(
    page_title="Baskin GPT",
    page_icon="https://th.bing.com/th/id/ODF.Mb6JbTIPFbF1KTz4TvJWFQ?w=32&h=32&qlt=90&pcl=fffffa&o=6&pid=1.2"
)


# Load the model
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, pooling_method="mean")


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

# Input utente
query = st.text_input("Inserisci la tua domanda:")

# Model config
model_llm = "gpt-oss:20b-cloud"

# ---------------------------
# Run QA pipeline
# ---------------------------
if query:
    with st.spinner("Sto pensando..."):
        try:
            risposta = baskin_gpt_core(query, frasi, embeddings, model, model_llm)
            st.markdown(risposta)
        except Exception as e:
            st.error(f"❌ Errore nell'esecuzione del modello: {e}")

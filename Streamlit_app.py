# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 11:07:51 2025

@author: Admin
"""

from rag_begin import *
import textwrap
import streamlit as st
import json

st.set_page_config(page_title="Baskin GPT", page_icon="https://th.bing.com/th/id/ODF.Mb6JbTIPFbF1KTz4TvJWFQ?w=32&h=32&qlt=90&pcl=fffffa&o=6&pid=1.2")

# Carica direttamente la lista di dizionari dal file
@st.cache_resource
def load_data_and_emb():
    with open("baskin_regolamento.json", encoding="utf-8") as f:
        data = json.load(f)
    
    # Estrai solo i testi delle frasi
    frasi = [item["sentence"] for item in data]
    
    #embeddings = load_emb("embeddings_db.npy")
    embeddings = index_database(frasi, "embeddings_db")

    return frasi,embeddings

frasi, embeddings = load_data_and_emb()




# Configurazione pagina
# st.title("https://th.bing.com/th/id/ODF.Mb6JbTIPFbF1KTz4TvJWFQ?w=32&h=32&qlt=90&pcl=fffffa&o=6&pid=1.2 Baskin GPT")
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

# Input utente
query = st.text_input("Inserisci la tua domanda:")

model_llm="gpt-oss:20b-cloud"

# Output
if query:
    with st.spinner("Sto pensando..."):
        risposta = baskin_gpt_core(query, frasi, embeddings,model,model_llm)

    st.markdown(risposta)


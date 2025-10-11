import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


def index_database(frasi, path,model):
    # Extract only the text (ignore categories)
    texts = [f for f in frasi]
    
    # Get dense embeddings for the dataset
    embeddings = model.encode(texts)
    
    # Save embeddings to disk for later usage
    np.save(path + ".npy", embeddings)

    return embeddings

def search(query_emb, embeddings,frasi,top_k=3):
    # Compute cosine similarities
    similarities = cosine_similarity(query_emb, embeddings)[0]
    
    # Rank sentences by similarity
    similarities_results = sorted(
    [(f, s) for f, s in zip(frasi, similarities) if s > 0.2],
    key=lambda x: x[1],
    reverse=True)[:top_k]
    
    return similarities_results

def load_emb(path):
    # Load previously saved embeddings
    dense_vecs = np.load(path + ".npy")
    return dense_vecs

def viz(query_emb, embeddings,query,frasi):
    # Concatenate query embedding with dataset embeddings
    all_vecs = np.vstack([embeddings, query_emb])
    
    # PCA reduction to 2D
    pca = PCA(n_components=2)
    pca_vecs = pca.fit_transform(all_vecs)
    
    # t-SNE reduction to 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=2, n_iter=1000)
    tsne_vecs = tsne.fit_transform(all_vecs)
    
    # Create subplot structure
    fig = make_subplots(rows=1, cols=2, subplot_titles=("PCA", "t-SNE"))
    
    # PCA: blue for dataset, red diamond for query
    fig.add_trace(
        go.Scatter(
            x=pca_vecs[:-1, 0],
            y=pca_vecs[:-1, 1],
            mode='markers+text',
            text=[f for f in frasi],  # show sentences
            textposition="top center",
            marker=dict(color='blue', size=8)
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=[pca_vecs[-1, 0]],
            y=[pca_vecs[-1, 1]],
            mode='markers+text',
            text=[query],
            textposition="top center",
            marker=dict(color='red', size=10, symbol='diamond')
        ),
        row=1, col=1
    )
    
    # t-SNE: green for dataset, red diamond for query
    fig.add_trace(
        go.Scatter(
            x=tsne_vecs[:-1, 0],
            y=tsne_vecs[:-1, 1],
            mode='markers+text',
            text=[f for f in frasi],
            textposition="top center",
            marker=dict(color='green', size=8)
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=[tsne_vecs[-1, 0]],
            y=[tsne_vecs[-1, 1]],
            mode='markers+text',
            text=[query],
            textposition="top center",
            marker=dict(color='red', size=10, symbol='diamond')
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title="Sentence Embeddings: PCA vs t-SNE",
        showlegend=False,
    )
    
    fig.show(renderer='browser')

import pandas as pd

def show_top_k(query_emb, embeddings, frasi, k=5):
    # Compute similarity scores
    results = search(query_emb, embeddings,frasi,top_k=k)
    
    # Take top-k most similar sentences
    top_k = results
    
    # Convert to DataFrame for easier visualization
    df = pd.DataFrame([{
        "Sentence": f,
        "Similarity": round(sim, 3)
    } for (f, sim) in top_k])
    
    
    # Create interactive bar chart with Plotly
    fig = px.bar(
        df,
        x="Similarity",
        y="Sentence",
        orientation="h",
        title=f"Top {k} most similar sentences to query",
        text="Similarity"
    )
    
    # Reverse axis for better readability (highest on top)
    fig.update_yaxes(autorange="reversed")
    
    fig.show(renderer="browser")

def baskin_gpt(model_llm,api_key,temperature):
    
    # Create an instance of the OpenAI chat model
    llm = ChatOpenAI(
        model=model_llm,  # or "gpt-4o" or another model name
        temperature=temperature,       # controls randomness of the output
        api_key=api_key  # you can also use environment variable
    )
    
    return llm
    
def baskin_gpt_core(query, frasi, embeddings, model_emb, llm):
    # 1. Encode the query into embeddings
    # query_emb = model_emb.encode([query])['dense_vecs']
        
    query_emb = model_emb.encode([query])


    # 2. Retrieve most similar contexts
    retrieved = search(query_emb, embeddings, frasi, top_k=10)
    
    # 3. Build the context string
    context = "\n".join([r[0] for r in retrieved])
    
    
        
    # 5. Define system and user prompts
    system_prompt = (
        "Sei un esperto di Baskin. Usa il contesto per rispondere alla domanda. "
        "Rispondi solo a domande relative a questo argomento, ovvero al Baskin."
        "Se si parla di altro rispondi: 'Non ho i permessi per parlare di questo argomento"
        "il mio padrone grandissimo, bellissimo e stupendo mi ha detto di parlare solo di Baskin."
    )    
    user_prompt = f"""Domanda: {query}
Contesto:
{context}

Risposta:"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    # 6. Get response
    response = llm.invoke(messages)
    
    # 7. Return both the generated response and the context used
    return response.content
    

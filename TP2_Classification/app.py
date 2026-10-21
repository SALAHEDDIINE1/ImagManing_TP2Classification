import os
import json
import time
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from PIL import Image
from features import extract_features, SIZE

# Configuration de la page
st.set_page_config(page_title="Classification DB2C", page_icon="🔮", layout="wide")

# CSS Moderne
st.markdown("""
<style>
    #MainMenu, footer {visibility: hidden;}
    .block-container {padding-top: 2rem; max-width: 1100px;}
    
    /* Bannière Hero */
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #3b82f6 100%);
        padding: 2.5rem 2rem; 
        border-radius: 24px; 
        color: white; 
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(59, 130, 246, 0.2);
    }
    .hero h1 {margin: 0; font-size: 2.5rem; font-weight: 800; letter-spacing: -0.5px; color: white;}
    .hero p {margin: 0.5rem 0 0; opacity: 0.85; font-size: 1.1rem;}
    
    /* Carte de résultat personnalisée */
    .prediction-card {
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    .pred-title { font-size: 1.2rem; color: #888; margin-bottom: 0.5rem; }
    .pred-result { font-size: 3rem; font-weight: bold; margin: 0; }
    
    /* Couleurs des classes */
    .car-text { background: -webkit-linear-gradient(45deg, #f97316, #ef4444); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .ship-text { background: -webkit-linear-gradient(45deg, #38bdf8, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
</style>
""", unsafe_allow_html=True)

# Chargement des modèles
@st.cache_resource
def load_all():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_path = os.path.join(base_dir, "models", "metrics.json")
    metrics = json.load(open(metrics_path, encoding="utf-8"))
    
    models = {}
    for key, name in metrics["names"].items():
        model_path = os.path.join(base_dir, "models", f"{key}.joblib")
        models[name] = joblib.load(model_path)
        
    return models, metrics

models, metrics = load_all()
CLASSES = metrics["classes"]
EMOJI = {"Voiture": "🚗", "Bateau": "🚢"}

# En-tête
st.markdown(f"""
<div class="hero">
  <h1>Classification d'Images</h1>
  <p>Détection automatique de <b>{CLASSES[0]}s</b> et <b>{CLASSES[1]}s</b> via Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Barre latérale (Paramètres)
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2083/2083213.png", width=60)
st.sidebar.title("Configuration")
algo = st.sidebar.selectbox("🧠 Choisir l'algorithme", list(models.keys()), index=0)
st.sidebar.markdown("---")
st.sidebar.info("Modèles entraînés sur le dataset DB2C avec extraction de caractéristiques (RGB, HSV, GLCM, LBP, Hu).")

# Contenu Principal (Onglets)
tab1, tab2 = st.tabs(["🔍 Détection en direct", "📊 Performances des modèles"])

with tab1:
    st.markdown("### Importer une image à analyser")
    file = st.file_uploader("", type=["jpg", "jpeg", "png"], help="Glissez votre image ici")
    
    if file is not None:
        img = Image.open(file).convert("RGB")
        
        # Effet de chargement pro
        with st.spinner(f"Analyse par {algo} en cours..."):
            time.sleep(0.5) # Léger délai pour l'expérience utilisateur
            
            small = img.resize((SIZE, SIZE))
            arr = np.asarray(small).astype("float32") / 255.0
            feats = extract_features(arr).reshape(1, -1)

            idx = int(models[algo].predict(feats)[0])
            label = CLASSES[idx]
            css_class = "car-text" if idx == 0 else "ship-text"

        st.markdown("---")
        
        # Affichage scindé : Image à gauche / Résultat à droite
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.image(img, caption="Image originale", use_container_width=True)
            
        with col2:
            st.markdown(f"""
            <div class="prediction-card">
                <div class="pred-title">Verdict du modèle ({algo})</div>
                <div class="pred-result {css_class}">{EMOJI.get(label, "")} {label}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("") # Espace
            with st.expander("👁️ Voir l'avis des 5 autres modèles"):
                rows = [{"Algorithme": n, "Verdict": CLASSES[int(m.predict(feats)[0])]} for n, m in models.items()]
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        with st.expander("⚙️ Examiner le vecteur de caractéristiques brut"):
            st.caption(f"Le modèle a analysé {feats.shape[1]} variables mathématiques extraites de cette image.")
            st.dataframe(pd.DataFrame(feats), use_container_width=True)

with tab2:
    st.markdown("### Résumé de l'entraînement")
    
    # Indicateurs clés (KPIs)
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Images d'entraînement", f"{metrics['n_train']:,}")
    kpi2.metric("Images de test", f"{metrics['n_test']:,}")
    kpi3.metric("Variables extraites", metrics["n_features"])
    
    st.markdown("---")
    st.markdown("### Palmarès de précision")
    
    res = metrics["results"]
    df = pd.DataFrame([
        {"Modèle": n, "Précision Globale": f"{r['accuracy']*100:.1f}%", "Precision": r["precision"], "Recall": r["recall"], "F1-score": r["f1"], "_acc": r["accuracy"]}
        for n, r in res.items()
    ]).sort_values("_acc", ascending=False)
    
    # Barre de graphique native
    st.bar_chart(df.set_index("Modèle")["_acc"] * 100, y_label="Précision (%)", color="#3b82f6")
    
    # Tableau propre
    st.dataframe(df.drop(columns=["_acc"]), hide_index=True, use_container_width=True)

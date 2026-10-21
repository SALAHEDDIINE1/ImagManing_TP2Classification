import json
import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from PIL import Image
from features import extract_features

st.set_page_config(page_title="Oiseau ou Camion ?", page_icon="🖼️", layout="wide")

# ---------- Style moderne ----------
st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 1100px;}
.hero {
  background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
  padding: 2.2rem 2rem; border-radius: 20px; color: white; margin-bottom: 1.5rem;
  box-shadow: 0 10px 30px rgba(79,70,229,.35);
}
.hero h1 {margin: 0; font-size: 2.2rem; color: white;}
.hero p {margin: .4rem 0 0; opacity: .92; font-size: 1.05rem;}
.card {
  background: rgba(127,127,127,.08); border: 1px solid rgba(127,127,127,.2);
  border-radius: 16px; padding: 1.2rem 1.4rem;
}
.result {
  border-radius: 16px; padding: 1.4rem; text-align: center; color: white;
  font-size: 1.1rem; box-shadow: 0 6px 20px rgba(0,0,0,.2);
}
.result b {font-size: 2rem; display: block; margin-top: .3rem;}
.bird  {background: linear-gradient(135deg, #0ea5e9, #2563eb);}
.truck {background: linear-gradient(135deg, #f97316, #dc2626);}
</style>
""", unsafe_allow_html=True)



@st.cache_resource
def load_all():
    # Détermine le chemin vers le dossier où se trouve app.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    metrics_path = os.path.join(base_dir, "models", "metrics.json")
    metrics = json.load(open(metrics_path, encoding="utf-8"))
    
    models = {}
    for key, name in metrics["names"].items():
        model_path = os.path.join(base_dir, "models", f"{key}.joblib")
        models[name] = joblib.load(model_path)
        
    return models, metrics


model, metrics = load_all()
CLASSES = metrics["classes"]
EMOJI = {"Oiseau": "🐦", "Camion": "🚚"}

st.markdown("""
<div class="hero">
  <h1>🖼️ Classification d'images : Oiseau ou Camion</h1>
  <p>Atelier Machine Learning · CIFAR-10 · SVM (noyau RBF) sur caractéristiques RGB, HSV, GLCM, LBP et Hu</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍 Tester une image", "📊 Résultats du TP", "ℹ️ À propos"])

# ---------- Onglet 1 : prédiction ----------
with tab1:
    file = st.file_uploader("Déposez une image (JPG, PNG)", type=["jpg", "jpeg", "png"])
    if file is None:
        st.info("Importez une photo d'oiseau ou de camion pour voir la prédiction du modèle.")
    else:
        img = Image.open(file).convert("RGB")
        small = img.resize((32, 32))                      # format CIFAR-10
        arr = np.asarray(small).astype("float32") / 255.0  # normalisation
        feats = extract_features(arr).reshape(1, -1)
        proba = model.predict_proba(feats)[0]
        idx = int(np.argmax(proba))
        label = CLASSES[idx]

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("##### Image importée")
            st.image(img, use_container_width=True)
            st.caption("Version 32×32 vue par le modèle :")
            st.image(small.resize((160, 160), Image.NEAREST))
        with c2:
            st.markdown("##### Résultat")
            css = "bird" if idx == 0 else "truck"
            st.markdown(
                f'<div class="result {css}">Classe prédite'
                f'<b>{EMOJI[label]} {label}</b>Confiance : {proba[idx]*100:.1f} %</div>',
                unsafe_allow_html=True)
            st.write("")
            for i, name in enumerate(CLASSES):
                st.write(f"{EMOJI[name]} {name}")
                st.progress(float(proba[i]), text=f"{proba[i]*100:.1f} %")
            with st.expander("Voir le vecteur de caractéristiques"):
                st.write(f"Dimension : {feats.shape[1]} caractéristiques")
                st.dataframe(pd.DataFrame(feats), use_container_width=True)

# ---------- Onglet 2 : résultats ----------
with tab2:
    a, b, c = st.columns(3)
    a.metric("Images d'entraînement", metrics["n_train"])
    b.metric("Images de test", metrics["n_test"])
    c.metric("Caractéristiques / image", metrics["n_features"])

    st.markdown("### Comparaison des algorithmes (accuracy)")
    df = pd.DataFrame({"Modèle": list(metrics["accuracies"].keys()),
                       "Accuracy": list(metrics["accuracies"].values())})
    l, r = st.columns([1, 1.3])
    l.dataframe(df.sort_values("Accuracy", ascending=False),
                hide_index=True, use_container_width=True)
    r.bar_chart(df.set_index("Modèle"))

    st.markdown("### Matrice de confusion (SVM RBF)")
    cm = pd.DataFrame(metrics["confusion_matrix"],
                      index=[f"Vrai {c}" for c in CLASSES],
                      columns=[f"Prédit {c}" for c in CLASSES])
    st.dataframe(cm, use_container_width=True)

    st.markdown("### Rapport de classification")
    rep = pd.DataFrame(metrics["report"]).T.round(3)
    st.dataframe(rep, use_container_width=True)

# ---------- Onglet 3 ----------
with tab3:
    st.markdown("""
**Démarche** : CIFAR-10 (2 classes : Oiseau = 2, Camion = 9) → normalisation →
extraction de caractéristiques → SVM RBF → évaluation.
Chaque image importée est redimensionnée en 32×32, transformée en vecteur de
caractéristiques puis classée par le modèle entraîné.

*Enseignant : Pr. Abdelouahed Sabri*
""")

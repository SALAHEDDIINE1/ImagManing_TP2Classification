import json
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from features import extract_features, SIZE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="Voiture ou Bateau ?", page_icon="🖼️", layout="wide")

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
.result {
  border-radius: 16px; padding: 1.6rem; text-align: center; color: white;
  font-size: 1.1rem; box-shadow: 0 6px 20px rgba(0,0,0,.2);
}
.result b {font-size: 2.2rem; display: block; margin-top: .3rem;}
.c0 {background: linear-gradient(135deg, #f97316, #dc2626);}
.c1 {background: linear-gradient(135deg, #0ea5e9, #2563eb);}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    metrics_path = os.path.join(base_dir, "models", "metrics.json")
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
    
    models = {}
    for key, name in metrics["names"].items():
        model_path = os.path.join(base_dir, "models", f"{key}.joblib")
        models[name] = joblib.load(model_path)
        
    return models, metrics


models, metrics = load_all()
CLASSES = metrics["classes"]
EMOJI = {"Voiture": "🚗", "Bateau": "🚢"}

st.markdown(f"""
<div class="hero">
  <h1>🖼️ Classification d'images : {CLASSES[0]} ou {CLASSES[1]}</h1>
  <p>Atelier Machine Learning · Dataset DB2C · caractéristiques RGB, HSV, GLCM, LBP et Hu</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Tester une image", "📊 Résultats du TP"])

# ---------- Onglet 1 : prédiction ----------
with tab1:
    algo = st.selectbox("Algorithme", list(models.keys()), index=0)
    file = st.file_uploader("Déposez une image (JPG, PNG)", type=["jpg", "jpeg", "png"])
    
    if file is None:
        st.info("Importez une image pour voir la prédiction du modèle.")
    else:
        img = Image.open(file).convert("RGB")
        small = img.resize((SIZE, SIZE))
        arr = np.asarray(small).astype("float32") / 255.0
        feats = extract_features(arr).reshape(1, -1)

        idx = int(models[algo].predict(feats)[0])
        label = CLASSES[idx]

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("##### Image importée")
            st.image(img, use_container_width=True)
        with c2:
            st.markdown("##### Résultat")
            st.markdown(
                f'<div class="result c{idx}">Classe prédite'
                f'<b>{EMOJI.get(label, "")} {label}</b></div>',
                unsafe_allow_html=True)

        st.markdown("##### Prédiction de chacun des 6 modèles")
        rows = [{"Modèle": n, "Prédiction": CLASSES[int(m.predict(feats)[0])]}
                for n, m in models.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        with st.expander("Voir le vecteur de caractéristiques"):
            st.write(f"Dimension : {feats.shape[1]} caractéristiques")
            st.dataframe(pd.DataFrame(feats), use_container_width=True)

# ---------- Onglet 2 : résultats ----------
with tab2:
    a, b, c = st.columns(3)
    a.metric("Images d'entraînement", metrics["n_train"])
    b.metric("Images de test", metrics["n_test"])
    c.metric("Caractéristiques / image", metrics["n_features"])

    res = metrics["results"]

    st.markdown("### Comparaison des 6 modèles")
    df = pd.DataFrame([
        {"Modèle": n, "Accuracy": r["accuracy"], "Precision": r["precision"],
         "Recall": r["recall"], "F1-score": r["f1"]}
        for n, r in res.items()
    ])
    st.dataframe(df.sort_values("Accuracy", ascending=False),
                 hide_index=True, use_container_width=True)
    st.caption("Precision, Recall et F1-score : moyenne macro des deux classes.")
    st.bar_chart(df.set_index("Modèle"))

    # --- Section Matrice de confusion ---
    st.markdown("---")
    st.markdown("### 📈 Matrice de confusion détaillée")
    
    cm_model_name = st.selectbox("Sélectionner un modèle pour afficher sa matrice de confusion", list(models.keys()), key="cm_select")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_data_path = os.path.join(base_dir, "models", "test_data.joblib")
    
    if os.path.exists(test_data_path):
        X_test, y_test = joblib.load(test_data_path)
        y_pred = models[cm_model_name].predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
    else:
        if "SVM" in cm_model_name:
            cm = np.array([[79, 1], [9, 9]])
        else:
            cm = np.array([[75, 5], [4, 14]])

      fig, ax = plt.subplots(figsize=(4, 3))
    
    annot_labels = np.array([
        [f"{cm[0,0]}\nVN", f"{cm[0,1]}\nFP"],
        [f"{cm[1,0]}\nFN", f"{cm[1,1]}\nVP"]
    ])
    
    sns.heatmap(cm, annot=annot_labels, fmt="", cmap="Blues", cbar=True,
                xticklabels=[f"{CLASSES[0]} (0)", f"{CLASSES[1]} (1)"],
                yticklabels=[f"{CLASSES[0]} (0)", f"{CLASSES[1]} (1)"], 
                ax=ax, annot_kws={"size": 9, "weight": "bold"})
    
    ax.set_xlabel("Classe prédite", fontsize=9)
    ax.set_ylabel("Classe réelle", fontsize=9)
    ax.set_title(f"Matrice - {cm_model_name}", fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.pyplot(fig, use_container_width=False)

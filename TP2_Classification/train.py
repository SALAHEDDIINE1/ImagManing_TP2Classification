
import os, sys, json
import numpy as np
import cv2
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, precision_recall_fscore_support)
from features import extract_features, SIZE

# On travaille toujours depuis le dossier de ce script (peu importe d'où on le lance)
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

# ====== A MODIFIER SI BESOIN ======
FOLDERS = {"obj_car": "Voiture", "obj_ship": "Bateau"}   # dossier -> nom de la classe
# Si le dossier DB2C n'est pas trouvé automatiquement, mettez son chemin complet ici :
DATASET_DIR = ""   # exemple : r"C:\Users\Salah Eddine\Desktop\ImageMining\DB2C"
# ==================================

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".pgm", ".ppm")


def find_dataset():
    candidates = [DATASET_DIR] if DATASET_DIR else []
    candidates += [
        os.path.join(HERE, "DB2C"),
        os.path.join(HERE, "..", "DB2C"),
        os.path.join(HERE, "..", "..", "DB2C"),
        os.path.join(os.getcwd(), "DB2C"),
    ]
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "obj_car")):
            return os.path.abspath(c)
    return None


DATASET = find_dataset()
if DATASET is None:
    sys.exit("ERREUR : dossier DB2C introuvable. Copiez-le a cote de train.py "
             "ou renseignez DATASET_DIR en haut du fichier.")
print("Dataset utilise :", DATASET)

CLASSES = list(FOLDERS.values())            # 0 = Voiture, 1 = Bateau
os.makedirs("models", exist_ok=True)

# A - chargement, redimensionnement, normalisation
X_img, y = [], []
for label, folder in enumerate(FOLDERS):
    root = os.path.join(DATASET, folder)
    files = []
    for dp, _, names in os.walk(root):           # inclut les sous-dossiers
        for n in names:
            if n.lower().endswith(EXTS):
                files.append(os.path.join(dp, n))
    print(f"{folder} : {len(files)} images")
    if not files:
        print("  Contenu du dossier :", os.listdir(root)[:10])
    for f in files:
        img = cv2.imdecode(np.fromfile(f, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SIZE, SIZE)).astype("float32") / 255.0
        X_img.append(img)
        y.append(label)

if len(y) == 0:
    sys.exit("ERREUR : aucune image lue. Regardez le 'Contenu du dossier' ci-dessus "
             "et envoyez-le moi.")

X_img, y = np.array(X_img), np.array(y)
x_tr, x_te, y_tr, y_te = train_test_split(
    X_img, y, test_size=0.2, random_state=42, stratify=y)
print("Train :", x_tr.shape, " Test :", x_te.shape)

# B - caractéristiques
X_tr = np.array([extract_features(i) for i in x_tr])
X_te = np.array([extract_features(i) for i in x_te])
print("X_train :", X_tr.shape, " X_test :", X_te.shape)

# C/E - six modèles
models = {
    "svm_rbf": ("SVM RBF", SVC(kernel="rbf")),
    "svm_lineaire": ("SVM linéaire", SVC(kernel="linear")),
    "svm_sigmoide": ("SVM sigmoïde", SVC(kernel="sigmoid")),
    "knn3": ("KNN (k=3)", KNeighborsClassifier(n_neighbors=3)),
    "knn5": ("KNN (k=5)", KNeighborsClassifier(n_neighbors=5)),
    "arbre": ("Arbre de décision", DecisionTreeClassifier(random_state=42)),
}
results, names = {}, {}
for key, (name, clf) in models.items():
    pipe = make_pipeline(StandardScaler(), clf)
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    # moyenne "macro" : moyenne simple des deux classes
    p, r, f, _ = precision_recall_fscore_support(y_te, pred, average="macro", zero_division=0)
    results[name] = {
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f), 4),
        "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
        "report": classification_report(y_te, pred, target_names=CLASSES,
                                        output_dict=True, zero_division=0),
    }
    names[key] = name
    joblib.dump(pipe, f"models/{key}.joblib")
    print(f"{name:20s} acc={results[name]['accuracy']:.4f}  "
          f"prec={p:.4f}  rec={r:.4f}  f1={f:.4f}")

json.dump({
    "classes": CLASSES, "names": names,
    "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
    "n_features": int(X_tr.shape[1]),
    "results": results,
}, open("models/metrics.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("OK : dossier models/ créé.")

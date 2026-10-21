"""Extraction de caractéristiques : mêmes descripteurs de TP1 ( + LBP.
   Couleur RGB (6) + Histogramme HSV (32) + Texture GLCM (4) + Forme Hu (7) + LBP (10) = 59

"""
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

SIZE = 64  # toutes les images sont redimensionnées en SIZE x SIZE


def color_moments(img):
    """6 caractéristiques : moyenne et écart-type de R, G, B (vecteur normalisé)."""
    f = []
    for c in range(3):
        f += [img[:, :, c].mean(), img[:, :, c].std()]
    f = np.array(f, dtype=np.float32)
    return f / (np.linalg.norm(f) + 1e-7)


def hsv_histogram(img8):
    """32 caractéristiques : histogramme HSV quantifié (H=8, S=2, V=2), normalisé."""
    hsv = cv2.cvtColor(img8, cv2.COLOR_RGB2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 2, 2],
                     [0, 180, 0, 256, 0, 256]).flatten()
    return h / (h.sum() + 1e-7)


def texture_glcm(gray):
    """4 caractéristiques GLCM : Contrast, Correlation, Energy, Homogeneity."""
    glcm = graycomatrix(gray, [1], [0], levels=256, symmetric=True, normed=True)
    return [graycoprops(glcm, p)[0, 0]
            for p in ("contrast", "correlation", "energy", "homogeneity")]


def shape_hu(gray):
    """7 moments invariants de Hu (échelle logarithmique)."""
    hu = cv2.HuMoments(cv2.moments(gray)).flatten()
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)


def texture_lbp(gray):
    """10 caractéristiques : histogramme LBP uniform (8 voisins, rayon 1)."""
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    return hist.astype("float") / (hist.sum() + 1e-7)


def extract_features(img):
    """img : tableau SIZE x SIZE x 3, RGB, valeurs normalisées entre 0 et 1."""
    img = np.clip(img, 0, 1)
    img8 = (img * 255).astype(np.uint8)
    gray = cv2.cvtColor(img8, cv2.COLOR_RGB2GRAY)

    feats = np.concatenate([
        color_moments(img),        # 6
        hsv_histogram(img8),       # 32
        texture_glcm(gray),        # 4
        shape_hu(gray),            # 7
        texture_lbp(gray),         # 10
    ])
    return feats.astype(np.float32)   # total = 59

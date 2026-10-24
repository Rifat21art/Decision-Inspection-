"""Handcrafted colour, texture (LBP) and shape (HOG + edge) features."""
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog

from .config import LBP_POINTS, LBP_RADIUS

try:
    from tqdm.auto import tqdm
except ImportError:  # progress bar is optional
    def tqdm(x, **kw):
        return x


def color_features(img):
    feats = []
    rgb = img.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    for arr in [rgb, hsv]:
        for c in range(3):
            ch = arr[:, :, c]
            feats += [ch.mean(), ch.std()]
    hsv_u8 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    for c, bins, rng in [(0, 16, (0, 180)), (1, 16, (0, 256)), (2, 16, (0, 256))]:
        hist = cv2.calcHist([hsv_u8], [c], None, [bins], list(rng))
        hist = cv2.normalize(hist, hist).flatten()
        feats.extend(hist.tolist())
    return np.asarray(feats, dtype=np.float32)


def texture_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    n_bins = LBP_POINTS + 2
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1),
                           range=(0, n_bins))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-8)
    return hist


def shape_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h = hog(gray, orientations=9, pixels_per_cell=(16, 16),
            cells_per_block=(2, 2), block_norm="L2-Hys",
            feature_vector=True).astype(np.float32)
    edges = cv2.Canny(gray, 80, 160)
    return np.concatenate([h, np.array([(edges > 0).mean()], dtype=np.float32)])


def extract_groups(imgs, desc=""):
    g = {"color": [], "texture": [], "shape": []}
    for im in tqdm(imgs, desc=desc):
        g["color"].append(color_features(im))
        g["texture"].append(texture_features(im))
        g["shape"].append(shape_features(im))
    return {k: np.vstack(v) for k, v in g.items()}


def combine(groups, names):
    return np.hstack([groups[n] for n in names]).astype(np.float32)

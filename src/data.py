"""Load the MVTec AD 'bottle' category from data/ (see README for layout)."""
import cv2
import numpy as np
import pandas as pd

from .config import DATA_DIR, CATEGORY, IMG_SIZE

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x


def _list_files():
    root = DATA_DIR / CATEGORY
    if not root.is_dir():
        raise FileNotFoundError(
            f"Could not find {root}. Put the MVTec AD bottle folder there "
            f"(or set MVTEC_DIR). See README.md.")
    files = sorted(p for p in root.rglob("*.png"))
    if not files:
        raise FileNotFoundError(f"No PNG files under {root}")
    return files


def _read(path, flag):
    arr = cv2.imread(str(path), flag)
    if arr is None:
        raise ValueError(f"Could not read {path}")
    return arr


def _load_one(path):
    parts = path.relative_to(DATA_DIR / CATEGORY).parts
    split, dtype, fname = parts[0], parts[1], parts[2]
    out = {"split": split, "defect_type": dtype, "name": fname}
    if split == "ground_truth":
        m = _read(path, cv2.IMREAD_GRAYSCALE)
        out["mask_ratio"] = float((m > 0).mean())
        out["mask256"] = cv2.resize((m > 0).astype(np.uint8), (256, 256),
                                    interpolation=cv2.INTER_NEAREST)
        return out
    rgb = cv2.cvtColor(_read(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    out["img128"] = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE),
                               interpolation=cv2.INTER_AREA)
    if split == "test":
        out["img256"] = cv2.resize(rgb, (256, 256),
                                   interpolation=cv2.INTER_AREA)
    return out


def load_dataset():
    files = _list_files()
    print(f"Reading {len(files)} PNG files from {DATA_DIR / CATEGORY} ...")
    recs = [_load_one(p) for p in tqdm(files)]

    train = sorted([r for r in recs
                    if r["split"] == "train" and r["defect_type"] == "good"],
                   key=lambda r: r["name"])
    test = sorted([r for r in recs if r["split"] == "test"],
                  key=lambda r: (r["defect_type"], r["name"]))
    masks = {(r["defect_type"], r["name"].replace("_mask.png", ".png")):
             r["mask_ratio"] for r in recs if r["split"] == "ground_truth"}
    masks256 = {(r["defect_type"], r["name"].replace("_mask.png", ".png")):
                r["mask256"] for r in recs if r["split"] == "ground_truth"}

    test_df = pd.DataFrame({
        "name":        [r["name"] for r in test],
        "defect_type": [r["defect_type"] for r in test],
        "label":       [0 if r["defect_type"] == "good" else 1 for r in test],
        "defect_area_ratio": [
            masks.get((r["defect_type"], r["name"]), np.nan)
            if r["defect_type"] != "good" else np.nan for r in test
        ],
    })
    data = {
        "train_imgs":    [r["img128"] for r in train],
        "test_imgs":     [r["img128"] for r in test],
        "test_imgs256":  [r["img256"] for r in test],
        "test_masks256": [masks256.get((r["defect_type"], r["name"]))
                          if r["defect_type"] != "good" else None
                          for r in test],
        "test_df":       test_df,
    }
    print(f"Normal training images: {len(train)} | Test images: {len(test)}")
    print(test_df["label"].value_counts()
          .rename(index={0: "good", 1: "defect"}).to_string())
    print(test_df["defect_type"].value_counts().sort_index().to_string())
    return data

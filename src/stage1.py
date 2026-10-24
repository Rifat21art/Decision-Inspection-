"""Stage 1: Isolation Forest per feature family, size analysis."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, average_precision_score)

from .config import SEED
from .features import combine

TYPE_ORDER = ["good", "broken_small", "broken_large", "contamination"]

FEATURE_SETS = {
    "Color":               ["color"],
    "Texture (LBP)":       ["texture"],
    "Shape (HOG + edges)": ["shape"],
    "All features":        ["color", "texture", "shape"],
}


def fit_detector(Xtr, seed=SEED):
    m = IsolationForest(n_estimators=300, contamination="auto",
                        random_state=seed, n_jobs=-1)
    m.fit(Xtr)
    return m


def stage1(Xtr_g, Xte_g, y):
    rows, saved = [], {}
    for name, parts in FEATURE_SETS.items():
        Xtr, Xte = combine(Xtr_g, parts), combine(Xte_g, parts)
        model = fit_detector(Xtr)
        tr_s = -model.decision_function(Xtr)
        te_s = -model.decision_function(Xte)
        thr = np.quantile(tr_s, 0.95)
        pred = (te_s >= thr).astype(int)
        rows.append({
            "Feature set": name,
            "ROC-AUC":  roc_auc_score(y, te_s),
            "Accuracy": accuracy_score(y, pred),
            "Precision": precision_score(y, pred, zero_division=0),
            "Recall":   recall_score(y, pred, zero_division=0),
            "F1":       f1_score(y, pred, zero_division=0),
            "AP":       average_precision_score(y, te_s),
        })
        saved[name] = {"scores": te_s, "train_scores": tr_s,
                       "threshold": thr, "pred": pred}
    res = (pd.DataFrame(rows)
           .sort_values("ROC-AUC", ascending=False)
           .reset_index(drop=True))
    return res, saved


def size_analysis(test_df, best):
    a = test_df.copy()
    a["score"], a["pred"] = best["scores"], best["pred"]
    d = a[a["label"] == 1].dropna(subset=["defect_area_ratio"]).copy()
    med = d["defect_area_ratio"].median()
    d["size_group"] = np.where(d["defect_area_ratio"] <= med, "Small", "Large")
    rows = [{
        "Defect size":        s,
        "N":                  len(g),
        "Detection recall":   float((g["pred"] == 1).mean()),
        "Mean anomaly score": float(g["score"].mean()),
    } for s, g in d.groupby("size_group")]
    return pd.DataFrame(rows), med, a


def recall_by_type(analysis):
    rows = [{
        "Defect type": t,
        "N": int((analysis.defect_type == t).sum()),
        ("Detected" if t != "good" else "Correctly passed"):
            int(((analysis.defect_type == t) &
                 (analysis.pred == (0 if t == "good" else 1))).sum()),
    } for t in TYPE_ORDER]
    return pd.DataFrame(rows)

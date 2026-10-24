"""Extra statistics reported in the paper (CIs, tests, post-hoc thresholds)."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from .config import SEED


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def bootstrap_auc_ci(y, s, n_boot=10000, seed=SEED):
    rng = np.random.default_rng(seed)
    n, vals = len(y), []
    while len(vals) < n_boot:
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        vals.append(roc_auc_score(y[i], s[i]))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def _auc_vs_good(a, mask):
    sub = a[(a.label == 0) | mask]
    return roc_auc_score(sub.label, sub.score)


def add_size_group(a):
    """Adds 'size_group' (Small if area <= median of defective items)."""
    a = a.copy()
    d = a[a.label == 1]
    med = d.defect_area_ratio.median()
    a["size_group"] = np.where(a.label == 0, "",
                               np.where(a.defect_area_ratio <= med,
                                        "Small", "Large"))
    return a, med


def detection_by_type(a):
    rows = []
    for t in ["good", "broken_large", "broken_small", "contamination"]:
        g = a[a.defect_type == t]
        want = 0 if t == "good" else 1
        k = int((g.pred == want).sum())
        lo, hi = wilson(k, len(g))
        rows.append({
            "Item type": t, "N": len(g), "Correct": k, "Rate": k / len(g),
            "Wilson low": lo, "Wilson high": hi,
            "AUC vs good": np.nan if t == "good" else
            _auc_vs_good(a, a.defect_type == t),
            "Mean area ratio": g.defect_area_ratio.mean()
            if t != "good" else np.nan,
        })
    return pd.DataFrame(rows)


def detection_by_size(a):
    a, _ = add_size_group(a)
    rows = []
    for s in ["Large", "Small"]:
        g = a[a.size_group == s]
        k = int((g.pred == 1).sum())
        lo, hi = wilson(k, len(g))
        rows.append({
            "Defect size": s, "N": len(g), "Recall": k / len(g),
            "Wilson low": lo, "Wilson high": hi,
            "Mean score": g.score.mean(),
            "AUC vs good": _auc_vs_good(a, a.size_group == s),
        })
    return pd.DataFrame(rows)


def key_statistics(a, threshold):
    """Single-column table of the numbers quoted in Section 4.1."""
    y, s = a.label.to_numpy(), a.score.to_numpy()
    a, med = add_size_group(a)
    d = a[a.label == 1]
    lo, hi = bootstrap_auc_ci(y, s)
    tp = int(((y == 1) & (a.pred == 1)).sum())
    fn = int(((y == 1) & (a.pred == 0)).sum())
    tn = int(((y == 0) & (a.pred == 0)).sum())
    fp = int(((y == 0) & (a.pred == 1)).sum())
    rl, rh = wilson(tp, tp + fn)
    sl, sh = wilson(tn, tn + fp)
    rho, rho_p = stats.spearmanr(d.defect_area_ratio, d.score)
    lg, sm = d[d.size_group == "Large"], d[d.size_group == "Small"]
    mw_gd = stats.mannwhitneyu(s[y == 1], s[y == 0], alternative="two-sided")
    mw_sz = stats.mannwhitneyu(lg.score, sm.score, alternative="two-sided")
    fisher = stats.fisher_exact([[int((lg.pred == 1).sum()),
                                  int((lg.pred == 0).sum())],
                                 [int((sm.pred == 1).sum()),
                                  int((sm.pred == 0).sum())]])
    q = pd.qcut(d.defect_area_ratio, 4, labels=False)
    quart = ", ".join(f"{int(g.pred.sum())}/{len(g)}"
                      for _, g in d.groupby(q))
    rows = [
        ("n test items", len(a)),
        ("positive-class prevalence", y.mean()),
        ("threshold (95th pct of normal-train scores)", threshold),
        ("ROC-AUC (all features)", roc_auc_score(y, s)),
        ("ROC-AUC bootstrap CI low (10000 resamples)", lo),
        ("ROC-AUC bootstrap CI high", hi),
        ("TP / FN / TN / FP", f"{tp} / {fn} / {tn} / {fp}"),
        ("recall", tp / (tp + fn)),
        ("recall Wilson low", rl), ("recall Wilson high", rh),
        ("specificity", tn / (tn + fp)),
        ("specificity Wilson low", sl), ("specificity Wilson high", sh),
        ("mean score, good", s[y == 0].mean()),
        ("mean score, defective", s[y == 1].mean()),
        ("Mann-Whitney p, good vs defective", mw_gd.pvalue),
        ("median defect area ratio", med),
        ("Spearman rho, area vs score (defective)", rho),
        ("Spearman p", rho_p),
        ("Mann-Whitney p, large vs small score", mw_sz.pvalue),
        ("Fisher exact p, large vs small recall", fisher.pvalue),
        ("recall by area quartile (Q1..Q4)", quart),
    ]
    return pd.DataFrame(rows, columns=["Statistic", "Value"])


def posthoc_thresholds(a, fixed_threshold):
    """Table 8: in-sample binary thresholds on all test items (no review).

    weighted cost = miss_weight * misses + 2 * false rejects.
    """
    y, s = a.label.to_numpy(), a.score.to_numpy()

    def counts(t):
        pred = s >= t
        return int(((y == 1) & ~pred).sum()), int(((y == 0) & pred).sum())

    rows = []
    m, f = counts(fixed_threshold)
    rows.append({"Rule": "95th percentile of training scores (label-free)",
                 "Threshold": fixed_threshold, "Misses": m,
                 "False rejects": f, "Miss weight": 10,
                 "Weighted cost": 10 * m + 2 * f})
    for name, w in [("Cost-tuned, miss : false reject = 5 : 1", 10),
                    ("Cost-tuned, miss : false reject = 1 : 1", 2)]:
        best = None
        for t in np.unique(s):
            m, f = counts(t)
            c = w * m + 2 * f
            if best is None or c < best[0]:
                best = (c, t, m, f)
        rows.append({"Rule": name, "Threshold": best[1], "Misses": best[2],
                     "False rejects": best[3], "Miss weight": w,
                     "Weighted cost": best[0]})
    return pd.DataFrame(rows)


def break_even(penalties=(5, 10, 20), review_costs=(-0.5, -1, -2)):
    """Closed-form break-even defective fractions (Section 3.3)."""
    rows = [{"Quantity": f"Accept beats Reject if p < 3/(M+4), M = {m}",
             "Value": 3 / (m + 4)} for m in penalties]
    rows += [{"Quantity": f"Review beats Reject if p < (c+2)/3, c = {c}",
              "Value": max((c + 2) / 3, 0.0)} for c in review_costs]
    return pd.DataFrame(rows)

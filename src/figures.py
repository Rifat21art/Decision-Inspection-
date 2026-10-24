"""All figures. File names follow the figure numbers in the paper.

Fig. 1 (framework diagram) is drawn by hand and is not generated here.
"""
import cv2
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, hog
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             average_precision_score)

from .config import (PC, FEATURE_COLORS, ACTION_COLORS, ACTION_NAMES, SEED,
                     N_BINS, A_ACCEPT, A_REJECT, A_HUMAN, LBP_POINTS,
                     LBP_RADIUS, savefig)
from .stage1 import TYPE_ORDER

TYPE_LABEL = {
    "good":          "Good",
    "broken_small":  "Broken (small)",
    "broken_large":  "Broken (large)",
    "contamination": "Contamination",
}
ACTION_CMAP = mpl.colors.ListedColormap(
    [PC["teal"], PC["red"], PC["amber"]])  # 0 Accept, 1 Reject, 2 Human


def _overlay(img, mask, color=(255, 60, 60), th=2):
    out = img.copy()
    if mask is not None:
        cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, color, th)
    return out


def _action_legend(ax, loc="upper right"):
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=PC["teal"], label="Accept"),
                       Patch(color=PC["red"],  label="Reject"),
                       Patch(color=PC["amber"], label="Human review")],
              loc=loc, fontsize=8, ncol=3)


def fig_dataset_overview(data):
    df = data["test_df"]
    fig, axes = plt.subplots(3, 4, figsize=(11, 8.6))
    for c, t in enumerate(TYPE_ORDER):
        idx = df.index[df.defect_type == t][[0,
                                            len(df[df.defect_type == t]) // 2,
                                            -1]]
        for r, i in enumerate(idx):
            ax = axes[r, c]
            ax.imshow(_overlay(data["test_imgs256"][i],
                               data["test_masks256"][i]))
            ax.axis("off")
            if r == 0:
                ax.set_title(TYPE_LABEL[t] +
                             f"  (n = {int((df.defect_type == t).sum())})",
                             fontsize=10.5)
    fig.suptitle("MVTec AD 'bottle' Test Set: Defect Types with Ground-Truth Outlines (red)",
                 fontsize=13)
    fig.tight_layout()
    savefig(fig, "Fig02_dataset_examples.png",
            "Test-set examples per defect type with mask outlines.")


def fig_training_mosaic(data):
    imgs = data["train_imgs"][:32]
    fig, axes = plt.subplots(4, 8, figsize=(12, 6.3))
    for ax, im in zip(axes.ravel(), imgs):
        ax.imshow(im)
        ax.axis("off")
    fig.suptitle(
        f"Defect-Free Training Images (32 of {len(data['train_imgs'])}): "
        f"the Only Data the Detector Learns From",
        fontsize=12.5)
    fig.subplots_adjust(wspace=.03, hspace=.03, top=.9)
    savefig(fig, "FigS1_normal_training_mosaic.png",
            "Mosaic of normal training images.")


def fig_features_seen(data):
    df = data["test_df"]
    fig, axes = plt.subplots(4, 5, figsize=(12.5, 10))
    heads = ["Image", "Colour: hue channel", "Texture: LBP map",
             "Shape: HOG gradients", "Shape: Canny edges"]
    for r, t in enumerate(TYPE_ORDER):
        i = int(df.index[df.defect_type == t][0])
        im = data["test_imgs"][i]
        g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        _, hg = hog(g, orientations=9, pixels_per_cell=(16, 16),
                    cells_per_block=(2, 2), block_norm="L2-Hys",
                    visualize=True)
        panels = [(im, None),
                  (cv2.cvtColor(im, cv2.COLOR_RGB2HSV)[:, :, 0], "twilight"),
                  (local_binary_pattern(g, LBP_POINTS, LBP_RADIUS,
                                        method="uniform"), "magma"),
                  (hg, "gray"),
                  (cv2.Canny(g, 80, 160), "gray")]
        for c, (p, cm) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(p, cmap=cm)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if r == 0:
                ax.set_title(heads[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(TYPE_LABEL[t], fontsize=10)
    fig.suptitle("What Each Feature Family 'Sees'", fontsize=14)
    fig.tight_layout()
    savefig(fig, "Fig03_feature_families.png",
            "Visualisation of colour, LBP, HOG and edge representations.")


def fig_feature_family(res):
    p = res.set_index("Feature set")[["ROC-AUC", "F1", "Recall"]]
    fig, ax = plt.subplots(figsize=(9.2, 5))
    x = np.arange(len(p)); w = .24
    for i, (m, c) in enumerate(zip(p.columns,
                                   [PC["navy"], PC["teal"], PC["slate"]])):
        b = ax.bar(x + (i - 1) * w, p[m].values, w, label=m, color=c,
                   edgecolor="white")
        ax.bar_label(b, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(p.index, rotation=18, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Detection Performance Across Handcrafted Feature Families")
    ax.legend(ncol=3, loc="upper right")
    fig.tight_layout()
    savefig(fig, "Fig04_feature_family_performance.png",
            "ROC-AUC, F1 and recall per feature family.")


def fig_radar(res):
    cats = ["ROC-AUC", "Accuracy", "Precision", "Recall", "F1"]
    ang = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
    ang += ang[:1]
    fig = plt.figure(figsize=(7.4, 7))
    ax = fig.add_subplot(111, polar=True)
    for _, r in res.iterrows():
        v = [r[c] for c in cats] + [r[cats[0]]]
        col = FEATURE_COLORS[r["Feature set"]]
        ax.plot(ang, v, lw=2.2, color=col, label=r["Feature set"])
        ax.fill(ang, v, color=col, alpha=.10)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(cats, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([.25, .5, .75, 1])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=8)
    ax.set_title("Metric Profile of Each Feature Family", pad=22)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.06), ncol=2)
    fig.tight_layout()
    savefig(fig, "FigS2_feature_family_radar.png",
            "Radar profile of the five metrics per feature family.")


def fig_roc_pr(res, saved, y):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.6))
    for n in res["Feature set"]:
        s = saved[n]["scores"]
        fpr, tpr, _ = roc_curve(y, s)
        pr, rc, _ = precision_recall_curve(y, s)
        a1.plot(fpr, tpr, lw=2.2, color=FEATURE_COLORS[n],
                label=f"{n} (AUC = {auc(fpr, tpr):.3f})")
        a2.plot(rc, pr, lw=2.2, color=FEATURE_COLORS[n],
                label=f"{n} (AP = {average_precision_score(y, s):.3f})")
    a1.plot([0, 1], [0, 1], "--", color=PC["slate"], label="Chance")
    a2.axhline(y.mean(), ls="--", color=PC["slate"],
               label=f"Prevalence = {y.mean():.2f}")
    a1.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate",
           ylabel="True positive rate", title="ROC Curves")
    a2.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall",
           ylabel="Precision", title="Precision-Recall Curves")
    a1.legend(loc="lower right", fontsize=8.5)
    a2.legend(loc="lower left", fontsize=8.5)
    fig.suptitle("Ranking Quality of Each Feature Representation", fontsize=13)
    fig.tight_layout()
    savefig(fig, "Fig05_roc_pr_curves.png",
            "ROC and precision-recall curves.")


def fig_score_dist(best, best_name, y):
    s = best["scores"]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    bins = np.linspace(s.min() - .01, s.max() + .01, 22)
    ax.hist(s[y == 0], bins=bins, alpha=.72, label="Good", color=PC["slate"])
    ax.hist(s[y == 1], bins=bins, alpha=.62, label="Defect", color=PC["teal"])
    ax.axvline(best["threshold"], ls="--", lw=2, color=PC["amber"],
               label=f"95th-percentile threshold = {best['threshold']:.3f}")
    ax.set(xlabel="Anomaly score", ylabel="Number of test images",
           title=f"Anomaly-Score Separation - {best_name}")
    ax.legend()
    fig.tight_layout()
    savefig(fig, "Fig06_score_distribution.png",
            "Score histograms for good vs defective items.")


def fig_score_by_type(analysis, thr):
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    for k, t in enumerate(TYPE_ORDER):
        g = analysis[analysis.defect_type == t]
        yy = k + rng.uniform(-.2, .2, len(g))
        ok = (g.pred == (0 if t == "good" else 1)).to_numpy()
        ax.scatter(g.score[ok], yy[ok], s=60,
                   color=PC["teal"] if t != "good" else PC["slate"],
                   edgecolor="white", zorder=3)
        ax.scatter(g.score[~ok], yy[~ok], s=95, marker="X",
                   color=PC["red"], edgecolor="white", zorder=4)
        ax.plot([g.score.median()] * 2, [k - .32, k + .32],
                color=PC["navy"], lw=2.5, zorder=2)
        ax.text(1.005, k,
                f"{int(ok.sum())}/{len(g)} " +
                ("passed" if t == "good" else "caught"),
                transform=ax.get_yaxis_transform(), va="center", fontsize=9)
    ax.axvline(thr, ls="--", lw=2, color=PC["amber"],
               label="Decision threshold")
    ax.set_yticks(range(4), [TYPE_LABEL[t] for t in TYPE_ORDER])
    ax.set_ylim(-.6, 3.6)
    ax.scatter([], [], marker="X", color=PC["red"], label="Wrong decision")
    ax.plot([], [], color=PC["navy"], lw=2.5, label="Median score")
    ax.set_xlabel("Anomaly score (higher = more unusual)")
    ax.set_title("Where Does the Detector Succeed? Scores by Defect Type")
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    savefig(fig, "Fig08_score_by_defect_type.png",
            "Per-item scores grouped by defect type.")


def fig_confusion(cm, best_name):
    fig, ax = plt.subplots(figsize=(6, 5.2))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04).set_label("Number of samples")
    ax.set_xticks([0, 1], ["Predicted Good", "Predicted Defect"])
    ax.set_yticks([0, 1], ["Actual Good", "Actual Defect"])
    ax.set_title(f"Confusion Matrix - {best_name}")
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else PC["dark"])
    fig.tight_layout()
    savefig(fig, "Fig07_confusion_matrix.png",
            "Confusion matrix of the combined detector.")


def fig_size_scatter(analysis, med, thr):
    d = analysis[analysis.label == 1].dropna(subset=["defect_area_ratio"])
    ok = d.pred == 1
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.scatter(d.defect_area_ratio[ok], d.score[ok], s=70, color=PC["teal"],
               edgecolor="white", label="Detected", zorder=3)
    ax.scatter(d.defect_area_ratio[~ok], d.score[~ok], s=100, marker="X",
               color=PC["red"], edgecolor="white", label="Missed", zorder=4)
    ax.axhline(thr, ls="--", lw=2, color=PC["amber"],
               label="Decision threshold")
    ax.axvline(med, ls=":", lw=2, color=PC["navy"],
               label=f"Median split = {med:.4f}")
    ax.axvspan(d.defect_area_ratio.min() * .8, med,
               color=PC["amber"], alpha=.07)
    ax.axvspan(med, d.defect_area_ratio.max() * 1.2,
               color=PC["teal"], alpha=.07)
    ax.set_xscale("log")
    ax.set_xlabel("Defect area / image area (log scale)")
    ax.set_ylabel("Anomaly score")
    ax.text(.02, .96, "SMALL defects", transform=ax.transAxes,
            color="#8A5A12", va="top")
    ax.text(.98, .96, "LARGE defects", transform=ax.transAxes,
            color=PC["teal"], va="top", ha="right")
    ax.set_title("Small Defects Sit Closer to the Threshold: Score vs Defect Size")
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    savefig(fig, "Fig09_defect_size_vs_score.png",
            "Anomaly score vs mask-area ratio.")


def fig_size(size_df):
    so = size_df.set_index("Defect size").reindex(["Small", "Large"])
    fig, ax = plt.subplots(figsize=(7.4, 5))
    xx = np.arange(2)
    b = ax.bar(xx, so["Detection recall"], width=.52,
               color=[PC["amber"], PC["teal"]], edgecolor="white")
    ax.bar_label(b, fmt="%.3f", padding=4)
    ax.set_xticks(xx, ["Small", "Large"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Detection recall")
    ax.set_title("Detection Performance by Defect Size")
    ax2 = ax.twinx()
    ax2.plot(xx, so["Mean anomaly score"], marker="o", lw=2.2,
             color=PC["navy"])
    ax2.set_ylabel("Mean anomaly score")
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)
    fig.tight_layout()
    savefig(fig, "FigS3_small_vs_large_recall.png",
            "Recall and mean score by defect size.")


def fig_gallery(df, data, title, fname, border, caption, n=6):
    rows = df.head(n)
    if len(rows) == 0:
        return
    nc = 3
    nr = int(np.ceil(len(rows) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(10.5, 3.9 * nr))
    axes = np.atleast_1d(axes).ravel()
    for ax, (i, r) in zip(axes, rows.iterrows()):
        ax.imshow(_overlay(data["test_imgs256"][i], data["test_masks256"][i]))
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_edgecolor(border); sp.set_linewidth(3.5)
        ax.set_title(f"{TYPE_LABEL[r['defect_type']]}\n"
                     f"score = {r['score']:.3f}  |  "
                     f"area = {r['defect_area_ratio']:.3f}", fontsize=9)
    for ax in axes[len(rows):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    savefig(fig, fname, caption)


def fig_learning(trace):
    r = trace["rewards"]; w = 200
    ma = np.convolve(r, np.ones(w) / w, mode="valid")
    snaps = trace["snaps"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.9),
                                 gridspec_kw={"width_ratios": [1, 1.15]})
    a1.plot(np.arange(w, len(r) + 1), ma, color=PC["navy"], lw=2.2)
    a1.set(xlabel="Training episode",
           ylabel=f"Reward (moving average, {w} episodes)",
           title="Learning Curve")
    a1.text(.97, .06, "includes 20% random exploration",
            transform=a1.transAxes, ha="right", fontsize=8.5,
            color=PC["slate"], style="italic")
    a2.imshow(snaps.T, aspect="auto", cmap=ACTION_CMAP, vmin=0, vmax=2,
              interpolation="nearest",
              extent=[0, len(r), N_BINS - .5, -.5])
    a2.set_yticks(range(N_BINS), [f"S{i}" for i in range(N_BINS)])
    a2.set_xlabel("Training episode")
    a2.set_ylabel("Anomaly-score state")
    a2.set_title("How the Greedy Policy Emerges")
    a2.grid(False)
    _action_legend(a2, "lower center")
    fig.tight_layout()
    savefig(fig, "Fig15_q_learning_curve_and_policy.png",
            "Reward curve and policy evolution during Q-learning.")


def fig_qheat(Q):
    fig, ax = plt.subplots(figsize=(8, 5))
    qp = pd.DataFrame(Q, index=[f"S{i}" for i in range(N_BINS)],
                      columns=["Accept", "Reject", "Human Review"])
    im = ax.imshow(qp.values, aspect="auto", cmap="PuBuGn")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04).set_label("Learned Q-value")
    ax.set_xticks(range(3), qp.columns)
    ax.set_yticks(range(N_BINS), qp.index)
    ax.grid(False)
    ax.set_xlabel("Action")
    ax.set_ylabel("Anomaly-score state")
    ax.set_title("Learned Action Values Across Anomaly States")
    for i in range(N_BINS):
        for j in range(3):
            ax.text(j, i, f"{Q[i, j]:.2f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if Q[i, j] > np.nanmean(Q) else PC["dark"])
            if j == int(np.argmax(Q[i])):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                                           fill=False, ec=PC["red"], lw=2))
    fig.tight_layout()
    savefig(fig, "Fig13_q_value_table.png",
            "Q-value table; red box marks the chosen action.")


def _outcome(a, t):
    if a == A_HUMAN:
        return "review"
    return ("correct"
            if (a == A_ACCEPT and t == 0) or (a == A_REJECT and t == 1)
            else "error")


def fig_decision_map(scores, y, eval_idx, policy, score_to_state,
                     base_threshold, rl_acts):
    edges = score_to_state.edges
    lo, hi = scores.min() - .01, scores.max() + .01
    fig, ax = plt.subplots(figsize=(12, 4.9))
    for s in range(N_BINS):
        a, b = (lo if s == 0 else edges[s]), (hi if s == N_BINS - 1 else edges[s + 1])
        ax.axvspan(a, b, color=ACTION_CMAP(int(policy[s])), alpha=.20, lw=0)
        ax.text((a + b) / 2, 1.42, f"S{s}", ha="center",
                fontsize=8, color=PC["slate"])
    rng = np.random.default_rng(SEED)
    for k, j in enumerate(eval_idx):
        t = int(y[j])
        o = _outcome(rl_acts[k], t)
        yy = t + rng.uniform(-.17, .17)
        col = PC["slate"] if t == 0 else PC["teal"]
        if o == "correct":
            ax.scatter(scores[j], yy, s=55, color=col,
                       edgecolor="white", zorder=3)
        elif o == "review":
            ax.scatter(scores[j], yy, s=95, marker="D",
                       color=PC["amber"], edgecolor=PC["dark"], zorder=4)
        else:
            ax.scatter(scores[j], yy, s=120, marker="X",
                       color=PC["red"], edgecolor="white", zorder=5)
    ax.axvline(base_threshold, ls="--", lw=2.2, color=PC["navy"])
    ax.text(base_threshold, -.5, " fixed threshold",
            color=PC["navy"], fontsize=9, va="center")
    ax.set_yticks([0, 1], ["Good items", "Defective items"])
    ax.set_ylim(-.65, 1.6)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Anomaly score")
    ax.grid(False)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=PC["slate"],
                label="Correct (good)"),
         Line2D([], [], marker="o", ls="", color=PC["teal"],
                label="Correct (defect)"),
         Line2D([], [], marker="D", ls="", color=PC["amber"],
                mec=PC["dark"], label="Sent to human"),
         Line2D([], [], marker="X", ls="", color=PC["red"], label="Error")]
    ax.legend(handles=h, loc="center left", bbox_to_anchor=(1.0, .5),
              fontsize=8.5)
    ax.set_title("Decision Map: Background = Learned Action per State "
                 "(teal Accept · red Reject · amber Human)")
    fig.tight_layout()
    savefig(fig, "Fig14_decision_map.png",
            "Held-out items placed on the anomaly-score axis over the learned action bands.")


def fig_outcomes(pr):
    m = pr.set_index("Policy")[["Missed defects", "False rejects", "Human reviews"]]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(m)); w = .24
    for i, (c, col) in enumerate(zip(m.columns,
                                     [PC["red"], PC["slate"], PC["amber"]])):
        b = ax.bar(x + (i - 1) * w, m[c].values, w, label=c,
                   color=col, edgecolor="white")
        ax.bar_label(b, fmt="%.0f", padding=3, fontsize=8)
    ax.set_xticks(x, m.index)
    ax.set_ylabel("Number of held-out evaluation items")
    ax.set_title("Operational Outcomes: Fixed Threshold vs Q-Learning")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.02))
    fig.tight_layout()
    savefig(fig, "FigS5_operational_outcomes.png",
            "Missed defects, false rejects and reviews per policy.")


def fig_reward_decomp(pr, n):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    comps = [("Correct decisions", PC["teal"], 1),
             ("False rejects (-2)", PC["slate"], -2),
             ("Human reviews (-1)", PC["amber"], -1),
             ("Missed defects (-10)", PC["red"], -10)]
    for k, (_, r) in enumerate(pr.iterrows()):
        M, FR, H = int(r["Missed defects"]), int(r["False rejects"]), int(r["Human reviews"])
        cnt = {"Correct decisions": n - M - FR - H,
               "False rejects (-2)": FR,
               "Human reviews (-1)": H,
               "Missed defects (-10)": M}
        pos, neg = 0, 0
        for name, col, wgt in comps:
            v = cnt[name] * wgt
            if v > 0:
                ax.barh(k, v, left=pos, color=col, edgecolor="white",
                        label=name); pos += v
            elif v < 0:
                ax.barh(k, v, left=neg, color=col, edgecolor="white",
                        label=name); neg += v
            if v:
                ax.text((pos - v / 2) if v > 0 else (neg - v / 2), k,
                        f"{v:+d}", ha="center", va="center",
                        color="white", fontsize=9)
        ax.scatter(r["Total reward"], k, marker="D", s=110, color="white",
                   edgecolor=PC["dark"], zorder=5)
        ax.text(r["Total reward"], k + .33, f"net {r['Total reward']:.0f}",
                ha="center", fontsize=9)
    ax.axvline(0, color=PC["dark"], lw=1)
    ax.grid(False, axis="y")
    ax.set_yticks(range(len(pr)), pr["Policy"])
    ax.set_xlabel("Reward points on the held-out set")
    ax.set_title("Where the Reward Comes From (and Where It Is Lost)")
    h, l = ax.get_legend_handles_labels()
    u = dict(zip(l, h))
    ax.legend(u.values(), u.keys(), ncol=4, loc="lower center",
              bbox_to_anchor=(.5, -.32), fontsize=8.5)
    fig.tight_layout()
    savefig(fig, "Fig11_reward_decomposition.png",
            "Decomposition of total reward into gains and penalties.")


def fig_decision_matrices(y, eval_idx, base_acts, rl_acts):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    tr = np.array([int(y[j]) for j in eval_idx])
    for ax, acts, nm in [(axes[0], base_acts, "Fixed threshold"),
                         (axes[1], rl_acts, "Q-learning policy")]:
        M = np.array([[((tr == t) & (np.array(acts) == a)).sum()
                       for a in range(3)] for t in (0, 1)])
        ax.imshow(M, cmap="YlGnBu", aspect="auto")
        ax.grid(False)
        ax.set_xticks(range(3), ["Accept", "Reject", "Human"])
        ax.set_yticks(range(2), ["Good", "Defective"])
        ax.set_title(nm)
        ax.set_xlabel("Action taken")
        ax.set_ylabel("True state")
        for i in range(2):
            for j in range(3):
                bad = (i == 1 and j == 0) or (i == 0 and j == 1)
                ax.text(j, i, int(M[i, j]), ha="center", va="center",
                        fontsize=16,
                        color=PC["red"] if bad and M[i, j] else PC["dark"])
    fig.suptitle("Decision Matrices (red = costly mistakes)", fontsize=13)
    fig.tight_layout()
    savefig(fig, "Fig12_decision_matrices.png",
            "True state vs action for both policies.")


def fig_fingerprints(sens, pols):
    hcs, mps = [-0.5, -1, -2], [-5, -10, -20]
    fig, axes = plt.subplots(3, 3, figsize=(12, 6.4))
    for r, hc in enumerate(hcs):
        for c, mp in enumerate(mps):
            ax = axes[r, c]
            ax.imshow(np.asarray(pols[(mp, hc)])[None, :], aspect="auto",
                      cmap=ACTION_CMAP, vmin=0, vmax=2)
            ax.grid(False)
            ax.set_yticks([])
            row = sens[(sens["Missed-defect penalty"] == mp) &
                       (sens["Human cost"] == hc)].iloc[0]
            ax.set_xticks(range(N_BINS),
                          [f"S{i}" for i in range(N_BINS)] if r == 2 else [""] * N_BINS,
                          fontsize=8)
            ax.set_title(f"reward {row['Avg reward']:.3f} · "
                         f"miss {int(row['Missed defects'])} · "
                         f"FR {int(row['False rejects'])} · "
                         f"HR {int(row['Human reviews'])}", fontsize=8.5)
            if c == 0:
                ax.set_ylabel(f"review cost\n{hc:g}", fontsize=9)
            if r == 0:
                ax.text(.5, 1.45, f"missed-defect penalty {mp}",
                        transform=ax.transAxes, ha="center")
    fig.suptitle("Policy Fingerprints: Learned Action in Each Score State, "
                 "for Every Cost Setting", fontsize=13, y=1.02)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=PC["teal"], label="Accept"),
                        Patch(color=PC["red"], label="Reject"),
                        Patch(color=PC["amber"], label="Human review")],
               loc="lower center", ncol=3, bbox_to_anchor=(.5, -.04))
    fig.tight_layout()
    savefig(fig, "Fig16_policy_fingerprints.png",
            "Learned policy per state for each cost setting.")


def fig_sensitivity(sens):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for hc, g in sens.groupby("Human cost"):
        g = g.sort_values("Missed-defect penalty")
        ax.plot(g["Missed-defect penalty"], g["Avg reward"],
                marker="o", lw=2.1, label=f"Human cost = {hc:g}")
    ax.set_xlabel("Missed-defect penalty")
    ax.set_ylabel("Average reward")
    ax.set_title("Sensitivity of Learned Policy to Decision Costs")
    ax.legend()
    fig.tight_layout()
    savefig(fig, "Fig17_cost_sensitivity_reward.png",
            "Average reward across cost settings.")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, metric, cmap in [(axes[0], "Missed defects", "OrRd"),
                             (axes[1], "False rejects", "PuBu"),
                             (axes[2], "Human reviews", "YlOrBr")]:
        p = sens.pivot(index="Missed-defect penalty",
                       columns="Human cost", values=metric)
        im = ax.imshow(p.values, aspect="auto", cmap=cmap)
        ax.set_title(metric)
        ax.grid(False)
        ax.set_xlabel("Human-review cost")
        ax.set_ylabel("Missed-defect penalty")
        ax.set_xticks(range(len(p.columns)), p.columns)
        ax.set_yticks(range(len(p.index)), p.index)
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                ax.text(j, i, int(p.iloc[i, j]), ha="center",
                        va="center", fontsize=10)
    fig.suptitle("Operational Sensitivity to Manufacturing Cost Assumptions",
                 fontsize=13)
    fig.tight_layout()
    savefig(fig, "Fig18_cost_sensitivity_errors.png",
            "Missed defects, false rejects and reviews across cost settings.")


def fig_tradeoff(sens, pr):
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    sc = ax.scatter(sens["False rejects"], sens["Missed defects"],
                    s=(sens["Human reviews"] + 1) * 55,
                    c=sens["Avg reward"], cmap="viridis",
                    edgecolor="white", alpha=.9, zorder=3)
    fig.colorbar(sc, ax=ax, fraction=.046, pad=.04).set_label("Average reward")
    b = pr[pr.Policy == "Fixed threshold"].iloc[0]
    ax.scatter(b["False rejects"], b["Missed defects"], s=190, marker="*",
               color=PC["red"], edgecolor="white", zorder=5)
    ax.annotate("Fixed threshold", (b["False rejects"], b["Missed defects"]),
                xytext=(10, 10), textcoords="offset points", fontsize=9)
    q = pr[pr.Policy == "Q-learning policy"].iloc[0]
    ax.annotate("Q-learning", (q["False rejects"], q["Missed defects"]),
                xytext=(10, -18), textcoords="offset points", fontsize=9,
                arrowprops=dict(arrowstyle="-", color=PC["slate"]))
    ax.set_xlabel("False rejects (good items scrapped)")
    ax.set_ylabel("Missed defects (defects shipped)")
    ax.set_title("The Cost Trade-off: Fewer Misses Cost More False Rejects "
                 "and Reviews\n(bubble size = human reviews; "
                 "colour = average reward)", fontsize=11)
    ax.set_xlim(-.5, max(6, sens["False rejects"].max() + 1))
    ax.set_ylim(-.5, max(4, pr["Missed defects"].max() + 1))
    fig.tight_layout()
    savefig(fig, "Fig19_tradeoff_frontier.png",
            "Missed defects vs false rejects for all configurations.")


def fig_graphical_abstract(res, size_df, pr):
    r = res.set_index("Feature set")
    s = size_df.set_index("Defect size")["Detection recall"]
    p = pr.set_index("Policy")
    fig = plt.figure(figsize=(14, 7.6))
    fig.suptitle("From Anomaly Score to Inspection Decision: Results at a Glance",
                 fontsize=16, color=PC["navy"], y=.98)

    cards = [
        ("ROC-AUC (all features)", f"{r.loc['All features', 'ROC-AUC']:.3f}",
         PC["navy"]),
        ("Recall: large vs small defects",
         f"{s['Large']:.3f} vs {s['Small']:.3f}", PC["teal"]),
        ("Avg reward: threshold -> Q-learning",
         f"{p.loc['Fixed threshold', 'Average reward']:.3f} -> "
         f"{p.loc['Q-learning policy', 'Average reward']:.3f}", PC["blue"]),
        ("Missed defects",
         f"{int(p.loc['Fixed threshold', 'Missed defects'])} -> "
         f"{int(p.loc['Q-learning policy', 'Missed defects'])}", PC["red"]),
    ]
    for k, (t, v, c) in enumerate(cards):
        ax = fig.add_axes([.04 + k * .24, .74, .22, .17])
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=c, alpha=.12,
                                   transform=ax.transAxes))
        ax.add_patch(plt.Rectangle((0, 0), .025, 1, color=c,
                                   transform=ax.transAxes))
        ax.text(.1, .66, v, fontsize=21, color=c, va="center")
        ax.text(.1, .22, t, fontsize=9.5, color=PC["dark"], va="center")

    a1 = fig.add_axes([.06, .1, .27, .5])
    fs = res.sort_values("ROC-AUC")
    b = a1.barh(fs["Feature set"], fs["ROC-AUC"],
                color=[FEATURE_COLORS[n] for n in fs["Feature set"]])
    a1.bar_label(b, fmt="%.3f", padding=3)
    a1.grid(False)
    a1.set_xlim(0, 1.1)
    a1.set_title("1  Which features work?\n(ROC-AUC)", fontsize=11)

    a2 = fig.add_axes([.40, .1, .22, .5])
    b = a2.bar(["Small", "Large"], [s["Small"], s["Large"]],
               color=[PC["amber"], PC["teal"]], width=.55)
    a2.bar_label(b, fmt="%.3f", padding=3)
    a2.grid(False, axis="x")
    a2.set_ylim(0, 1.15)
    a2.set_title("2  Does size matter?\n(recall)", fontsize=11)

    a3 = fig.add_axes([.70, .1, .26, .5])
    mm = pr.set_index("Policy")[["Missed defects", "False rejects", "Human reviews"]]
    x = np.arange(2); w = .26
    for i, (c, col) in enumerate(zip(mm.columns,
                                     [PC["red"], PC["slate"], PC["amber"]])):
        bb = a3.bar(x + (i - 1) * w, mm[c].values, w, label=c, color=col)
        a3.bar_label(bb, fmt="%.0f", padding=2, fontsize=8)
    a3.set_xticks(x, ["Fixed\nthreshold", "Q-learning"])
    a3.set_ylim(0, max(mm.values.max() * 1.45, 2))
    a3.legend(fontsize=7.5, loc="upper right", ncol=1)
    a3.set_title("3  What does the policy trade?\n(held-out counts)",
                 fontsize=11)
    savefig(fig, "graphical_abstract.png",
            "One-page visual summary of the main results.")

import argparse
import platform

import numpy as np
import pandas as pd
import cv2
import sklearn
import skimage
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from src import figures as F
from src import stats as S
from src.config import (SEED, N_BINS, OUT_DIR, FIG_DIR, ACTION_NAMES,
                        A_ACCEPT, A_REJECT, set_seed, set_plot_style)
from src.data import load_dataset
from src.features import extract_groups
from src.stage1 import stage1, size_analysis, recall_by_type
from src.stage2 import (make_reward, make_state_fn, train_q, evaluate,
                        run_sensitivity, seed_robustness)


def main(seed_check=False):
    set_seed()
    set_plot_style()
    print("Python", platform.python_version(), "| numpy", np.__version__,
          "| pandas", pd.__version__, "| opencv", cv2.__version__,
          "| scikit-image", skimage.__version__,
          "| scikit-learn", sklearn.__version__)

    data = load_dataset()
    test_df = data["test_df"]
    y = test_df["label"].to_numpy(dtype=int)

    # ---------------- features ----------------
    F.fig_dataset_overview(data)          # Fig. 2
    F.fig_training_mosaic(data)           # Fig. S1
    print("\nExtracting features ...")
    Xtr_g = extract_groups(data["train_imgs"], "train")
    Xte_g = extract_groups(data["test_imgs"], "test")
    for k in Xtr_g:
        print(f"  {k:8s} train {Xtr_g[k].shape}  test {Xte_g[k].shape}")
    F.fig_features_seen(data)             # Fig. 3

    # ---------------- Stage 1 ----------------
    print("\n=== STAGE 1: anomaly detection ===")
    res, saved = stage1(Xtr_g, Xte_g, y)
    print(res.round(3).to_string(index=False))
    best_name = "All features"   # combined detector is the paper's main one
    best = saved[best_name]
    cm = confusion_matrix(y, best["pred"])
    print(f"\nBest feature set: {best_name} | threshold = "
          f"{best['threshold']:.4f}")

    size_df, med, analysis = size_analysis(test_df, best)
    by_type = recall_by_type(analysis)
    print(f"\nMedian defect-area ratio: {med:.5f}")
    print(size_df.round(3).to_string(index=False))

    key_stats = S.key_statistics(analysis, best["threshold"])
    type_tab = S.detection_by_type(analysis)
    size_tab = S.detection_by_size(analysis)
    print("\n" + key_stats.to_string(index=False))

    F.fig_feature_family(res)                                   # Fig. 4
    F.fig_roc_pr(res, saved, y)                                 # Fig. 5
    F.fig_score_dist(best, best_name, y)                        # Fig. 6
    F.fig_confusion(cm, best_name)                              # Fig. 7
    F.fig_score_by_type(analysis, best["threshold"])            # Fig. 8
    F.fig_size_scatter(analysis, med, best["threshold"])        # Fig. 9
    F.fig_radar(res)                                            # Fig. S2
    F.fig_size(size_df)                                         # Fig. S3

    missed = analysis[(analysis.label == 1) &
                      (analysis.pred == 0)].sort_values("score")
    correct = analysis[(analysis.label == 1) &
                       (analysis.pred == 1)].sort_values("score",
                                                         ascending=False)
    F.fig_gallery(missed, data,
                  "Missed Defects (red frame; red outline = ground truth)",
                  "Fig10_missed_defects.png", F.PC["red"],
                  "Missed defects with mask outlines.")         # Fig. 10
    F.fig_gallery(correct, data,
                  "Correctly Detected Defects "
                  "(green frame; red outline = ground truth)",
                  "FigS4_detected_defects.png", F.PC["green"],
                  "Detected defects with mask outlines.")       # Fig. S4

    # ---------------- Stage 2 ----------------
    print("\n=== STAGE 2: cost-sensitive decision policy ===")
    scores = saved["All features"]["scores"]
    base_threshold = saved["All features"]["threshold"]
    cal_idx, eval_idx = train_test_split(np.arange(len(test_df)),
                                         test_size=0.50, stratify=y,
                                         random_state=SEED)
    print(f"Calibration items: {len(cal_idx)} | "
          f"Evaluation items: {len(eval_idx)}")

    score_to_state = make_state_fn(scores[cal_idx])
    reward = make_reward()
    Q, trace = train_q(scores, y, cal_idx, score_to_state, reward, track=True)
    policy = np.argmax(Q, axis=1)
    policy_table = pd.DataFrame({
        "State/bin":      np.arange(N_BINS),
        "Learned action": [ACTION_NAMES[int(a)] for a in policy],
        "Q(Accept)":      Q[:, 0],
        "Q(Reject)":      Q[:, 1],
        "Q(Human)":       Q[:, 2],
    })
    print(policy_table.round(3).to_string(index=False))

    base_fn = lambda s: A_REJECT if s >= base_threshold else A_ACCEPT
    rl_fn = lambda s: int(policy[score_to_state(s)])
    base = evaluate(eval_idx, base_fn, reward, scores, y, "Fixed threshold")
    rl = evaluate(eval_idx, rl_fn, reward, scores, y, "Q-learning policy")
    policy_results = pd.DataFrame([base, rl])
    print(policy_results.round(3).to_string(index=False))
    base_acts = [base_fn(float(scores[j])) for j in eval_idx]
    rl_acts = [rl_fn(float(scores[j])) for j in eval_idx]

    F.fig_reward_decomp(policy_results, len(eval_idx))          # Fig. 11
    F.fig_decision_matrices(y, eval_idx, base_acts, rl_acts)    # Fig. 12
    F.fig_qheat(Q)                                              # Fig. 13
    F.fig_decision_map(scores, y, eval_idx, policy, score_to_state,
                       base_threshold, rl_acts)                 # Fig. 14
    F.fig_learning(trace)                                       # Fig. 15
    F.fig_outcomes(policy_results)                              # Fig. S5

    print("\n--- Cost sensitivity (3 x 3 grid) ---")
    sens, pols = run_sensitivity(scores, y, cal_idx, eval_idx, score_to_state,
                                 [-5, -10, -20], [-0.5, -1, -2])
    print(sens.round(3).to_string(index=False))
    F.fig_fingerprints(sens, pols)                              # Fig. 16
    F.fig_sensitivity(sens)                                     # Fig. 17, 18
    F.fig_tradeoff(sens, policy_results)                        # Fig. 19
    F.fig_graphical_abstract(res, size_df, policy_results)

    posthoc = S.posthoc_thresholds(analysis, base_threshold)    # Table 8
    print("\nPost-hoc thresholds (Table 8):")
    print(posthoc.round(3).to_string(index=False))
    be = S.break_even()

    seeds = None
    if seed_check:
        seeds = seed_robustness(scores, y, cal_idx, eval_idx, score_to_state)

    # ---------------- save tables ----------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table2 = res[["Feature set", "ROC-AUC", "AP", "Accuracy", "Precision",
                  "Recall", "F1"]]
    csvs = {
        "feature_comparison": table2,
        "key_statistics": key_stats,
        "detection_by_type": type_tab,
        "detection_by_size": size_tab,
        "small_vs_large": size_df,
        "outcome_by_defect_type": by_type,
        "policy_comparison": policy_results,
        "learned_policy_table": policy_table,
        "cost_sensitivity": sens,
        "posthoc_thresholds": posthoc,
        "break_even": be,
        "test_predictions": analysis,
    }
    if seeds is not None:
        csvs["seed_robustness"] = seeds
    for name, df in csvs.items():
        df.to_csv(OUT_DIR / f"{name}.csv", index=False)

    sheets = {
        "T2_feature_families": table2, "T3_by_type": type_tab,
        "T4_by_size": size_tab, "T5_policies": policy_results,
        "T6_q_table": policy_table, "T7_cost_sensitivity": sens,
        "T8_posthoc_thresholds": posthoc, "key_statistics": key_stats,
        "break_even": be, "test_predictions": analysis,
    }
    if seeds is not None:
        sheets["seed_robustness"] = seeds
    with pd.ExcelWriter(OUT_DIR / "results.xlsx", engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0
                            for c in col)
                ws.column_dimensions[col[0].column_letter].width = \
                    min(max(width + 2, 10), 60)

    r = res.set_index("Feature set")
    p = policy_results.set_index("Policy")
    s = size_df.set_index("Defect size")["Detection recall"]
    print("\n=== Summary ===")
    print(f"  ROC-AUC (all features)  : {r.loc['All features', 'ROC-AUC']:.3f}")
    print(f"  Recall large / small    : {s['Large']:.3f} / {s['Small']:.3f}")
    print(f"  Avg reward fixed / QL   : "
          f"{p.loc['Fixed threshold', 'Average reward']:.3f} / "
          f"{p.loc['Q-learning policy', 'Average reward']:.3f}")
    print(f"  Missed defects fixed/QL : "
          f"{int(p.loc['Fixed threshold', 'Missed defects'])} / "
          f"{int(p.loc['Q-learning policy', 'Missed defects'])}")
    print(f"\nTables  -> {OUT_DIR}\nFigures -> {FIG_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-check", action="store_true",
                    help="also run the 20-seed Q-learning robustness check")
    main(ap.parse_args().seed_check)

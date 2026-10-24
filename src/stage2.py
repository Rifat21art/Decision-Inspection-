"""Stage 2: states, rewards, tabular Q-learning, evaluation, sensitivity."""
import numpy as np
import pandas as pd

from .config import (SEED, N_BINS, ALPHA, EPSILON, EPISODES,
                     A_ACCEPT, A_REJECT, A_HUMAN)


def make_reward(missed_penalty=-10.0, human_cost=-1.0, false_reject=-2.0):
    def reward(action, true_label):
        if action == A_HUMAN:
            return human_cost
        if action == A_ACCEPT:
            return 1.0 if true_label == 0 else missed_penalty
        return 1.0 if true_label == 1 else false_reject
    return reward


def make_state_fn(cal_scores, n_bins=N_BINS):
    edges = np.quantile(cal_scores, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9

    def score_to_state(s):
        return int(np.clip(np.digitize(s, edges[1:-1], right=False),
                           0, n_bins - 1))
    score_to_state.edges = edges
    return score_to_state


def train_q(scores, y, cal_idx, score_to_state, reward_fn, seed=SEED,
            episodes=EPISODES, alpha=ALPHA, epsilon=EPSILON, track=False):
    Q = np.zeros((N_BINS, 3), dtype=np.float32)
    rng = np.random.default_rng(seed)
    rewards, snaps = [], []
    for ep in range(episodes):
        j = int(rng.choice(cal_idx))
        s = score_to_state(float(scores[j]))
        a = (int(rng.integers(0, 3)) if rng.random() < epsilon
             else int(np.argmax(Q[s])))
        r = reward_fn(a, int(y[j]))
        Q[s, a] += alpha * (r - Q[s, a])
        if track:
            rewards.append(r)
            if (ep + 1) % 100 == 0:
                snaps.append(np.argmax(Q, axis=1))
    trace = ({"rewards": np.array(rewards), "snaps": np.array(snaps)}
             if track else None)
    return Q, trace


def evaluate(indices, action_fn, reward_fn, scores, y, name):
    acts = np.array([action_fn(float(scores[j])) for j in indices])
    true = np.array([int(y[j]) for j in indices])
    rew = np.array([reward_fn(a, t) for a, t in zip(acts, true)])
    return {
        "Policy":            name,
        "Average reward":    rew.mean(),
        "Total reward":      rew.sum(),
        "Missed defects":    int(((true == 1) & (acts == A_ACCEPT)).sum()),
        "False rejects":     int(((true == 0) & (acts == A_REJECT)).sum()),
        "Human reviews":     int((acts == A_HUMAN).sum()),
        "Human review rate": float((acts == A_HUMAN).mean()),
    }


def run_sensitivity(scores, y, cal_idx, eval_idx, score_to_state,
                    penalties, human_costs):
    rows, pols = [], {}
    for hc in human_costs:
        for mp in penalties:
            rf = make_reward(mp, hc)
            Q, _ = train_q(scores, y, cal_idx, score_to_state, rf)
            pol = np.argmax(Q, axis=1)
            pols[(mp, hc)] = pol
            s = evaluate(eval_idx, lambda sc: int(pol[score_to_state(sc)]),
                         rf, scores, y, "Q")
            rows.append({
                "Missed-defect penalty": mp,
                "Human cost":            hc,
                "Avg reward":            s["Average reward"],
                "Missed defects":        s["Missed defects"],
                "False rejects":         s["False rejects"],
                "Human reviews":         s["Human reviews"],
            })
    return pd.DataFrame(rows), pols


def seed_robustness(scores, y, cal_idx, eval_idx, score_to_state,
                    seeds=range(20)):
    rf = make_reward()
    rows = []
    for sd in seeds:
        Q, _ = train_q(scores, y, cal_idx, score_to_state, rf, seed=sd)
        pol = np.argmax(Q, axis=1)
        rows.append(evaluate(eval_idx,
                             lambda sc: int(pol[score_to_state(sc)]),
                             rf, scores, y, f"seed{sd}"))
    d = pd.DataFrame(rows)
    print("\nSeed robustness (Q-learning, mean +/- sd):")
    for c in ["Average reward", "Missed defects",
              "False rejects", "Human reviews"]:
        print(f"  {c:16s} {d[c].mean():.3f} +/- {d[c].std():.3f}")
    return d

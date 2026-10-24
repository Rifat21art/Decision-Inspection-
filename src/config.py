"""Constants, palette, seed and plot style."""
import os
import random
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
if "ipykernel" not in sys.modules and not os.environ.get("MPLBACKEND"):
    mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
CATEGORY = "bottle"
DATA_DIR = Path(os.environ.get("MVTEC_DIR", ROOT / "data"))
OUT_DIR = Path(os.environ.get("OUT_DIR", ROOT / "results"))
FIG_DIR = Path(os.environ.get("FIG_DIR", ROOT / "figures"))

IMG_SIZE, LBP_POINTS, LBP_RADIUS = 128, 24, 3
N_BINS = 10
ALPHA, EPSILON, EPISODES = 0.15, 0.20, 4000

A_ACCEPT, A_REJECT, A_HUMAN = 0, 1, 2
ACTION_NAMES = {0: "Accept", 1: "Reject", 2: "Human"}

# Muted, print-friendly palette (same colours in every figure)
PC = {
    "navy":  "#16324F",
    "teal":  "#2F6F73",
    "slate": "#667085",
    "blue":  "#4C78A8",
    "amber": "#C08A3E",
    "red":   "#A64B4B",
    "green": "#557A61",
    "dark":  "#1F2933",
}
FEATURE_COLORS = {
    "Color":               PC["slate"],
    "Texture (LBP)":       PC["amber"],
    "Shape (HOG + edges)": PC["teal"],
    "All features":        PC["navy"],
}
ACTION_COLORS = {"Accept": PC["teal"], "Reject": PC["red"], "Human": PC["amber"]}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


def set_plot_style():
    """Plain Matplotlib defaults; no font family or weight overrides."""
    mpl.rcParams.update({
        "figure.dpi":        110,
        "savefig.dpi":       300,
        "axes.edgecolor":    "#98A2B3",
        "axes.linewidth":    0.8,
        "axes.grid":         True,
        "grid.color":        "#D0D5DD",
        "grid.linewidth":    0.6,
        "grid.alpha":        0.65,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "legend.frameon":    False,
    })


def savefig(fig, name, caption=""):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)

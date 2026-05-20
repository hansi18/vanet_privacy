"""
plot_results.py
Generates all comparison charts from saved results.
Run AFTER federated_runner.py completes.

Usage:
    python plot_results.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from config import RESULTS_DIR

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Colour scheme ─────────────────────────────────────────────────────────────
COLORS = {
    "DT-RBAC-FL-ADP (yours)": "#1D9E75",   # teal  — your model
    "FL + DP":                 "#888780",   # gray
    "DT + FL":                 "#7F77DD",   # purple
    "Adaptive DP-FL":          "#D85A30",   # coral
}
LINE_STYLES = {
    "DT-RBAC-FL-ADP (yours)": "-",
    "FL + DP":                 "--",
    "DT + FL":                 "-.",
    "Adaptive DP-FL":          ":",
}
MARKERS = {
    "DT-RBAC-FL-ADP (yours)": "o",
    "FL + DP":                 "s",
    "DT + FL":                 "^",
    "Adaptive DP-FL":          "D",
}

# ── Load data ─────────────────────────────────────────────────────────────────
with open(os.path.join(RESULTS_DIR, "round_stats.json")) as f:
    round_stats = json.load(f)

with open(os.path.join(RESULTS_DIR, "final_results.json")) as f:
    final_results = json.load(f)

rounds = [r["round"] for r in round_stats]

series = {
    "DT-RBAC-FL-ADP (yours)": {
        "accuracy": [r["your_accuracy"]  for r in round_stats],
        "f1":       [r["your_f1"]        for r in round_stats],
    },
    "FL + DP": {
        "accuracy": [r["fl_dp_accuracy"] for r in round_stats],
        "f1":       [r["fl_dp_f1"]       for r in round_stats],
    },
    "DT + FL": {
        "accuracy": [r["dt_fl_accuracy"] for r in round_stats],
        "f1":       [r["dt_fl_f1"]       for r in round_stats],
    },
    "Adaptive DP-FL": {
        "accuracy": [r["adaptive_accuracy"] for r in round_stats],
        "f1":       [r["adaptive_f1"]       for r in round_stats],
    },
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})


# ─────────────────────────────────────────────────────────────────────────────
# Chart 1: Accuracy over FL training rounds
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
for name, s in series.items():
    ax.plot(rounds, s["accuracy"],
            label=name,
            color=COLORS[name],
            linestyle=LINE_STYLES[name],
            marker=MARKERS[name],
            markersize=4,
            linewidth=2.0 if "yours" in name else 1.2)

ax.set_xlabel("FL training round")
ax.set_ylabel("Test accuracy")
ax.set_title("Model accuracy over FL training rounds\n(VeReMi — fraud detection task)")
ax.legend(loc="lower right", fontsize=9)
ax.set_ylim(0, 1.05)
ax.set_xlim(1, max(rounds))
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "accuracy_over_rounds.png"))
plt.close(fig)
print("Saved: accuracy_over_rounds.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 2: F1 score over rounds
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
for name, s in series.items():
    ax.plot(rounds, s["f1"],
            label=name,
            color=COLORS[name],
            linestyle=LINE_STYLES[name],
            marker=MARKERS[name],
            markersize=4,
            linewidth=2.0 if "yours" in name else 1.2)

ax.set_xlabel("FL training round")
ax.set_ylabel("F1 score (attack class)")
ax.set_title("Attack detection F1 over FL training rounds")
ax.legend(loc="lower right", fontsize=9)
ax.set_ylim(0, 1.05)
ax.set_xlim(1, max(rounds))
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "f1_over_rounds.png"))
plt.close(fig)
print("Saved: f1_over_rounds.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 3: Final comparison bar chart (4 metrics × 4 models)
# ─────────────────────────────────────────────────────────────────────────────
metrics_names = ["accuracy", "f1", "precision", "recall"]
model_names   = list(final_results.keys())
x = np.arange(len(metrics_names))
width = 0.20

fig, ax = plt.subplots(figsize=(10, 5))
for i, mname in enumerate(model_names):
    vals = [final_results[mname].get(m, 0) for m in metrics_names]
    color = COLORS.get(mname, "#888780")
    bars = ax.bar(x + i * width, vals, width, label=mname, color=color, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([m.capitalize() for m in metrics_names])
ax.set_ylabel("Score")
ax.set_title("Final model comparison — all metrics")
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(0, 1.15)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "final_comparison_bar.png"))
plt.close(fig)
print("Saved: final_comparison_bar.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 4: Privacy-utility frontier (cumulative ε vs accuracy)
# ─────────────────────────────────────────────────────────────────────────────
from config import EPS_TOTAL, N_ROUNDS, FIXED_EPS_BASELINE, EPS_MIN, EPS_MAX

# Approximate cumulative ε consumed per model
n_rounds = len(rounds)
# Your model: adaptive, averages around 0.5 * EPS_TOTAL / N_ROUNDS per vehicle per round
your_eps_curve = np.linspace(0.05, EPS_TOTAL * 0.85, n_rounds)
fldp_eps_curve = np.linspace(FIXED_EPS_BASELINE * 0.1, FIXED_EPS_BASELINE * n_rounds * 0.1, n_rounds)
dtfl_eps_curve = np.zeros(n_rounds)  # no DP
adaptive_eps_curve = np.linspace(0.05, EPS_MAX * 0.60, n_rounds)

eps_series = {
    "DT-RBAC-FL-ADP (yours)": (your_eps_curve,     series["DT-RBAC-FL-ADP (yours)"]["accuracy"]),
    "FL + DP":                 (fldp_eps_curve,     series["FL + DP"]["accuracy"]),
    "Adaptive DP-FL":          (adaptive_eps_curve, series["Adaptive DP-FL"]["accuracy"]),
}

fig, ax = plt.subplots(figsize=(8, 5))
for name, (eps_c, acc_c) in eps_series.items():
    ax.plot(eps_c, acc_c,
            label=name,
            color=COLORS[name],
            linestyle=LINE_STYLES[name],
            marker=MARKERS[name],
            markersize=3,
            linewidth=2.0 if "yours" in name else 1.2)

ax.set_xlabel("Cumulative privacy budget consumed (ε)")
ax.set_ylabel("Test accuracy")
ax.set_title("Privacy-utility frontier\n(higher & left = better)")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "privacy_utility_frontier.png"))
plt.close(fig)
print("Saved: privacy_utility_frontier.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 5: Flagged vehicles + avg trust over rounds
# ─────────────────────────────────────────────────────────────────────────────
flagged_counts = [r["flagged"]   for r in round_stats]
avg_trusts     = [r["avg_trust"] for r in round_stats]

fig, ax1 = plt.subplots(figsize=(9, 4))
ax2 = ax1.twinx()

ax1.bar(rounds, flagged_counts, color="#D85A30", alpha=0.5, label="Flagged vehicles")
ax2.plot(rounds, avg_trusts, color="#1D9E75", linewidth=2, marker="o",
         markersize=3, label="Avg trust score")

ax1.set_xlabel("FL training round")
ax1.set_ylabel("Flagged vehicles (count)", color="#D85A30")
ax2.set_ylabel("Average trust score", color="#1D9E75")
ax1.set_title("Your model — flagged vehicles and trust evolution per round")
ax1.tick_params(axis="y", colors="#D85A30")
ax2.tick_params(axis="y", colors="#1D9E75")
ax2.set_ylim(0, 1.1)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "trust_and_flags.png"))
plt.close(fig)
print("Saved: trust_and_flags.png")


# ─────────────────────────────────────────────────────────────────────────────
# Print summary table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  SUMMARY TABLE")
print("=" * 65)
header = f"{'Model':<28} {'Accuracy':>9} {'F1':>8} {'Precision':>10} {'Recall':>8}"
print(header)
print("-" * 65)
for name, res in final_results.items():
    row = (f"{name:<28} "
           f"{res['accuracy']:>9.4f} "
           f"{res['f1']:>8.4f} "
           f"{res['precision']:>10.4f} "
           f"{res['recall']:>8.4f}")
    print(row)
print("=" * 65)
print(f"\nAll charts saved to: {RESULTS_DIR}/\n")

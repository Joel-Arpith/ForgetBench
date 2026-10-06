"""Figures for the Review III report. Every plotted number is a measured value
from notebook 05 (single seed) -- nothing here is estimated."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np, os

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times"],
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
GREY, ACC, DARK = "#9aa3b2", "#1f6f8b", "#222222"

ARMS = [  # name, retain_lp, durability, truth_ratio, updates
    ("Full retrain", -0.236, -0.582, 1.007, 150), ("SISA", -0.459, -0.457, 1.023, 26),
    ("Sharded control", -0.390, -0.380, 0.994, 0), ("Gradient ascent", -1.421, -1.088, 1.665, 40),
    ("NPO", -1.594, -0.762, 13.166, 40), ("RMU", -0.578, -0.450, 0.986, 40),
    ("Scrub only (sharded)", -0.484, -0.459, 1.155, 120), ("HRU (sharded)", -0.444, -0.420, 1.059, 120),
    ("HRU (unsharded)", -1.692, -1.309, 1.292, 120), ("HRU k=2", -0.415, -0.378, 1.096, 120),
    ("HRU k=4", -0.460, -0.439, 1.088, 120), ("HRU k=8", -0.694, -0.670, 1.029, 120),
]

def box(ax, x, y, w, h, text, fc="white", ec=DARK, bold=False, fs=7.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)

def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.2))

# ---------------------------------------------------------------- Fig 3.1 pipeline
fig, ax = plt.subplots(figsize=(6.6, 3.4)); ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis("off")
box(ax, 0.1, 1.9, 1.7, 0.9, "Synthetic data\n130 fictional\nprofiles", fc="#eef4f7")
box(ax, 2.2, 1.9, 1.6, 0.9, "Base model\nGPT-2 (124M)\nfine-tuned", fc="#eef4f7")
for i, (t, sub) in enumerate([("Arm A — exact", "Retrain · SISA"), ("Arm B — approximate", "GA · NPO · RMU"),
                               ("Arm C — HRU", "Shard · SAM-NPO · harden")]):
    y = 3.3 - i * 1.25
    box(ax, 4.1, y, 2.45, 0.85, f"{t}\n{sub}", fc="#d5e8f0" if i == 2 else "white", bold=(i == 2))
    arrow(ax, 3.8, 2.35, 4.1, y + 0.42)
    arrow(ax, 6.55, y + 0.42, 6.8, 2.35)
box(ax, 6.8, 1.9, 1.4, 0.9, "Relearning\nattack\n15 steps", fc="#fbeaea")
box(ax, 8.55, 1.6, 1.4, 1.5, "Evaluation\nforget · retain\ntruth ratio\nMIA · cost\ndurability", fc="#eef4f7")
arrow(ax, 1.8, 2.35, 2.2, 2.35); arrow(ax, 8.2, 2.35, 8.55, 2.35)
ax.text(5.3, 0.05, "Every arm: same base model, forget set, probes, seeds and cost accounting",
        ha="center", fontsize=8, style="italic")
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_1_pipeline.png", dpi=300); plt.close(fig)

# ---------------------------------------------------------------- Fig 3.2 HRU method
fig, ax = plt.subplots(figsize=(6.6, 2.9)); ax.set_xlim(0, 10); ax.set_ylim(0, 3.8); ax.axis("off")
box(ax, 0.1, 1.4, 2.0, 1.0, "1. Localise\nforget data in\nshard 0", fc="#eef4f7")
box(ax, 2.7, 1.4, 2.3, 1.0, "2. Scrub\nNPO loss, SAM\nmid-stack MLP (22.8%)", fc="#d5e8f0", bold=True)
box(ax, 5.6, 1.4, 2.2, 1.0, "3. Probe\nattack a copy,\nmeasure leakage", fc="#fbeaea")
box(ax, 8.4, 1.4, 1.5, 1.0, "Hardened\nmodel", fc="#eef4f7")
arrow(ax, 2.1, 1.9, 2.7, 1.9); arrow(ax, 5.0, 1.9, 5.6, 1.9); arrow(ax, 7.8, 1.9, 8.4, 1.9)
ax.annotate("", xy=(3.85, 2.4), xytext=(6.7, 2.4),
            arrowprops=dict(arrowstyle="-|>", color=ACC, lw=1.4, connectionstyle="arc3,rad=0.45"))
ax.text(5.27, 3.45, "repeat each round until leakage stops improving", ha="center", fontsize=8, color=ACC)
ax.text(5.0, 0.35, "Validation phase adds per-example re-weighting: examples the probe relearns fastest get more weight in the next scrub steps.",
        ha="center", fontsize=7.5, style="italic")
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_2_hru.png", dpi=300); plt.close(fig)

# ---------------------------------------------------------------- Fig 4.1 durability bars
sel = ["HRU (unsharded)", "Gradient ascent", "NPO", "HRU k=8", "Full retrain", "SISA", "RMU", "HRU (sharded)"]
d = {a[0]: a for a in ARMS}
vals = [-d[n][2] for n in sel]
fig, ax = plt.subplots(figsize=(6.5, 3.6))
bars = ax.barh(range(len(sel))[::-1], vals, color=[ACC if n == "HRU (unsharded)" else GREY for n in sel])
for i, (n, v) in enumerate(zip(sel, vals)):
    ax.text(v + 0.02, len(sel) - 1 - i, f"{v:.3f}", va="center", fontsize=10)
ax.set_yticks(range(len(sel))[::-1], sel)
ax.set_xlabel("Forget-set loss after attack (negated log-prob; higher = more durable)", fontsize=9)
ax.set_xlim(0, 1.5); ax.grid(axis="x", alpha=.3)
ax.set_title("HRU (unsharded) retains the most forgetting after the attack", fontsize=10.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_1_durability.png", dpi=300); plt.close(fig)

# ---------------------------------------------------------------- Fig 4.2 sharding dilution
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3.0))
a1.bar(["Scrubbed shard\nalone", "Inside 4-shard\nensemble"], [4.179, 0.502], color=[ACC, GREY])
for i, v in enumerate([4.179, 0.502]): a1.text(i, v + 0.08, f"{v:.3f}", ha="center")
a1.set_ylim(0, 4.179 * 1.15)
a1.set_ylabel("Forget-set loss (negated log-prob)"); a1.set_title("Same scrubbed shard", fontsize=11)
a2.bar(["Unsharded", "Sharded"], [1.309, 0.420], color=[ACC, GREY])
for i, v in enumerate([1.309, 0.420]): a2.text(i, v + 0.03, f"{v:.3f}", ha="center")
a2.set_ylim(0, 1.309 * 1.15)
a2.set_ylabel("Loss after attack (higher = durable)"); a2.set_title("Durability after attack", fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_2_sharding.png", dpi=300); plt.close(fig)

# ---------------------------------------------------------------- Fig 4.3 durability vs damage
x = np.array([a[1] for a in ARMS]); y = np.array([a[2] for a in ARMS])
b, a0 = np.polyfit(x, y, 1); r = np.corrcoef(x, y)[0, 1]
fig, ax = plt.subplots(figsize=(6.5, 4.0))
xs = np.linspace(x.min(), x.max(), 2); ax.plot(xs, a0 + b * xs, "--", color=DARK, lw=1, label="least-squares fit")
for n, rx, dy, *_ in ARMS:
    hi = n == "HRU (unsharded)"
    ax.scatter(rx, dy, s=80 if hi else 45, color=ACC if hi else GREY, zorder=3, edgecolor="black", lw=.5)
    if n in ("HRU (unsharded)", "Gradient ascent", "NPO", "Full retrain"):
        ax.annotate(n, (rx, dy), xytext=(-8 if n == "Full retrain" else 8, 6), textcoords="offset points",
                    ha="right" if n == "Full retrain" else "left", fontsize=10, fontweight="bold" if hi else "normal")
ax.set_xlabel("Retain log-prob (further left = more damage to the model)")
ax.set_ylabel("Durability: forget log-prob after attack\n(lower = more durable)")
ax.set_title(f"Durability correlates with model damage (r = {r:+.2f}, 12 arms)", fontsize=10.5)
ax.grid(alpha=.3); ax.legend(loc="lower right", frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_3_damage.png", dpi=300); plt.close(fig)

res = {n: dy - (a0 + b * rx) for n, rx, dy, *_ in ARMS}
print(f"fit: dur = {a0:.3f} + {b:.3f} * retain ; r = {r:.3f}")
for n, v in sorted(res.items(), key=lambda kv: kv[1]): print(f"  {n:22s} residual {v:+.3f}")

# ---------------------------------------------------------------- Fig 4.4 cost-durability sweet spot
# Target zone: durability between -0.6 and -1.56 (at least as durable as full retraining)
# at a cost below full retraining (150 updates). Same measured values as Table 4.1.
ZONE = (-1.56, -0.6)
SWEET = ["Full retrain", "SISA", "Gradient ascent", "NPO", "RMU", "HRU (unsharded)", "HRU (sharded)", "HRU k=8"]
LBL = {"Full retrain": "Full retrain", "SISA": "SISA", "Gradient ascent": "Gradient ascent  (TR 1.67)",
       "NPO": "NPO  (TR 13.17, over-forgets)", "RMU": "RMU", "HRU (unsharded)": "HRU  (TR 1.29)",
       "HRU (sharded)": "HRU sharded k=4", "HRU k=8": "HRU k=8"}
OFF = {"Full retrain": (-8, 8, "right"), "SISA": (8, -12, "left"), "Gradient ascent": (8, 6, "left"),
       "NPO": (8, 6, "left"), "RMU": (8, 6, "left"), "HRU (unsharded)": (-10, 8, "right"),
       "HRU (sharded)": (-10, -14, "right"), "HRU k=8": (-12, 3, "right")}

def sweet_spot(path, dark=False):
    fg, mut, acc, grey, zone = (("#EAF0FA", "#7C89A3", "#2DD4BF", "#7C89A3", "#2DD4BF") if dark
                                else (DARK, "#555555", ACC, GREY, ACC))
    with plt.rc_context({"font.family": "sans-serif" if dark else "serif", "text.color": fg,
                         "axes.labelcolor": fg, "xtick.color": mut, "ytick.color": mut, "axes.edgecolor": mut}):
        fig, ax = plt.subplots(figsize=(6.5, 4.2) if not dark else (11, 6.2))
        if dark: fig.patch.set_facecolor("#0A1020"); ax.set_facecolor("#0A1020")
        ax.axhspan(ZONE[0], ZONE[1], xmin=0, xmax=150 / 165, color=zone, alpha=.13, lw=0)
        ax.plot([0, 150, 150, 0, 0], [ZONE[1], ZONE[1], ZONE[0], ZONE[0], ZONE[1]], "--", color=zone, lw=1)
        ax.text(4, ZONE[0] + 0.05, "TARGET ZONE: more durable than retraining, cheaper than retraining",
                fontsize=8 if not dark else 12, color=zone, fontweight="bold")
        for n in SWEET:
            _, rl, dur, tr, upd = d[n]
            hi = n == "HRU (unsharded)"
            ax.scatter(upd, dur, s=(140 if hi else 50) * (2 if dark else 1), color=acc if hi else grey,
                       edgecolor=fg if hi else "none", lw=1, zorder=3)
            dx, dy, ha = OFF[n]
            ax.annotate(LBL[n], (upd, dur), xytext=(dx, dy), textcoords="offset points", ha=ha,
                        fontsize=(9.5 if hi else 8.5) * (1.45 if dark else 1), color=acc if hi else fg,
                        fontweight="bold" if hi else "normal")
        ax.set_xlim(0, 165); ax.set_ylim(-1.75, -0.25); ax.invert_yaxis()
        ax.set_xlabel("Cost: optimiser updates per erasure request", fontsize=10 if not dark else 14)
        ax.set_ylabel("Durability after attack\n(forget log-prob; up = more durable)", fontsize=10 if not dark else 14)
        ax.tick_params(labelsize=9 if not dark else 13)
        ax.grid(alpha=.15 if dark else .3)
        fig.tight_layout(); fig.savefig(path, dpi=300 if not dark else 200, facecolor=fig.get_facecolor()); plt.close(fig)

sweet_spot(f"{OUT}/fig4_4_sweetspot.png")
sweet_spot(f"{os.path.dirname(OUT)}/fig_sweetspot_dark.png", dark=True)

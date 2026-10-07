"""Regenerate the three data figures of the MA paper from the model_13000 measurements."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = "/home/qiwang/torque_jump2/A Sample Article Using IEEEtran.cls for IEEE Journals and Transactions"
SP = "/tmp/claude-1000/-home-qiwang-torque-jump2/27d349a7-1663-4a13-9cd2-49b951a1ba44/scratchpad"
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": "0.6"})
BLUE, ORANGE, GREY = "#1f77b4", "#d1471a", "0.72"

# ---------------------------------------------------------------- tracking
CMD = np.array([0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
DIST = np.array([0.478, 0.558, 0.684, 0.816, 0.930, 0.979, 1.014, 1.040])
HIT = np.array([0.74, 0.95, 0.98, 0.97, 0.99, 0.91, 0.35, 0.00])
ERR = np.array([0.086, 0.058, 0.058, 0.057, 0.046, 0.057, 0.121, 0.198])
BEST = 4                      # index of the 0.70 m command
CEIL = 1.04

fig, ax = plt.subplots(3, 1, figsize=(3.35, 3.9), sharex=True)
ax[0].plot(CMD, DIST, "-o", color=BLUE, ms=3.5, lw=1.4)
ax[0].axhline(CEIL, ls="--", lw=0.9, color="0.35")
ax[0].text(0.62, CEIL + 0.012, "reach ceiling %.2f m" % CEIL, fontsize=6.5, color="0.3")
ax[0].annotate("%.3f m" % DIST[BEST], (CMD[BEST], DIST[BEST]), textcoords="offset points",
               xytext=(2, -13), fontsize=7, color=BLUE)
ax[0].set_ylabel("total distance (m)")
ax[0].set_ylim(0.40, 1.13)

bars = ax[1].bar(CMD, HIT, width=0.062, color=BLUE)
bars[BEST].set_color(ORANGE)
ax[1].text(CMD[BEST], HIT[BEST] + 0.04, "%.2f" % HIT[BEST], ha="center", fontsize=7, color=ORANGE)
ax[1].set_ylabel("hit rate")
ax[1].set_ylim(0, 1.18)

ax[2].plot(CMD, ERR, "-o", color=BLUE, ms=3.5, lw=1.4)
ax[2].axhline(0.10, ls="--", lw=0.9, color=ORANGE)
ax[2].text(0.30, 0.108, "0.10 m tolerance", fontsize=6.5, color=ORANGE)
ax[2].set_ylabel("landing error (m)")
ax[2].set_xlabel("commanded landing distance (m)")
ax[2].set_ylim(0, 0.215)
ax[2].set_xticks(CMD)
fig.tight_layout(pad=0.4)
fig.subplots_adjust(left=0.175)
fig.savefig(os.path.join(OUT, "fig_tracking.pdf"))
fig.savefig(os.path.join(SP, "fig_tracking.png"), dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- payload
KG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
PHIT = np.array([1.00, 0.99, 1.00, 0.99, 0.91, 0.88, 0.14])
PDIST = np.array([0.932, 0.916, 0.909, 0.889, 0.865, 0.828, 0.738])

fig, ax = plt.subplots(2, 1, figsize=(3.35, 2.6), sharex=True)
ax[0].axvspan(-1.0, 1.0, color="#dce6f1", alpha=0.65, lw=0)
ax[0].text(-0.9, 1.02, "training range", fontsize=6.5, color="0.35", va="bottom")
ax[0].bar(KG, PHIT, width=0.22, color=BLUE)
ax[0].set_ylabel("hit rate"); ax[0].set_ylim(0, 1.2)
ax[1].axvspan(-1.0, 1.0, color="#dce6f1", alpha=0.65, lw=0)
ax[1].plot(KG, PDIST, "-o", color=ORANGE, ms=3.5, lw=1.4)
ax[1].set_ylabel("total distance (m)")
ax[1].set_xlabel("trunk payload (kg)")
ax[1].set_xlim(-1.0, 5.4)
ax[1].set_xticks([0, 1, 2, 3, 5])
fig.tight_layout(pad=0.4)
fig.subplots_adjust(left=0.175)
fig.savefig(os.path.join(OUT, "fig_payload.pdf"))
fig.savefig(os.path.join(SP, "fig_payload.png"), dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- push
npz = os.path.join(SP, "push13k.npz")
if os.path.exists(npz):
    d = np.load(npz)
    V, T, G, dt = d["V"], d["T"], d["G"], float(d["dt"])
    K = V.shape[1]
    ms = (np.arange(K) - (K - 1)) * dt * 1000.0
    groups = {"front hip": [0, 3], "front thigh": [1, 4], "front calf": [2, 5],
              "rear hip": [6, 9], "rear thigh": [7, 10], "rear calf": [8, 11]}
    fig, ax = plt.subplots(3, 1, figsize=(3.35, 4.2), sharex=True)
    g = G.mean(axis=0)
    ax[0].plot(ms, g, color=BLUE, lw=1.4)
    ax[0].axhline(147.0, ls="--", lw=0.9, color="0.35")
    ax[0].text(ms[0] + 5, 162, "body weight", fontsize=6.5, color="0.3")
    ax[0].annotate("re-plant", (ms[g.argmax()], g.max()), textcoords="offset points",
                   xytext=(6, -4), fontsize=7, color="0.25")
    air = (G < 1.0).mean(axis=0)
    off = np.where(air > 0.5)[0]
    if len(off):
        ax[0].annotate("feet unloaded", (ms[off[len(off) // 2]], 8), textcoords="offset points",
                       xytext=(-14, 42), fontsize=6.5, color="0.25",
                       arrowprops=dict(arrowstyle="->", lw=0.7, color="0.45"))
    ax[0].set_ylabel("foot force (N)")

    for nm, ids in groups.items():
        q = T[:, :, ids].mean(axis=(0, 2))
        if nm in ("front thigh", "rear thigh", "rear calf"):
            c = {"front thigh": BLUE, "rear thigh": ORANGE, "rear calf": "#5a9bd4"}[nm]
            off = {"front thigh": (-44, 4), "rear thigh": (-46, -11), "rear calf": (-34, 5)}[nm]
            ax[1].plot(ms, q, color=c, lw=1.4, zorder=3)
            i = q.argmax()
            ax[1].annotate(nm, (ms[i], q[i]), textcoords="offset points", xytext=off,
                           fontsize=6.5, color=c)
        else:
            ax[1].plot(ms, q, color=GREY, lw=1.0, zorder=1)
    ax[1].set_ylabel("torque / limit"); ax[1].set_ylim(0, 1.18); ax[1].set_xlim(ms[0], 40)

    v = V[:, :, [8, 11]].mean(axis=(0, 2))
    q = T[:, :, [8, 11]].mean(axis=(0, 2))
    ax[2].plot(ms, v, color=BLUE, lw=1.4, label="speed / no-load speed")
    ax[2].plot(ms, q, color=ORANGE, lw=1.4, label="torque / torque limit")
    ax[2].axhline(1.0, ls="--", lw=0.9, color="0.35")
    ax[2].set_ylabel("rear calf"); ax[2].set_xlabel("time before take-off (ms)")
    ax[2].set_ylim(0, 1.55)
    ax[2].legend(loc="lower center", bbox_to_anchor=(0.5, -0.62), frameon=False, ncol=1)
    fig.tight_layout(pad=0.4)
    fig.subplots_adjust(bottom=0.185, left=0.155)
    fig.savefig(os.path.join(OUT, "fig_push.pdf"))
    fig.savefig(os.path.join(SP, "fig_push.png"), dpi=170)
    plt.close(fig)
    print("push figure written")
else:
    print("no npz yet, skipped push")
print("done")

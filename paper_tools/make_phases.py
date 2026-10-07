import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import os

SP = "/tmp/claude-1000/-home-qiwang-torque-jump2/27d349a7-1663-4a13-9cd2-49b951a1ba44/scratchpad"
FR = os.path.join(SP, "frames")
PANELS = [("stand", 109, -440), ("load", 162, -175), ("push", 186, -55),
          ("flight", 249, 260), ("pre-landing", 275, 390), ("touchdown", 295, 490)]
CROP = (200, 100, 700, 492)   # left, top, right, bottom

fig, axes = plt.subplots(1, len(PANELS), figsize=(7.16, 1.55))
for ax, (lbl, k, ms) in zip(axes, PANELS):
    im = Image.open(os.path.join(FR, "f%05d.png" % k)).convert("RGB").crop(CROP)
    ax.imshow(im)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.5); s.set_color("0.4")
    ax.set_title(lbl, fontsize=7, pad=2)
    ax.set_xlabel("%+d ms" % ms, fontsize=6, labelpad=1)
plt.subplots_adjust(left=0.004, right=0.996, top=0.86, bottom=0.10, wspace=0.03)
out = os.path.join(SP, "fig_phases.pdf")
plt.savefig(out, dpi=300)
plt.savefig(os.path.join(SP, "fig_phases.png"), dpi=200)
print("wrote", out)

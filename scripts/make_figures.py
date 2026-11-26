"""Figures 1 and 2 of the paper (values copied from the paper's tables)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
unsw = {"XGBoost": (0.626, 0.510), "RF": (0.583, 0.475), "CNN": (0.437, 0.397), "LR on XGB": (0.616, 0.503),
        "Stack RF+XGB": (0.619, 0.504), "Stack RF+XGB+CNN": (0.619, 0.504)}
nsl = {"XGBoost": (0.944, 0.521), "RF": (0.926, 0.489), "CNN": (0.769, 0.463), "LR on XGB": (0.941, 0.510),
       "Stack RF+XGB": (0.927, 0.511), "Stack RF+XGB+CNN": (0.946, 0.516)}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, d, t in zip(axes, [unsw, nsl], ["UNSW-NB15", "NSL-KDD"]):
    n = list(d); x = np.arange(len(n)); w = 0.38
    ax.bar(x - w/2, [d[k][0] for k in n], w, label="Grouped CV", color="#4C72B0")
    ax.bar(x + w/2, [d[k][1] for k in n], w, label="Official test", color="#DD8452")
    for i, k in enumerate(n): ax.text(i, max(d[k]) + 0.02, f"-{d[k][0]-d[k][1]:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(n, rotation=35, ha="right", fontsize=8); ax.set_title(t); ax.set_ylim(0, 1.05)
axes[0].set_ylabel("Macro-F1"); axes[0].legend(frameon=False); plt.tight_layout(); plt.savefig("fig1_cv_vs_test.png", dpi=300)
plt.close()
s = {"XGBoost": ([.2461,.1973,.1530,.1177,.0979,.0850,.0594], [.9727,.9624,.9509,.9376,.9293,.9238,.9068], [1,2,4,8,12,16,32]),
     "RF": ([.244,.136,.068,.034], [.978,.933,.883,.839], [1,2,4,8]),
     "Stack RF+XGB": ([.224,.175,.126,.080], [.973,.955,.926,.900], [1,2,4,8]),
     "Stack RF+XGB+CNN": ([.237,.181,.124,.071], [.974,.960,.936,.901], [1,2,4,8])}
fig, ax = plt.subplots(figsize=(6.4, 4.8))
for (n, (x, y, w)), c in zip(s.items(), ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]):
    ax.plot(x, y, "-o", ms=4, label=n, color=c)
    for xx, yy, wt in ((x[0], y[0], w[0]), (x[-1], y[-1], w[-1])):
        ax.annotate(f"w={wt}", (xx, yy), textcoords="offset points", xytext=(5, -10), fontsize=7, color=c)
ax.set_xlabel("False alarm rate (lower is better)"); ax.set_ylabel("Attack detection (higher is better)")
ax.set_title("UNSW-NB15 test: Normal-weight sweep (diagnostic, not tuned)", fontsize=10)
ax.legend(frameon=False, fontsize=8); plt.tight_layout(); plt.savefig("fig2_far_vs_detection.png", dpi=300)

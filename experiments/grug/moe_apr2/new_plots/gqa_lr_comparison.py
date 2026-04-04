"""Plot MHA vs GQA 4:1 BPB curves for d512 t4.5x."""
import numpy as np
import matplotlib.pyplot as plt

mha_data = [
    (0.0004, 1.3172), (0.0008, 1.2257), (0.0012, 1.1906), (0.0016, 1.1730),
    (0.0024, 1.1589), (0.0028, 1.1581), (0.0032, 1.1580), (0.0036, 1.1596),
    (0.0040, 1.1610), (0.0044, 1.1621),
]

gqa_data = [
    (0.0004, 1.3371), (0.0008, 1.2368), (0.0012, 1.2008), (0.0016, 1.1807),
    (0.0024, 1.1677), (0.0028, 1.1625), (0.0032, 1.1616), (0.0036, 1.1622),
    (0.0040, 1.1634), (0.0044, 1.1637),
]

fig, ax = plt.subplots(figsize=(8, 5))

for data, label, color in [(mha_data, "MHA (kv=4)", "C0"), (gqa_data, "GQA 4:1 (kv=1)", "C3")]:
    lrs = np.array([d[0] for d in data])
    bpbs = np.array([d[1] for d in data])

    ax.plot(lrs, bpbs, "o", color=color, markersize=7, alpha=0.8)

    # Log-space quadratic fit
    log_lrs = np.log(lrs)
    coeffs = np.polyfit(log_lrs, bpbs, 2)
    lr_fine = np.geomspace(lrs.min() * 0.8, lrs.max() * 1.2, 200)
    ax.plot(lr_fine, np.polyval(coeffs, np.log(lr_fine)), "--", color=color, alpha=0.5)

    # Optimal
    if coeffs[0] > 0:
        opt_log = -coeffs[1] / (2 * coeffs[0])
        opt_lr = np.exp(opt_log)
        opt_bpb = np.polyval(coeffs, opt_log)
        ax.axvline(opt_lr, color=color, alpha=0.3, linewidth=1, linestyle=":")
        ax.plot(opt_lr, opt_bpb, "*", color=color, markersize=12)
        label += f" (opt={opt_lr:.4f}, bpb={opt_bpb:.4f})"

    ax.plot([], [], "o-", color=color, label=label)

ax.set_xlabel("Adam LR")
ax.set_ylabel("c4_en/bpb")
ax.set_title("d512 t4.5x: MHA vs GQA 4:1 (log-space quadratic fit)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("experiments/grug/moe_apr2/new_plots/gqa_lr_comparison.png", dpi=150)
plt.close(fig)
print("Saved gqa_lr_comparison.png")

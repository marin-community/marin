# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate all v2-style plots from LR sweep data."""

import json
import math
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

PLOT_DIR = "experiments/grug/moe_apr2/new_plots"
SEQ_LEN = 4096

with open(f"{PLOT_DIR}/all_lr_sweep_results.json") as f:
    results = json.load(f)

# Hardcoded active_params used in the original sweeps (for step count calculation)
active_params = {512: 154e6, 768: 262e6, 1024: 420e6, 1280: 441e6}
batch_sizes = {512: 32, 768: 64, 1024: 128, 1280: 128}

# Real active params (lm_head, no embed) for display
real_active_params = {512: 86.3e6, 768: 160.2e6, 1024: 282.0e6, 1280: 442.1e6}


def total_steps(dim, tok_ratio):
    tokens = tok_ratio * active_params[dim]
    bs = batch_sizes[dim]
    return max(1, round(tokens / (bs * SEQ_LEN)))


# Only use finished runs (state == 'finished'), deduplicate by (dim, tok_ratio, adam_lr)
# Prefer v12-retry over v10/v11 when both exist
best_runs = {}
for r in results:
    if r.get("state") != "finished":
        continue
    if not (r["dim"] and r["tok_ratio"] and r["adam_lr"] and r["bpb"]):
        continue
    # Drop outlier high-LR points
    if r["dim"] == 1024 and r["tok_ratio"] == 50.0 and r["adam_lr"] >= 0.0088:
        continue
    if r["dim"] == 1280 and r["tok_ratio"] == 22.0 and r["adam_lr"] >= 0.0072:
        continue
    if r["dim"] == 1280 and r["tok_ratio"] == 50.0 and r["adam_lr"] >= 0.0064:
        continue
    if r["dim"] == 1280 and r["tok_ratio"] == 10.0 and r["adam_lr"] >= 0.0088:
        continue
    key = (r["dim"], r["tok_ratio"], r["adam_lr"])
    is_retry = "v12-retry" in r.get("group", "")
    prev = best_runs.get(key)
    if prev is None or (is_retry and not prev[1]):
        best_runs[key] = (r, is_retry)

complete = [v[0] for v in best_runs.values()]
print(f"Using {len(complete)} finished deduplicated runs")

# Group by (dim, tok_ratio)
grouped = defaultdict(list)
for r in complete:
    grouped[(r["dim"], r["tok_ratio"])].append(r)


# FLOPs per token (excluding lm_head) for each dim
def compute_fpt(hidden_dim):
    num_layers = hidden_dim // 128 * 2
    num_heads = hidden_dim // 128
    num_kv_heads = num_heads
    intermediate_dim = hidden_dim // 2
    num_experts_per_tok = 4
    shared_ffn = intermediate_dim

    attn_per_layer = 2 * hidden_dim * (hidden_dim + hidden_dim)
    attn_per_layer += 2 * hidden_dim * (num_kv_heads * (hidden_dim // num_heads))
    attn_per_layer += 2 * hidden_dim * (num_kv_heads * (hidden_dim // num_heads))

    moe_mlp_per_layer = num_experts_per_tok * 3 * 2 * hidden_dim * intermediate_dim
    shared_mlp_per_layer = 3 * 2 * hidden_dim * shared_ffn

    per_layer = attn_per_layer + moe_mlp_per_layer + shared_mlp_per_layer
    fpt_no_lm_head = num_layers * per_layer
    return fpt_no_lm_head


dim_fpt = {d: compute_fpt(d) for d in [512, 768, 1024, 1280]}
dim_colors = {512: "C0", 768: "C1", 1024: "C2", 1280: "C3"}
group_markers = {
    "isoflop-moe-v10": ("o", "C0"),
    "isoflop-moe-v11": ("s", "C3"),
    "isoflop-moe-v12": ("^", "C2"),
}


USE_MIN_POINT = {
    (512, 22.0),
    (512, 50.0),
    (768, 4.5),
    (768, 10.0),
    (768, 22.0),
    (768, 50.0),
    (1024, 2.0),
    (1024, 4.5),
    (1024, 10.0),
    (1024, 50.0),
}


def get_optimal_lr(points, dim=None, tr=None):
    if not points:
        return None
    if len(points) < 3:
        # Not enough points for a fit, use minimum
        min_idx = np.argmin([p[1] for p in points])
        return points[min_idx][0], points[min_idx][1], None
    lrs = np.array([p[0] for p in points])
    bpbs = np.array([p[1] for p in points])
    # Fit quadratic in log(lr) space: bpb = a*log(lr)^2 + b*log(lr) + c
    log_lrs = np.log(lrs)
    coeffs = np.polyfit(log_lrs, bpbs, 2)
    if coeffs[0] <= 0:
        # Fallback to minimum point
        min_idx = np.argmin(bpbs)
        return lrs[min_idx], bpbs[min_idx], None
    opt_log_lr = -coeffs[1] / (2 * coeffs[0])
    opt_lr = np.exp(opt_log_lr)
    opt_bpb = np.polyval(coeffs, opt_log_lr)
    if opt_lr < lrs.min() * 0.5 or opt_lr > lrs.max() * 2:
        # Optimum outside reasonable range, use minimum
        min_idx = np.argmin(bpbs)
        return lrs[min_idx], bpbs[min_idx], None
    return opt_lr, opt_bpb, coeffs


dims = sorted(set(r["dim"] for r in complete))
tok_ratios = sorted(set(r["tok_ratio"] for r in complete))

# ============================================================
# Plot 1: BPB vs Adam LR grid
# ============================================================
fig, axes = plt.subplots(
    len(dims), len(tok_ratios), figsize=(4 * len(tok_ratios), 3.5 * len(dims)), squeeze=False, sharex=False, sharey=False
)
for i, dim in enumerate(dims):
    for j, tr in enumerate(tok_ratios):
        ax = axes[i][j]
        pts = grouped.get((dim, tr), [])
        if not pts:
            ax.set_visible(False)
            continue

        for r in pts:
            ax.plot(r["adam_lr"], r["bpb"], "o", color="C0", markersize=5, alpha=0.7)

        lrs = [r["adam_lr"] for r in pts]
        bpbs = [r["bpb"] for r in pts]
        if len(lrs) >= 3:
            log_lrs = np.log(lrs)
            coeffs = np.polyfit(log_lrs, bpbs, 2)
            lr_fine = np.geomspace(min(lrs) * 0.8, max(lrs) * 1.2, 100)
            ax.plot(lr_fine, np.polyval(coeffs, np.log(lr_fine)), "k--", alpha=0.3)
            if coeffs[0] > 0:
                opt_log = -coeffs[1] / (2 * coeffs[0])
                opt = np.exp(opt_log)
                ax.axvline(opt, color="green", alpha=0.5, linewidth=1)

        real_tr = tr * active_params[dim] / real_active_params[dim]
        ax.set_title(f"d{dim} {real_tr:.1f}x", fontsize=9)
        if j == 0:
            ax.set_ylabel("c4_en/bpb")
        if i == len(dims) - 1:
            ax.set_xlabel("Adam LR")

fig.suptitle("BPB vs Adam LR (log-space quadratic fit)", fontsize=12)
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}/v3_all_dims_c4bpb_vs_lr.png", dpi=150)
plt.close(fig)
print("Saved v3_all_dims_c4bpb_vs_lr.png")

# ============================================================
# Plot 2: Optimal LR and BPB vs token ratio
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for dim in dims:
    opt_lrs = []
    opt_bpbs = []
    trs = []
    for tr in tok_ratios:
        pts_list = [(r["adam_lr"], r["bpb"]) for r in grouped.get((dim, tr), [])]
        res = get_optimal_lr(pts_list, dim=dim, tr=tr)
        if res:
            opt_lrs.append(res[0])
            opt_bpbs.append(res[1])
            trs.append(tr)
    if trs:
        ax1.plot(trs, opt_lrs, "o-", label=f"d{dim}", color=dim_colors.get(dim, "gray"))
        ax2.plot(trs, opt_bpbs, "o-", label=f"d{dim}", color=dim_colors.get(dim, "gray"))

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("Token Ratio")
ax1.set_ylabel("Optimal Adam LR")
ax1.legend()
ax1.set_title("Optimal LR vs Token Ratio")
ax1.grid(True, alpha=0.3)

ax2.set_xscale("log")
ax2.set_xlabel("Token Ratio")
ax2.set_ylabel("Best c4_en/bpb")
ax2.legend()
ax2.set_title("Best BPB vs Token Ratio")
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(f"{PLOT_DIR}/v3_opt_lr_and_bpb_vs_tokens.png", dpi=150)
plt.close(fig)
print("Saved v3_opt_lr_and_bpb_vs_tokens.png")

# ============================================================
# Plot 3: Optimal LR vs Compute (all points)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

all_C = []
all_lr_norm = []
for dim in dims:
    for tr in tok_ratios:
        pts_list = [(r["adam_lr"], r["bpb"]) for r in grouped.get((dim, tr), [])]
        res = get_optimal_lr(pts_list, dim=dim, tr=tr)
        if not res:
            continue
        opt_lr = res[0]
        sample = grouped[(dim, tr)][0]
        bs = sample["batch_size"] or batch_sizes[dim]
        tokens = tr * active_params[dim]
        C = 3 * dim_fpt[dim] * tokens
        lr_norm = opt_lr / math.sqrt(bs / 32)
        all_C.append(C)
        all_lr_norm.append(lr_norm)
        ax.plot(C, lr_norm, "o", color=dim_colors.get(dim, "gray"), markersize=8)

if len(all_C) >= 2:
    log_C = np.log(all_C)
    log_lr = np.log(all_lr_norm)
    slope, intercept = np.polyfit(log_C, log_lr, 1)
    C_fine = np.logspace(np.log10(min(all_C)) - 0.2, np.log10(max(all_C)) + 0.2, 100)
    ax.plot(C_fine, np.exp(intercept) * C_fine**slope, "k--", alpha=0.5, label=f"fit: C^({slope:.3f})")
    ax.legend()

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Compute (FLOPs, excl lm_head)")
ax.set_ylabel("Optimal Adam LR (normalized to bs=32)")
ax.set_title("Optimal LR vs Compute (all points)")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}/v3_opt_lr_vs_compute_all.png", dpi=150)
plt.close(fig)
print("Saved v3_opt_lr_vs_compute_all.png")

# ============================================================
# Plot 4: Optimal LR vs Compute (filtered)
# ============================================================
skip = set()  # Include all dims/token ratios

fig, ax = plt.subplots(figsize=(8, 6))

filt_C = []
filt_lr_norm = []
for dim in dims:
    for tr in tok_ratios:
        if (dim, tr) in skip:
            continue
        pts_list = [(r["adam_lr"], r["bpb"]) for r in grouped.get((dim, tr), [])]

        res = get_optimal_lr(pts_list, dim=dim, tr=tr)
        if not res:
            continue
        opt_lr = res[0]

        sample = grouped[(dim, tr)][0]
        bs = sample["batch_size"] or batch_sizes[dim]
        tokens = tr * active_params[dim]
        C = 3 * dim_fpt[dim] * tokens
        lr_norm = opt_lr / math.sqrt(bs / 32)
        filt_C.append(C)
        filt_lr_norm.append(lr_norm)
        ax.plot(C, lr_norm, "o", color=dim_colors.get(dim, "gray"), markersize=8)

if len(filt_C) >= 2:
    log_C = np.log(filt_C)
    log_lr = np.log(filt_lr_norm)
    slope, intercept = np.polyfit(log_C, log_lr, 1)
    C_fine = np.logspace(np.log10(min(filt_C)) - 0.2, np.log10(max(filt_C)) + 0.2, 100)
    ax.plot(C_fine, np.exp(intercept) * C_fine**slope, "k--", alpha=0.5, label=f"fit: C^({slope:.3f})")
    ax.legend()

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Compute (FLOPs, excl lm_head)")
ax.set_ylabel("Optimal Adam LR (normalized to bs=32)")
ax.set_title("Optimal LR vs Compute (filtered)")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}/v3_opt_lr_vs_compute_filtered.png", dpi=150)
plt.close(fig)
print("Saved v3_opt_lr_vs_compute_filtered.png")

# ============================================================
# Plot 5: LR = A * tokens^tx * dim^dx fit (3 panels)
# ============================================================
fit_data = []
for dim in dims:
    for tr in tok_ratios:
        if (dim, tr) in skip:
            continue
        pts_list = [(r["adam_lr"], r["bpb"]) for r in grouped.get((dim, tr), [])]
        res = get_optimal_lr(pts_list, dim=dim, tr=tr)
        if not res:
            continue
        opt_lr = res[0]

        sample = grouped[(dim, tr)][0]
        bs = sample["batch_size"] or batch_sizes[dim]
        tokens = tr * active_params[dim]
        lr_norm = opt_lr / math.sqrt(bs / 32)
        fit_data.append((dim, tokens, lr_norm))

if len(fit_data) >= 3:
    dims_arr = np.array([d[0] for d in fit_data])
    tokens_arr = np.array([d[1] for d in fit_data])
    lr_arr = np.array([d[2] for d in fit_data])

    X = np.column_stack([np.ones(len(fit_data)), np.log(tokens_arr), np.log(dims_arr)])
    coeffs_fit = np.linalg.lstsq(X, np.log(lr_arr), rcond=None)[0]
    A = np.exp(coeffs_fit[0])
    tx = coeffs_fit[1]
    dx = coeffs_fit[2]

    lr_pred = A * tokens_arr**tx * dims_arr**dx
    ss_res = np.sum((np.log(lr_arr) - np.log(lr_pred)) ** 2)
    ss_tot = np.sum((np.log(lr_arr) - np.mean(np.log(lr_arr))) ** 2)
    r2 = 1 - ss_res / ss_tot

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    for dim in sorted(set(dims_arr)):
        mask = dims_arr == dim
        ax1.plot(
            tokens_arr[mask],
            lr_arr[mask],
            "o",
            color=dim_colors.get(int(dim), "gray"),
            label=f"d{int(dim)}",
            markersize=8,
        )
        t_fine = np.logspace(np.log10(tokens_arr[mask].min()) - 0.2, np.log10(tokens_arr[mask].max()) + 0.2, 100)
        ax1.plot(t_fine, A * t_fine**tx * dim**dx, "--", color=dim_colors.get(int(dim), "gray"), alpha=0.4)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Tokens")
    ax1.set_ylabel("Optimal Adam LR (bs=32)")
    ax1.set_title(f"lr = {A:.1f} * tokens^({tx:.3f}) * dim^({dx:.3f})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    lr_collapsed = lr_arr * dims_arr ** (-dx)
    for dim in sorted(set(dims_arr)):
        mask = dims_arr == dim
        ax2.plot(
            tokens_arr[mask],
            lr_collapsed[mask],
            "o",
            color=dim_colors.get(int(dim), "gray"),
            label=f"d{int(dim)}",
            markersize=8,
        )

    t_fine = np.logspace(np.log10(tokens_arr.min()) - 0.2, np.log10(tokens_arr.max()) + 0.2, 100)
    ax2.plot(t_fine, A * t_fine**tx, "k--", alpha=0.5, label=f"tokens^({tx:.3f})")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Tokens")
    ax2.set_ylabel(f"LR * dim^({-dx:.3f})")
    ax2.set_title("Collapsed by dimension")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.plot(lr_pred, lr_arr, "o", markersize=8)
    lims = [min(lr_pred.min(), lr_arr.min()) * 0.8, max(lr_pred.max(), lr_arr.max()) * 1.2]
    ax3.plot(lims, lims, "k--", alpha=0.3)
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("Predicted LR")
    ax3.set_ylabel("Actual Optimal LR")
    ax3.set_title(f"R² = {r2:.3f}")
    ax3.grid(True, alpha=0.3)

    fig.suptitle(f"lr = {A * 32**-0.5:.2f} * tokens^({tx:.4f}) * dim^({dx:.4f}) * bs^(0.5)", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/v3_lr_tokens_dim_fit.png", dpi=150)
    plt.close(fig)
    print(f"Saved v3_lr_tokens_dim_fit.png  (A={A:.2f}, tx={tx:.4f}, dx={dx:.4f}, R²={r2:.3f})")
else:
    print("Not enough data for tokens+dim fit")

print("\nDone!")

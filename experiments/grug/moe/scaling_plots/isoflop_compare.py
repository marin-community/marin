# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Isoflop bowls: May Recipe vs v16 baseline, paloma macro_loss vs tokens.

Self-contained — no wandb, no JSON file. NEW_DATA is the four finished
isoflop sweeps (`grug-moe-isoflop-v{18,3e18,1e19,3e19}` in wandb project
`marin-community/marin_moe`). OLD_DATA is the v16 baseline sweep
(`isoflop-moe-v16` in `marin-community/dial_moe`).

Each tuple: (budget_FLOPs, hidden_dim, tokens, paloma_macro_loss). Per
isoflop bowl we fit a parabola in log(tokens), draw it as the bowl curve,
place a star at the parabola minimum, and arrow from the v16 minimum to
the May Recipe minimum.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "isoflop_compare.png")
SKIP_OLD_DIMS = {640, 896}

BUDGET_COLORS = {
    1e18: "#1f77b4",
    3e18: "#ff7f0e",
    1e19: "#2ca02c",
    3e19: "#d62728",
}

NEW_DATA: list[tuple[float, int, int, float]] = [
    (1e18, 512, 3771465728, 3.392345428466797),
    (1e18, 768, 1577058304, 3.402113437652588),
    (1e18, 1024, 729808896, 3.4711685180664062),
    (3e18, 512, 11314135040, 3.2693166732788086),
    (3e18, 768, 4730650624, 3.220895767211914),
    (3e18, 1024, 2189426688, 3.240917921066284),
    (3e18, 1280, 1301282816, 3.2983484268188477),
    (1e19, 512, 37717278720, 3.1842541694641113),
    (1e19, 768, 15770583040, 3.073707342147827),
    (1e19, 1024, 7298613248, 3.0422251224517822),
    (1e19, 1280, 4337434624, 3.0651612281799316),
    (3e19, 512, 113151836160, 3.1245718002319336),
    (3e19, 768, 47314894848, 2.9688315391540527),
    (3e19, 1024, 21895315456, 2.9105122089385986),
    (3e19, 1280, 13014401024, 2.9044876098632812),
]

OLD_DATA: list[tuple[float, int, int, float]] = [
    (1e18, 512, 3822845952, 3.5396740436553955),
    (1e18, 640, 2272788480, 3.5221519470214844),
    (1e18, 768, 1594884096, 3.5272915363311768),
    (1e18, 896, 969670656, 3.5876283645629883),
    (1e18, 1024, 736886784, 3.623152256011963),
    (1e18, 1280, 437518336, 3.748671770095825),
    (1e18, 1536, 261095424, 3.939781665802002),
    (1e18, 1792, 181141504, 4.121220588684082),
    (3e18, 512, 11468275712, 3.414754629135132),
    (3e18, 640, 6818627584, 3.367424249649048),
    (3e18, 768, 4784914432, 3.3397984504699707),
    (3e18, 896, 2908880896, 3.3458549976348877),
    (3e18, 1024, 2210660352, 3.3475966453552246),
    (3e18, 1280, 1312423936, 3.405738353729248),
    (3e18, 1536, 783417344, 3.509922504425049),
    (3e18, 1792, 543424512, 3.618996381759643),
    (3e18, 2048, 364904448, 3.737143039703369),
    (1e19, 512, 38227935232, 3.332810401916504),
    (1e19, 640, 22728933376, 3.2483696937561035),
    (1e19, 768, 15949365248, 3.209399700164795),
    (1e19, 896, 9696182272, 3.164015769958496),
    (1e19, 1024, 7368736768, 3.149415969848633),
    (1e19, 1280, 4374528000, 3.162015199661255),
    (1e19, 1536, 2611609600, 3.201606035232544),
    (1e19, 1792, 1811152896, 3.2618567943573),
    (1e19, 2048, 1216479232, 3.329171657562256),
    (3e19, 640, 68185751552, 3.1705727577209473),
    (3e19, 768, 47848620032, 3.114365577697754),
    (3e19, 896, 29088546816, 3.051757335662842),
    (3e19, 1024, 22106079232, 3.015672206878662),
    (3e19, 1280, 13123715072, 3.0079565048217773),
    (3e19, 1536, 7834697728, 3.006631135940552),
    (3e19, 1792, 5433589760, 3.037367582321167),
    (3e19, 2048, 3649437696, 3.070441722869873),
]


def _group(rows, skip_dims=()):
    out: dict[float, list[tuple[int, int, float]]] = {}
    for b, d, tok, l in rows:
        if d in skip_dims:
            continue
        out.setdefault(b, []).append((d, tok, l))
    for b in out:
        out[b].sort()
    return out


def _parabola_fit(xs, ys):
    lx = np.log(xs)
    coeffs = np.polyfit(lx, ys, deg=2)
    smooth_lx = np.linspace(lx.min(), lx.max(), 100)
    a, b, _c = coeffs
    opt_lx = -b / (2 * a)
    return np.exp(smooth_lx), np.polyval(coeffs, smooth_lx), float(np.exp(opt_lx)), float(np.polyval(coeffs, opt_lx))


def main() -> None:
    new = _group(NEW_DATA)
    old = _group(OLD_DATA, skip_dims=SKIP_OLD_DIMS)
    shared = sorted(set(new) & set(old))

    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(12, 7.5))

    for budget in shared:
        color = BUDGET_COLORS.get(budget, "#888")

        old_opt = None
        v16 = old.get(budget, [])
        if len(v16) >= 3:
            xs = [tok for _, tok, _ in v16]
            ys = [l for _, _, l in v16]
            fx, fy, ox, oy = _parabola_fit(xs, ys)
            ax.plot(fx, fy, "--", color=color, lw=1.3, alpha=0.55)
            ax.scatter(xs, ys, s=42, facecolors="white", edgecolors=color, lw=1.3, zorder=3)
            ax.scatter([ox], [oy], marker="*", s=260, facecolors="white", edgecolors=color, lw=1.6, zorder=5)
            old_opt = (ox, oy)

        new_opt = None
        n = new.get(budget, [])
        if len(n) >= 3:
            xs = [tok for _, tok, _ in n]
            ys = [l for _, _, l in n]
            fx, fy, ox, oy = _parabola_fit(xs, ys)
            ax.plot(fx, fy, "-", color=color, lw=2.0, alpha=0.95)
            ax.scatter(xs, ys, s=85, color=color, edgecolors="black", lw=0.5, zorder=4)
            ax.scatter([ox], [oy], marker="*", s=320, color=color, edgecolors="black", lw=0.8, zorder=6)
            new_opt = (ox, oy)

        if old_opt and new_opt:
            ax.annotate(
                "",
                xy=new_opt,
                xytext=old_opt,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.6, alpha=0.85, shrinkA=8, shrinkB=8),
            )

    for budget in shared:
        color = BUDGET_COLORS.get(budget, "#888")
        ax.plot([], [], "-", color=color, lw=2, label=f"{budget:.0e} FLOPs")
    ax.plot([], [], "-", color="#222", lw=2, label="May Recipe (solid, filled)")
    ax.plot([], [], "--", color="#222", lw=1.3, alpha=0.6, label="v16 baseline (dashed, hollow)")

    ax.set_xscale("log")
    ax.set_xlabel("tokens", fontsize=13)
    ax.set_ylabel("paloma macro_loss", fontsize=13)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title("Isoflop bowls: May Recipe vs v16 baseline (tokens x loss at fixed compute)", fontsize=14)
    ax.legend(loc="upper right", fontsize=10, ncol=2)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot the LSH banding S-curves for the two bandings we actually ran.

P(two docs with true Jaccard J become a candidate) = 1 - (1 - J^r)^b,
where a band is r signature slots and there are b bands (b*r = num_perms).
Shows why 286x26 is a ~0.8 dedup banding and 284x71 is a ~0.3 high-recall scan.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "plots/lsh_scurve_bandings.png"

# (rows r, bands b, label, color). r*b = num_perms (~286/284).
BANDINGS = [
    (11, 26, "286×26  (r=11, b=26) — DEDUP banding (the run we did)", "#1f77b4"),  # noqa: RUF001
    (4, 71, "284×71  (r=4,  b=71) — high-recall SCAN banding", "#d62728"),  # noqa: RUF001
]
PROBE_J = [0.3, 0.5, 0.7, 0.8, 0.9]


def p_candidate(j: np.ndarray, r: int, b: int) -> np.ndarray:
    return 1.0 - (1.0 - j**r) ** b


def threshold(r: int, b: int) -> float:
    """Exact J where P(candidate) = 0.5."""
    return (1.0 - 0.5 ** (1.0 / b)) ** (1.0 / r)


def main() -> None:
    j = np.linspace(0.0, 1.0, 2001)
    fig, ax = plt.subplots(figsize=(11, 6.5))

    for r, b, label, color in BANDINGS:
        ax.plot(j, p_candidate(j, r, b), color=color, lw=2.4, label=label)
        thr = threshold(r, b)
        ax.axvline(thr, color=color, ls=":", lw=1.3, alpha=0.7)
        ax.annotate(
            f"50% @ J≈{thr:.2f}",
            xy=(thr, 0.5),
            xytext=(thr + 0.015, 0.53),
            color=color,
            fontsize=9,
            fontweight="bold",
        )

    # Probe values: dotted verticals + a small table of P for each banding.
    rows = []
    for jv in PROBE_J:
        rows.append([f"{jv:.1f}"] + [f"{p_candidate(np.array([jv]), r, b)[0]:.3f}" for r, b, *_ in BANDINGS])
    table = ax.table(
        cellText=rows,
        colLabels=["true J", "286×26", "284×71"],  # noqa: RUF001
        cellLoc="center",
        colLoc="center",
        bbox=[0.62, 0.12, 0.35, 0.42],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    ax.set_xlabel("true Jaccard J between two documents", fontsize=11)
    ax.set_ylabel("P(become a candidate) = 1 − (1 − Jʳ)ᵇ", fontsize=11)  # noqa: RUF001
    ax.set_title(
        "LSH banding S-curves — the two bandings we ran\n"
        "the (r, b) split is the only knob: it slides where the step sits",
        fontsize=12,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT}")
    for r, b, label, *_ in BANDINGS:
        print(f"  {label.split('—')[0].strip()}: 50% threshold J≈{threshold(r, b):.3f}")


if __name__ == "__main__":
    main()

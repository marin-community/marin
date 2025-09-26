# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from collections.abc import Iterable

import numpy as np

try:
    from marin.utilities.wandb_utils import WANDB_ENTITY, WANDB_PROJECT
except Exception:
    WANDB_ENTITY = os.getenv("WANDB_ENTITY", "stanford-mercury")
    WANDB_PROJECT = os.getenv("WANDB_PROJECT", "marin")

# Reuse run collection and tag parsing from the existing utilities
from experiments.regmix.plot_results import (
    MIXTURE_TAG_KEYS,
    collect_regmix_summary,
)


def _flatten(values: Iterable[float]) -> np.ndarray:
    arr = np.array(list(values), dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch W&B runs and plot histograms of rewrite mixture proportions "
            "for each technique (mcq, nemo_qa, regular_text, wrap_med, wrap_qa)."
        )
    )
    parser.add_argument("--entity", default=WANDB_ENTITY, help="W&B entity")
    parser.add_argument("--project", default=WANDB_PROJECT, help="W&B project")
    parser.add_argument("--api_key", default=os.getenv("WANDB_API_KEY"), help="W&B API key")

    # Selection logic mirrors the strategy used elsewhere in regmix utilities
    parser.add_argument(
        "--require_tags",
        action="append",
        default=["regmix", "llama-130m", "lr=0.006"],
        help="Required tags to include runs (repeatable)",
    )
    parser.add_argument(
        "--also_accept_tag",
        action="append",
        default=["130m"],
        help="Fallback tags to accept if some required tags are missing (repeatable)",
    )
    parser.add_argument(
        "--name_contains",
        default="llama-130m-regmix-v3",
        help="Substring to match in run names (filter)",
    )

    # Plot controls
    parser.add_argument("--bins", type=int, default=20, help="Number of histogram bins")
    parser.add_argument("--logy", action="store_true", help="Use log scale on the y-axis")
    parser.add_argument(
        "--save_dir",
        default=os.path.join("experiments", "regmix", "plots"),
        help="Directory to save figures. Created if missing.",
    )
    parser.add_argument("--show", action="store_true", help="Display plots interactively as well")

    args = parser.parse_args()

    rows = collect_regmix_summary(
        entity=args.entity,
        project=args.project,
        api_key=args.api_key,
        require_tags=list(args.require_tags or []),
        also_accept_tag=list(args.also_accept_tag or []),
        name_contains=args.name_contains,
    )

    if not rows:
        raise SystemExit("No runs found for plotting mixtures.")

    # Prepare output directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Collect values per technique
    key_to_values: dict[str, list[float]] = {k: [] for k in MIXTURE_TAG_KEYS}
    for r in rows:
        mixture = r["mixture"]
        for k in MIXTURE_TAG_KEYS:
            v = float(mixture.get(k, 0.0))
            # Clamp just in case of tiny floating errors
            if v < 0.0:
                v = 0.0
            if v > 1.0:
                v = 1.0
            key_to_values[k].append(v)

    # Plot one histogram per technique
    import matplotlib.pyplot as plt

    for k in MIXTURE_TAG_KEYS:
        values = _flatten(key_to_values[k])
        if values.size == 0:
            # Skip if empty (shouldn't happen if rows exist)
            continue

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(values, bins=args.bins, range=(0.0, 1.0), edgecolor="black")
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Weight value")
        ax.set_ylabel("Frequency")
        title = f"Distribution of weights for {k}"
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        if args.logy:
            ax.set_yscale("log")

        out_path = os.path.join(args.save_dir, f"hist_{k}.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved {out_path}")

    if args.show:
        # Optionally, create a combined figure for quick viewing
        n_keys = len(MIXTURE_TAG_KEYS)
        n_cols = 3
        n_rows = int(np.ceil(n_keys / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3.5))
        axes = np.atleast_1d(axes).ravel()
        for idx, k in enumerate(MIXTURE_TAG_KEYS):
            ax = axes[idx]
            values = _flatten(key_to_values[k])
            ax.hist(values, bins=args.bins, range=(0.0, 1.0), edgecolor="black")
            ax.set_xlim(0.0, 1.0)
            ax.set_title(k)
            if args.logy:
                ax.set_yscale("log")
        for j in range(n_keys, len(axes)):
            axes[j].axis("off")
        fig.suptitle("Rewrite mixture distributions")
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

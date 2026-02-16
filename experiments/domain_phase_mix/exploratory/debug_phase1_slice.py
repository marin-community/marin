# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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

# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "matplotlib"]
# ///
"""Filter two_phase_starcoder.csv to phase_0_nemotron_full=1.0 and plot phase_1_starcoder vs programming BPB."""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

script_dir = Path(__file__).parent
df = pd.read_csv(script_dir / "two_phase_starcoder_combined.csv")
df = df[df["status"] == "completed"]

# Filter to rows where phase_0 is 100% nemotron (phase_0_starcoder ~ 0)
mask = df["phase_0_nemotron_full"].round(4) == 1.0
df_slice = df[mask].sort_values("phase_1_starcoder")

print(f"Total completed runs: {len(df)}")
print(f"Runs with phase_0_nemotron_full=1.0: {len(df_slice)}")
print()

target = "eval/paloma/dolma_100_programing_languages/bpb"
print(f"phase_1_starcoder vs {target}:")
for _, row in df_slice.iterrows():
    print(f"  p1_sc={row['phase_1_starcoder']:.4f}  bpb={row[target]:.4f}")

# Find minimum
best_idx = df_slice[target].idxmin()
best_row = df_slice.loc[best_idx]
print(f"\nMinimum: p1_sc={best_row['phase_1_starcoder']:.4f}  bpb={best_row[target]:.4f}")

# Epoch calculation for StarCoder in phase 1:
#   epochs = phase_1_starcoder * TARGET_BUDGET * phase_1_fraction / STARCODER_TOKENS
TARGET_BUDGET = 5_729_908_864_777  # Nemotron full token count
STARCODER_TOKENS = 216_567_300_822  # StarCoder token count
PHASE_1_FRACTION = 0.5
EPOCH_MULTIPLIER = TARGET_BUDGET * PHASE_1_FRACTION / STARCODER_TOKENS  # ~13.2

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df_slice["phase_1_starcoder"], df_slice[target], s=40, zorder=5)
ax.plot(df_slice["phase_1_starcoder"], df_slice[target], alpha=0.3, linestyle="--")
ax.set_xlabel("phase_1_starcoder weight")
ax.set_ylabel(target)
ax.set_title(f"phase_0 = 100% nemotron_full (n={len(df_slice)})\n{target}")
ax.axvline(
    best_row["phase_1_starcoder"],
    color="red",
    alpha=0.4,
    linestyle=":",
    label=f"min @ {best_row['phase_1_starcoder']:.3f} ({best_row['phase_1_starcoder'] * EPOCH_MULTIPLIER:.1f} epochs)",
)
ax.legend()

# Secondary x-axis showing StarCoder epochs
secax = ax.secondary_xaxis("top", functions=(lambda w: w * EPOCH_MULTIPLIER, lambda e: e / EPOCH_MULTIPLIER))
secax.set_xlabel("StarCoder epochs in phase 1")

fig.tight_layout()

out_path = script_dir / "debug_phase1_slice.png"
fig.savefig(out_path, dpi=300)
print(f"\nSaved to {out_path}")

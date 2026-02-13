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
# dependencies = ["pandas"]
# ///
"""Add epoch columns to two_phase_starcoder.csv.

Computes how many times each domain is repeated (epoched) in each phase,
based on the simulated epoching setup in two_phase_starcoder_experiment.py:

  epochs_d_phase_p = phase_fraction * target_budget * w_{p,d} / N_d

where:
  - target_budget = 5,729,908,864,777 (Nemotron token count)
  - N_nemotron = 5,729,908,864,777
  - N_starcoder = 216,567,300,822
  - phase_fraction = 0.5 (each phase is half the training)
  - w_{p,d} = domain weight in phase p
"""

import pandas as pd
from pathlib import Path

# Constants from two_phase_starcoder_experiment.py and domains.py
TARGET_BUDGET = 5_729_908_864_777
N_NEMOTRON = 5_729_908_864_777
N_STARCODER = 216_567_300_822
PHASE_FRACTION = 0.5  # Two equal phases

# Epoch multipliers: epochs = multiplier * weight
NEMOTRON_EPOCH_MULT = PHASE_FRACTION * TARGET_BUDGET / N_NEMOTRON  # = 0.5
STARCODER_EPOCH_MULT = PHASE_FRACTION * TARGET_BUDGET / N_STARCODER  # ≈ 13.22

print(f"Nemotron epoch multiplier per phase: {NEMOTRON_EPOCH_MULT:.4f}")
print(f"StarCoder epoch multiplier per phase: {STARCODER_EPOCH_MULT:.4f}")

script_dir = Path(__file__).parent
csv_path = script_dir / "two_phase_starcoder.csv"
df = pd.read_csv(csv_path)
print(f"\nLoaded {len(df)} rows from {csv_path.name}")

# Per-phase epoch counts
df["phase_0_nemotron_epochs"] = NEMOTRON_EPOCH_MULT * df["phase_0_nemotron_full"]
df["phase_0_starcoder_epochs"] = STARCODER_EPOCH_MULT * df["phase_0_starcoder"]
df["phase_1_nemotron_epochs"] = NEMOTRON_EPOCH_MULT * df["phase_1_nemotron_full"]
df["phase_1_starcoder_epochs"] = STARCODER_EPOCH_MULT * df["phase_1_starcoder"]

# Total epochs across both phases
df["total_nemotron_epochs"] = df["phase_0_nemotron_epochs"] + df["phase_1_nemotron_epochs"]
df["total_starcoder_epochs"] = df["phase_0_starcoder_epochs"] + df["phase_1_starcoder_epochs"]

# Save
df.to_csv(csv_path, index=False)
print(f"Saved with epoch columns to {csv_path.name}")

# Show some examples
completed = df[df["status"] == "completed"]
print("\nEpoch ranges (completed runs):")
for col in ["phase_0_starcoder_epochs", "phase_1_starcoder_epochs", "total_starcoder_epochs", "total_nemotron_epochs"]:
    vals = completed[col]
    print(f"  {col}: [{vals.min():.3f}, {vals.max():.3f}]")

# Show the best observed run
target = "eval/paloma/dolma_100_programing_languages/bpb"
best_idx = completed[target].idxmin()
best = completed.loc[best_idx]
print(f"\nBest run for {target} (bpb={best[target]:.4f}):")
print(f"  p0_sc={best['phase_0_starcoder']:.4f}, p1_sc={best['phase_1_starcoder']:.4f}")
print(
    f"  sc_epochs: phase_0={best['phase_0_starcoder_epochs']:.3f}, phase_1={best['phase_1_starcoder_epochs']:.3f}, total={best['total_starcoder_epochs']:.3f}"
)
print(
    f"  nem_epochs: phase_0={best['phase_0_nemotron_epochs']:.3f}, phase_1={best['phase_1_nemotron_epochs']:.3f}, total={best['total_nemotron_epochs']:.3f}"
)

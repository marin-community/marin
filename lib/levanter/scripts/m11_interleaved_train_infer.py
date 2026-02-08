#!/usr/bin/env python
# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""M11 orchestrator: train 2 steps -> inference 128 prompts -> train 5 more steps.

Runs each phase as a separate subprocess to avoid TPU runtime launch-group
contamination that occurs when switching between training and inference JIT
programs within a single process.

Launch on TPU via:
    python infra/launch.py --foreground --zone us-central1-a \
        --tpu_name simpo_worker --tpu_type v5p-32 --capacity_type on-demand -- \
        python scripts/m11_interleaved_train_infer.py
"""
import subprocess
import sys
import time


def run_phase(name: str, command: list[str]) -> None:
    print(f"\n{'='*60}")
    print(f"  Phase: {name}")
    print(f"  Command: {' '.join(command)}")
    print(f"{'='*60}\n", flush=True)

    t0 = time.time()
    result = subprocess.run(command)
    elapsed = time.time() - t0

    if result.returncode != 0:
        raise RuntimeError(f"Phase '{name}' failed with exit code {result.returncode} after {elapsed:.1f}s")
    print(f"\n=== Phase '{name}' completed in {elapsed:.1f}s ===\n", flush=True)


def main() -> None:
    py = sys.executable

    # Phase 1: Train 2 steps, save HF + orbax checkpoint
    run_phase(
        "train_phase1",
        [
            py,
            "src/levanter/main/train_simpo.py",
            "--config_path",
            "config/simpo_ultrafeedback_llama3_8b_v5p_32_m11_subprocess_phase1.yaml",
        ],
    )

    # Phase 2: Host-data-parallel inference from HF checkpoint (128 prompts x 2048 tokens)
    run_phase(
        "inference",
        [
            py,
            "src/levanter/main/sample_lm_multihost.py",
            "--config_path",
            "config/sampler/sample_llama8b_v5p_32_m11_subprocess_phase2.yaml",
        ],
    )

    # Phase 3: Resume training from orbax checkpoint for 5 more steps (step 2 -> step 7)
    run_phase(
        "train_phase3",
        [
            py,
            "src/levanter/main/train_simpo.py",
            "--config_path",
            "config/simpo_ultrafeedback_llama3_8b_v5p_32_m11_subprocess_phase3.yaml",
        ],
    )

    print("\n" + "=" * 60)
    print("  M11 interleaved train -> infer -> train COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

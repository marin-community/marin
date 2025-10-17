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

"""Shared helpers for memorization experiments (150M Llama on COMMA).

This module factors common setup across region/seed-set scripts:
- Build 150M model config w/ seq_len=4096
- Construct COMMA mixtures for 1M (wikimedia), 10M, 100M seed sets
- Create P(z) eval config aligned with levanter configs
- Create a `_mk_run` helper that auto-computes P(z) cadence (~1% of steps)
"""

from __future__ import annotations

import dataclasses
from datetime import datetime

from experiments.common_pile.tokenize_common_pile import (
    COMMA_MAIN_MIXTURE_WEIGHTS,
    common_pile_tokenized,
)
from experiments.llama import llama3_tokenizer, llama_150m
from experiments.simple_train_config import SimpleTrainConfig
from levanter.eval_pz_innerloop import PzInnerLoopConfig
from marin.execution.executor import ExecutorStep
from marin.processing.tokenize.data_configs import lm_mixture_data_config
from marin.resources import TpuPodConfig
from experiments.defaults import default_train

# Reasonable TPU defaults by region (can be adjusted per environment)
REGION_TPU_DEFAULTS: dict[str, str] = {
    "central1": "v5p-8",
    "east5": "v5p-8",
    "central2": "v4-64",
    # us-east1-d small slice used in 1M Wikimedia runs
    "east1d": "v6e-8",
}


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_llama150m_memorize_config():
    """150M Llama w/ seq_len=4096 used across all memorize experiments."""
    return dataclasses.replace(llama_150m, seq_len=4096)


def make_pz_eval_config(initial_batch_size: int) -> PzInnerLoopConfig:
    """P(z) configuration aligned with levanter YAML defaults (fast + informative).

    Key choices:
    - Evaluate across all datasets in the mixture (datasets=None) and ~1% of training steps.
    - doc length limited to 1024 for speed; chunk_size/prompt tokens as in YAML.
    - eval_batch_size 64 and num_documents 1000 (as in referenced config).
    """
    return PzInnerLoopConfig(
        datasets=None,  # include all datasets in the mixture
        mode="first",
        num_documents=1000,
        eval_batch_size=64,
        doc_tokens=1024,
        chunk_size=100,
        prompt_tokens=50,
        cursor_inc_tokens=16,
        histogram=False,
        pz_npz=False,
        decode_preview=1,
        verify_treecache=False,
        # Enforce subset + deterministic shuffle strictly within training head
        restrict_to_training_subset=True,
        initial_batch_size=initial_batch_size,
        doc_shuffle=True,
        doc_perm_type="feistel",
        doc_perm_seed=0,
    )


def make_comma_mixture_1m_wikimedia() -> tuple[object, int]:
    """Return (mixture, seed_set_batches) for ~1M token seed (Wikimedia only).

    - 2 batches from Wikimedia only
    - Seed set batches = 2
    """
    tokenized_all = common_pile_tokenized(tokenizer=llama3_tokenizer)
    # Only include wikimedia dataset to avoid downloading/tokenizing unused datasets
    tokenized = {ds: step for ds, step in tokenized_all.items() if "wikimedia" in ds}
    max_train_batches = {ds: 2 for ds in tokenized}
    weights = {ds: w for ds, w in COMMA_MAIN_MIXTURE_WEIGHTS.items() if ds in tokenized}
    mixture = lm_mixture_data_config(
        components=tokenized,
        weights=weights,
        max_train_batches=max_train_batches,
        shuffle=False,
    )
    # Enable Feistel PRP and per-epoch reshuffling on wrap
    mixture = dataclasses.replace(mixture, permutation_type="feistel", shuffle_per_epoch=True)
    return mixture, 2


def make_comma_mixture_10m() -> tuple[object, int]:
    """Return (mixture, seed_set_batches) for ~10M token seed (1 batch per dataset)."""
    tokenized = common_pile_tokenized(tokenizer=llama3_tokenizer)
    max_train_batches = {ds: 1 for ds in tokenized}
    mixture = lm_mixture_data_config(
        components=tokenized,
        weights=COMMA_MAIN_MIXTURE_WEIGHTS,
        max_train_batches=max_train_batches,
        shuffle=False,
    )
    # Enable Feistel PRP and per-epoch reshuffling on wrap
    mixture = dataclasses.replace(mixture, permutation_type="feistel", shuffle_per_epoch=True)
    # 15 datasets x 1 batch each → 15 steps/epoch
    return mixture, 15


def make_comma_mixture_100m() -> tuple[object, int]:
    """Return (mixture, seed_set_batches) for ~100M token seed.

    Assign 13 batches to datasets with positive weight; 0 to zero-weight datasets
    (e.g., public_domain_review) to avoid over-requesting sequences from tiny sets.
    Seed set size is the sum of per-dataset batches.
    """
    tokenized_all = common_pile_tokenized(tokenizer=llama3_tokenizer)
    # Exclude zero and ultra-tiny weight datasets (main stage sets public_domain_review=1e-5)
    # Exclude ultra-tiny components (e.g., public_domain_review=1e-5, python_enhancement_proposals=2e-5)
    MIN_MAIN_WEIGHT = 5e-5
    components = {
        ds: step for ds, step in tokenized_all.items() if COMMA_MAIN_MIXTURE_WEIGHTS.get(ds, 0.0) > MIN_MAIN_WEIGHT
    }
    weights = {ds: w for ds, w in COMMA_MAIN_MIXTURE_WEIGHTS.items() if ds in components}

    # 13 batches per positive-weight dataset
    max_train_batches = {ds: 13 for ds in components}

    mixture = lm_mixture_data_config(
        components=components,
        weights=weights,
        max_train_batches=max_train_batches,
        shuffle=False,
    )
    # Enable Feistel PRP and per-epoch reshuffling on wrap
    mixture = dataclasses.replace(mixture, permutation_type="feistel", shuffle_per_epoch=True)

    seed_set_batches = 13 * len(components)
    return mixture, seed_set_batches


def mk_run(
    *,
    region: str,
    name_prefix: str,
    epochs: int,
    seed_set_batches: int,
    train_batch_size: int = 128,
    model_config,
    mixture,
    timestamp: str | None = None,
) -> ExecutorStep:
    """Create a training ExecutorStep with auto P(z) cadence and sensible defaults.

    - P(z) cadence: ~1% of total steps, rounded, min 1
    - TPU type: chosen via REGION_TPU_DEFAULTS
    - Batch size: 128, steps_per_eval: 1000, max_eval_batches: 10
    - Tags incorporate seed size, region, and model size
    """
    tpu_type = REGION_TPU_DEFAULTS.get(region, "v5p-8")
    pz_steps = max(1, int((seed_set_batches * epochs) / 100 + 0.5))
    z_loss = 0

    # Derive seed size tag (e.g., 1M, 10M, 100M) from the name prefix
    _prefix_tail = name_prefix.split("/")[-1]
    _parts = _prefix_tail.split("_")
    seed_tag = next((p for p in _parts if p.endswith("M") and p[:-1].isdigit()), None)

    # Include dataset qualifier if present in prefix (e.g., wikimedia)
    dataset_tag = "wikimedia" if any(p == "wikimedia" for p in _parts) else None

    tags = [
        "memorize",
        "comma",
        "150m",
    ]
    if seed_tag:
        tags.append(seed_tag)
    if dataset_tag:
        tags.append(dataset_tag)
    tags.extend([region, tpu_type])

    # Add timestamp to run name for unique identification
    # Allow caller to override to support resuming pre-empted runs by name.
    ts = timestamp or now_timestamp()

    # Build P(z) config first to allow consistency checks
    pz_cfg = make_pz_eval_config(train_batch_size)

    # Consistency checks: enforce shuffle=False for training, and ensure max_train_batches covers P(z) datasets
    # mixture is a LMMixtureDatasetConfig
    if getattr(mixture, "shuffle", False) not in (False, 0):
        raise ValueError("Memorize runs require data.shuffle=False to align P(z) subset with training subset.")

    if pz_cfg.datasets is not None and getattr(mixture, "max_train_batches", None) is not None:
        mtb = mixture.max_train_batches
        for ds in pz_cfg.datasets:
            if ds in mixture.configs:
                if ds not in mtb or int(mtb[ds]) < 1:
                    raise ValueError(
                        f"P(z) dataset {ds} must have max_train_batches >= 1 in the training config to guarantee subset alignment."
                    )

    # HARD GUARDRAILS: These options desynchronize P(z) selection from the training subset head.
    # We do not support them in memorize experiments because P(z) must sample strictly from the
    # same head that training uses (as defined by max_train_batches × initial_batch_size).
    if getattr(mixture, "experiment_budget", None) is not None or getattr(mixture, "target_budget", None) is not None:
        raise ValueError(
            "Memorize runs must NOT set experiment_budget/target_budget. These change the effective training subset "
            "used by mixtures and will cause P(z) to sample from a different head than training."
        )

    if getattr(mixture, "num_validation_sequences", None) is not None:
        raise ValueError(
            "Memorize runs must NOT set num_validation_sequences. Holding out a tail for validation changes the "
            "training subset definition and breaks P(z) alignment with training."
        )

    return default_train(
        name=f"{name_prefix}_{epochs}epoch_{region}_{ts}",
        tokenized=mixture,
        model_config=model_config,
        train_config=SimpleTrainConfig(
            resources=TpuPodConfig(tpu_type=tpu_type, slice_count=1),
            train_batch_size=train_batch_size,
            num_train_steps=seed_set_batches * epochs,
            learning_rate=0.003,
            weight_decay=0.0,
            beta1=0.9,
            beta2=0.95,
            lr_schedule="cosine",
            warmup=0.01,
            min_lr_ratio=0.0,
            z_loss_weight=z_loss,
            steps_per_eval=1000,
            max_eval_batches=10,
            steps_per_task_eval=None,
            seed=0,
        ),
        tags=tags,
        eval_harness_tasks=(),
        pz_eval_config=pz_cfg,
        pz_eval_steps=pz_steps,
    )


# -------------------------------
# Region-bound runners
# -------------------------------


def make_runner_1m_wikimedia(region: str):
    """Return a callable runner(epochs) for ~1M Wikimedia-only seed set."""
    model_config = build_llama150m_memorize_config()
    mixture, seed_batches = make_comma_mixture_1m_wikimedia()
    name_prefix = "memorize/comma_150m_1M_wikimedia"

    def _runner(epochs: int, *, timestamp: str | None = None) -> ExecutorStep:
        return mk_run(
            region=region,
            name_prefix=name_prefix,
            epochs=epochs,
            seed_set_batches=seed_batches,
            model_config=model_config,
            mixture=mixture,
            timestamp=timestamp,
        )

    return _runner


def make_runner_10m(region: str):
    """Return a callable runner(epochs) for ~10M seed set."""
    model_config = build_llama150m_memorize_config()
    mixture, seed_batches = make_comma_mixture_10m()
    name_prefix = "memorize/comma_150m_10M"

    def _runner(epochs: int, *, timestamp: str | None = None) -> ExecutorStep:
        return mk_run(
            region=region,
            name_prefix=name_prefix,
            epochs=epochs,
            seed_set_batches=seed_batches,
            model_config=model_config,
            mixture=mixture,
            timestamp=timestamp,
        )

    return _runner


def make_runner_100m(region: str):
    """Return a callable runner(epochs) for ~100M seed set.

    Uses z_loss=1e-4 regardless of epoch count to mirror prior practice.
    """
    model_config = build_llama150m_memorize_config()
    mixture, seed_batches = make_comma_mixture_100m()
    name_prefix = "memorize/comma_150m_100M"

    def _runner(epochs: int, *, timestamp: str | None = None) -> ExecutorStep:
        return mk_run(
            region=region,
            name_prefix=name_prefix,
            epochs=epochs,
            seed_set_batches=seed_batches,
            model_config=model_config,
            mixture=mixture,
            timestamp=timestamp,
        )

    return _runner

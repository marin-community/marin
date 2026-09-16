# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Prespecified three-curve experiment at ten tokens per total parameter."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import equinox as eqx
import haliax as hax
import jax
import numpy as np
import requests
from levanter.data.dataset import AsyncDataset
from levanter.data.mixture import MixtureDataset
from levanter.models.qwen import Qwen3Config
from levanter.tokenizers import MarinTokenizer, load_tokenizer

from experiments.domain_phase_mix.starcoder_epoch_matching import canonical_sha256, file_sha256

ASSETS = Path(__file__).with_name("starcoder_tpp10_assets")
TOKENIZER = "experiments/domain_phase_mix/starcoder_tpp10_assets"
DESIGN_PATH = ASSETS / "design.json"
PREFIX = "gs://marin-us-central1"
REGION = "us-central1"
ZONE = "us-central1-a"
VERSION = "2026.09.09"
SEQ_LEN = 2048
BATCH_SIZE = 128
PROXY_BATCH_SIZE = 32
BLOCK_SIZE = 2048
VOCAB_SIZE = 32000
DATA_SEED = 20260910
TRAINER_SEEDS = (20260910, 20260911)
SUBSET_SEEDS = (20260912, 20260913, 20260914)
PARENT_ORDER_SEED = 20260915
PILOT_GRID = (0, 10, 30, 50, 70, 90, 100)
DENSE_GRID = tuple(range(0, 101, 5))
PARENT_SEQUENCES = 726 * BATCH_SIZE
MATCHED_SEQUENCES = 40 * BATCH_SIZE
PRIMARY_METRIC = "eval/paloma/dolma_100_programing_languages-tpp10/bpb"
WEB_COUNTS = {
    "hq_actual": 537_620_495_374,
    "hq_synth": 1_497_529_159_716,
    "medium_high": 489_053_720_257,
    "medium": 1_960_603_657_130,
    "medium_low": 860_999_424_951,
    "low_actual": 384_102_407_349,
}


def require_central1() -> None:
    """Check the execution host before regional payload reads or writes."""
    response = requests.get(
        "http://metadata.google.internal/computeMetadata/v1/instance/zone",
        headers={"Metadata-Flavor": "Google"},
        timeout=5,
    )
    response.raise_for_status()
    if response.text.rsplit("/", 1)[-1] != ZONE:
        raise ValueError("TPP10 execution must run on a us-central1-a host")
    if os.environ.get("MARIN_PREFIX") != PREFIX:
        raise ValueError("MARIN_PREFIX must be explicitly set to gs://marin-us-central1")


class Arm(StrEnum):
    TARGET = "target"
    UNMATCHED = "unmatched"
    MATCHED = "matched"


@dataclass(frozen=True)
class RunSpec:
    run_name: str
    arm: Arm
    percent: int
    trainer_seed: int
    subset_seed: int | None
    total_steps: int
    batch_size: int

    @property
    def tokens(self) -> int:
        return self.total_steps * self.batch_size * SEQ_LEN

    @property
    def support_sequences(self) -> int:
        return MATCHED_SEQUENCES if self.arm == Arm.MATCHED else PARENT_SEQUENCES


def model_config(arm: Arm) -> Qwen3Config:
    width, layers = (1024, 16) if arm == Arm.TARGET else (256, 8)
    return Qwen3Config(
        max_seq_len=SEQ_LEN,
        hidden_dim=width,
        intermediate_dim=4 * width,
        num_layers=layers,
        num_heads=width // 128,
        num_kv_heads=width // 128,
        tie_word_embeddings=True,
        tokenizer=TOKENIZER,
    )


def total_parameters(model: Qwen3Config) -> int:
    """Count actual model leaves, including Qwen3's two per-head RMSNorm weights."""
    shape = eqx.filter_eval_shape(lambda: model.build(hax.Axis("vocab", VOCAB_SIZE), key=jax.random.PRNGKey(0)))
    return sum(leaf.size for leaf in jax.tree_util.tree_leaves(shape) if isinstance(leaf, jax.ShapeDtypeStruct))


def mixture_weights(percent: int) -> dict[str, float]:
    if percent not in DENSE_GRID:
        raise ValueError("Mixture coordinate is outside the frozen grid")
    p = percent / 100
    total = sum(WEB_COUNTS.values())
    return {**{name: (1 - p) * n / total for name, n in WEB_COUNTS.items()}, "starcoder": p}


def subset_indices(seed: int, *, parent_count: int = PARENT_SEQUENCES, count: int = MATCHED_SEQUENCES) -> np.ndarray:
    """Uniform sample without replacement; sorted indices preserve parent order for I/O."""
    return np.sort(np.random.Generator(np.random.PCG64(seed)).choice(parent_count, size=count, replace=False)).astype(
        "<i8"
    )


def parent_permutation() -> np.ndarray:
    return np.random.Generator(np.random.PCG64(PARENT_ORDER_SEED)).permutation(PARENT_SEQUENCES).astype("<i8")


def verified_tokenizer() -> MarinTokenizer:
    """Check the bundled path and counted vocabulary without initializing a JAX backend."""
    if Path(TOKENIZER).resolve() != ASSETS.resolve():
        raise ValueError("Run from the repository root so the bundled tokenizer path resolves locally")
    tokenizer = load_tokenizer(TOKENIZER)
    if len(tokenizer) != VOCAB_SIZE:
        raise ValueError("Bundled tokenizer vocabulary differs from the counted model vocabulary")
    return tokenizer


def web_sequences(target_tokens: int) -> dict[str, int]:
    total = sum(WEB_COUNTS.values())
    return {name: math.ceil(1.2 * target_tokens * n / total / SEQ_LEN) + BLOCK_SIZE for name, n in WEB_COUNTS.items()}


def build_design() -> dict:
    models = {}
    for arm in (Arm.UNMATCHED, Arm.TARGET):
        model = model_config(arm)
        n = total_parameters(model)
        # Keep the reviewed horizons, rounded in 128-sequence units, across the batch screen.
        tokens = round(10 * n / (BATCH_SIZE * SEQ_LEN)) * BATCH_SIZE * SEQ_LEN
        batch_size = BATCH_SIZE if arm == Arm.TARGET else PROXY_BATCH_SIZE
        steps = tokens // (batch_size * SEQ_LEN)
        models[arm.value] = {
            "parameters": n,
            "steps": steps,
            "tokens": tokens,
            "tpp": tokens / n,
            "batch_size": batch_size,
            "training_flops": 3 * model.flops_per_token(VOCAB_SIZE, SEQ_LEN) * tokens,
        }
    runs = []
    for percent in DENSE_GRID:
        runs.append(
            RunSpec(
                f"tpp10_target_p{percent:03d}_s{TRAINER_SEEDS[0]}",
                Arm.TARGET,
                percent,
                TRAINER_SEEDS[0],
                None,
                models["target"]["steps"],
                BATCH_SIZE,
            )
        )
        for seed in TRAINER_SEEDS:
            runs.append(
                RunSpec(
                    f"tpp10_unmatched_p{percent:03d}_s{seed}",
                    Arm.UNMATCHED,
                    percent,
                    seed,
                    None,
                    models["unmatched"]["steps"],
                    PROXY_BATCH_SIZE,
                )
            )
            if percent:
                for subset_seed in SUBSET_SEEDS:
                    runs.append(
                        RunSpec(
                            f"tpp10_matched_p{percent:03d}_s{seed}_d{subset_seed}",
                            Arm.MATCHED,
                            percent,
                            seed,
                            subset_seed,
                            models["unmatched"]["steps"],
                            PROXY_BATCH_SIZE,
                        )
                    )
    calibration = [
        RunSpec(
            row.run_name + "_b128",
            row.arm,
            row.percent,
            row.trainer_seed,
            row.subset_seed,
            row.total_steps // 4,
            BATCH_SIZE,
        )
        for row in runs
        if row.percent == 50 and row.arm != Arm.TARGET and row.subset_seed in (None, SUBSET_SEEDS[0])
    ]
    pins = json.loads((ASSETS / "pins.json").read_text())
    import_sources = (Path(__file__), Path(__file__).with_name("starcoder_epoch_matching.py"))
    design = {
        "schema_version": 1,
        "version": VERSION,
        "primary_metric": PRIMARY_METRIC,
        "models": models,
        "tokenizer_pins": pins,
        "source_code_sha256": {p.name: file_sha256(p) for p in import_sources},
        "subset_indices_sha256": {
            str(seed): canonical_sha256({"indices": subset_indices(seed).tolist()}) for seed in SUBSET_SEEDS
        },
        "parent_permutation_sha256": canonical_sha256({"indices": parent_permutation().tolist()}),
        "parent_sequences": PARENT_SEQUENCES,
        "matched_sequences": MATCHED_SEQUENCES,
        "web_sequences": web_sequences(models["target"]["tokens"]),
        "pilot_grid": list(PILOT_GRID),
        "dense_grid": list(DENSE_GRID),
        "runs": [asdict(row) for row in runs],
        "calibration_runs": [asdict(row) for row in calibration],
    }
    return {**design, "design_sha256": canonical_sha256(design)}


def load_design(path: Path = DESIGN_PATH) -> dict:
    design = json.loads(path.read_text())
    if design != build_design():
        raise ValueError(
            "Frozen design, tokenizer pins, source code, or model geometry changed; create a new reviewed design"
        )
    for name, digest in design["tokenizer_pins"]["files_sha256"].items():
        if file_sha256(ASSETS / name) != digest:
            raise ValueError(f"Pinned asset changed: {name}")
    verified_tokenizer()
    return design


def write_design(design: dict, path: Path) -> None:
    """Create a frozen design or verify an identical existing one; never replace another design."""
    if path.exists():
        if json.loads(path.read_text()) != design:
            raise ValueError("Existing frozen design differs; review and version the experiment instead of overwriting")
        return
    with path.open("x") as handle:
        handle.write(json.dumps(design, indent=2) + "\n")


def validate_plan(plan: dict) -> None:
    if canonical_sha256({k: v for k, v in plan.items() if k != "plan_sha256"}) != plan["plan_sha256"]:
        raise ValueError("Submission plan checksum mismatch")


def select_runs(design: dict, stage: str) -> tuple[RunSpec, ...]:
    if stage not in ("calibration", "canary", "pilot", "dense"):
        raise ValueError(f"Unknown stage: {stage}")
    rows = tuple(RunSpec(**{**row, "arm": Arm(row["arm"])}) for row in design["runs"])
    if stage == "calibration":
        base = tuple(
            row
            for row in rows
            if row.percent == 50 and row.arm != Arm.TARGET and row.subset_seed in (None, SUBSET_SEEDS[0])
        )
        return base + tuple(RunSpec(**{**row, "arm": Arm(row["arm"])}) for row in design["calibration_runs"])
    if stage == "canary":
        return tuple(
            row
            for row in rows
            if row.percent == 100 and row.trainer_seed == TRAINER_SEEDS[0] and row.subset_seed in (None, SUBSET_SEEDS[0])
        )
    if stage == "pilot":
        return tuple(row for row in rows if row.percent in PILOT_GRID)
    return rows


def calibration_summary(plan: dict, values: dict[str, float]) -> dict:
    """Apply the fixed-mixture batch screen using only unmatched losses for release."""
    if plan["stage"] != "calibration":
        raise ValueError("Batch comparison needs the frozen calibration stage")
    means = {}
    for arm in ("unmatched", "matched"):
        means[arm] = {}
        for batch in (32, 128):
            losses = [values[r["run_name"]] for r in plan["runs"] if r["arm"] == arm and r["batch_size"] == batch]
            if len(losses) != len(TRAINER_SEEDS) or not all(math.isfinite(v) for v in losses):
                raise ValueError("Calibration requires both finite trainer-seed endpoints in every cell")
            means[arm][str(batch)] = math.fsum(losses) / len(losses)
    gap = means["unmatched"]["32"] - means["unmatched"]["128"]
    return {
        "batch_means": means,
        "unmatched_batch32_minus_batch128_bpb": gap,
        "batch32_loss_screen_passed": gap <= 0.01,
        "criterion": (
            "Review a new recipe if unmatched mean BPB at batch32 exceeds batch128 by more than 0.01. "
            "Matched losses are diagnostics; they do not select the optimizer recipe."
        ),
    }


class TaggedIndices(AsyncDataset[tuple[str, int]]):
    """Finite sequence identities for audits without loading training tokens."""

    def __init__(self, name: str, count: int):
        self.name = name
        self.count = count

    def is_finite(self) -> bool:
        return True

    async def async_len(self) -> int:
        return self.count

    async def get_batch(self, indices: Sequence[int]) -> Sequence[tuple[str, int]]:
        if any(i < 0 or i >= self.count for i in indices):
            raise IndexError("Sequence exceeds finite source")
        return [(self.name, int(i)) for i in indices]


async def sequence_allocation(percent: int, total_steps: int, batch_size: int) -> dict[str, int]:
    """Count the real block allocator, including the final partial block, without token reads."""
    weights = mixture_weights(percent)
    mix_key, _ = jax.random.split(jax.random.PRNGKey(DATA_SEED))
    n = total_steps * batch_size
    datasets = {name: TaggedIndices(name, n + BLOCK_SIZE) for name, w in weights.items() if w > 0}
    mixture = MixtureDataset(datasets, weights, BLOCK_SIZE, key=mix_key)
    full, remainder = divmod(n, BLOCK_SIZE)
    first = Counter(name for name, _ in await mixture.get_batch(range(BLOCK_SIZE)))
    counts = Counter({name: count * full for name, count in first.items()})
    if remainder:
        counts.update(name for name, _ in await mixture.get_batch(range(full * BLOCK_SIZE, n)))
    return {name: counts[name] for name in weights}


async def audit_allocations(design: dict) -> dict:
    rows: list[dict] = []
    for p in DENSE_GRID:
        proxy = await sequence_allocation(p, design["models"]["unmatched"]["steps"], PROXY_BATCH_SIZE)
        target = await sequence_allocation(p, design["models"]["target"]["steps"], BATCH_SIZE)
        if proxy["starcoder"] > PARENT_SEQUENCES:
            raise ValueError("Unmatched StarCoder repeats")
        for allocation in (proxy, target):
            for name, count in allocation.items():
                if name != "starcoder" and count > design["web_sequences"][name]:
                    raise ValueError(f"Web component repeats: {name}")
        ep, et = proxy["starcoder"] / MATCHED_SEQUENCES, target["starcoder"] / PARENT_SEQUENCES
        relative_error = abs(ep / et - 1) if et else 0.0
        if relative_error > 0.01:
            raise ValueError(f"Materialized epoch mismatch exceeds 1% at p={p}")
        rows.append(
            {
                "percent": p,
                "proxy_allocation": proxy,
                "target_allocation": target,
                "matched_epochs": ep,
                "target_epochs": et,
                "relative_error": relative_error,
            }
        )
    return {
        "design_sha256": design["design_sha256"],
        "status": "passed",
        "maximum_epoch_relative_error": max(row["relative_error"] for row in rows),
        "coordinates": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-design", action="store_true")
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()
    if args.write_design:
        write_design(build_design(), DESIGN_PATH)
    design = load_design()
    if args.audit_output:
        args.audit_output.write_text(json.dumps(asyncio.run(audit_allocations(design)), indent=2) + "\n")
    print(
        json.dumps(
            {
                "design_sha256": design["design_sha256"],
                "models": design["models"],
                "runs": {s: len(select_runs(design, s)) for s in ("canary", "pilot", "dense")},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train matched Grug MoE controls over tokenizer-sweep caches.

This recipe intentionally contains no chunked-MoE behavior. It reproduces the
historical AdamH d512 and d768 Grug rungs while varying only tokenizer family
and vocabulary size.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from urllib.parse import urlparse

import draccus
import fsspec
from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.step_runner import StepRunner
from marin.experiment.data import mixture
from marin.experiment.namespacing import user_namespaced_name
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import LevanterCheckpoint

from experiments.grug.moe.launch import GrugMoeLaunchConfig, run_grug_moe_trial
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.optimizer import GrugMoeAdamHConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

TOKENIZER_FAMILIES = (
    "gpt-oss",
    "llama",
    "gpt-oss-place-digits",
    "llama-place-digits",
)
DIGITS_FAMILIES = ("gpt-oss-place-digits", "llama-place-digits")
VOCAB_SIZES = (8_192, 32_768)
HIDDEN_DIMS = (512, 768)
SEQ_LEN = 4_096
BATCH_SIZE = 512

_MARIN_BUCKET_REGIONS = {
    "marin-eu-west4": "europe-west4",
    "marin-us-central1": "us-central1",
    "marin-us-central2": "us-central2",
    "marin-us-east5": "us-east5",
    "marin-us-west4": "us-west4",
}


@dataclass(frozen=True)
class GrugRung:
    hidden_dim: int
    num_layers: int
    intermediate_dim: int
    steps: int
    tokens: int
    learning_rate: float
    adam_lr: float
    epsilon: float
    beta2: float


GRUG_RUNGS = {
    512: GrugRung(
        hidden_dim=512,
        num_layers=6,
        intermediate_dim=256,
        steps=399,
        tokens=836_763_648,
        learning_rate=0.0498025161390265,
        adam_lr=0.011492888339775346,
        epsilon=1.932779486232198e-16,
        beta2=0.9841194418156399,
    ),
    768: GrugRung(
        hidden_dim=768,
        num_layers=8,
        intermediate_dim=384,
        steps=1_292,
        tokens=2_709_520_384,
        learning_rate=0.03082765217648045,
        adam_lr=0.007114073579187796,
        epsilon=3.477980290225924e-16,
        beta2=0.9841194418156399,
    ),
}


@dataclass(frozen=True)
class TokenizerMoeComparisonConfig:
    """Inputs for a matched tokenizer-comparison sweep."""

    cache_prefix: str
    output_prefix: str
    tokenizer_run_id: str
    region: str
    tpu_type: str
    version: str
    families: list[str] = field(default_factory=lambda: list(TOKENIZER_FAMILIES))
    vocab_sizes: list[int] = field(default_factory=lambda: list(VOCAB_SIZES))
    hidden_dims: list[int] = field(default_factory=lambda: list(HIDDEN_DIMS))
    train_cache_label: str = "train50b"
    expected_sources: int = 115
    max_concurrent: int = 8
    preemptible: bool = True
    dry_run: bool = False

    def validate(self) -> None:
        unknown_families = set(self.families) - set(TOKENIZER_FAMILIES)
        if unknown_families:
            raise ValueError(f"unknown tokenizer families: {sorted(unknown_families)}")
        unknown_vocab_sizes = set(self.vocab_sizes) - set(VOCAB_SIZES)
        if unknown_vocab_sizes:
            raise ValueError(f"unsupported vocab sizes: {sorted(unknown_vocab_sizes)}")
        unknown_hidden_dims = set(self.hidden_dims) - set(GRUG_RUNGS)
        if unknown_hidden_dims:
            raise ValueError(f"unsupported Grug rungs: {sorted(unknown_hidden_dims)}")
        if self.expected_sources <= 0:
            raise ValueError("expected_sources must be positive")
        validate_same_region(self.cache_prefix, self.output_prefix, self.region)


@dataclass(frozen=True)
class ComparisonCell:
    family: str
    vocab_size: int
    rung: GrugRung

    @property
    def name(self) -> str:
        return f"{self.family}-{self.vocab_size // 1024}k-d{self.rung.hidden_dim}"


def comparison_cells(config: TokenizerMoeComparisonConfig) -> list[ComparisonCell]:
    """Return the requested Cartesian product after validating the recipe."""
    config.validate()
    return [
        ComparisonCell(family=family, vocab_size=vocab_size, rung=GRUG_RUNGS[hidden_dim])
        for family in config.families
        for vocab_size in config.vocab_sizes
        for hidden_dim in config.hidden_dims
    ]


def grug_model(cell: ComparisonCell) -> GrugModelConfig:
    """Build the historical AdamH Grug rung, changing only vocabulary size."""
    hidden_dim = cell.rung.hidden_dim
    num_heads = hidden_dim // 128
    return GrugModelConfig(
        vocab_size=cell.vocab_size,
        hidden_dim=hidden_dim,
        intermediate_dim=cell.rung.intermediate_dim,
        shared_expert_intermediate_dim=hidden_dim,
        num_experts=64,
        num_experts_per_token=4,
        num_layers=cell.rung.num_layers,
        num_heads=num_heads,
        num_kv_heads=max(1, num_heads // 4),
        max_seq_len=SEQ_LEN,
        sliding_window=SEQ_LEN,
        initializer_std=0.5 / math.sqrt(hidden_dim),
        qk_mult=1.3,
        disable_pko=False,
        disable_long_rope=False,
    )


def grug_optimizer(rung: GrugRung) -> GrugMoeAdamHConfig:
    """Return the frozen v16 AdamH optimizer settings for one rung."""
    return GrugMoeAdamHConfig(
        learning_rate=rung.learning_rate,
        adam_lr=rung.adam_lr,
        min_lr_ratio=0.0,
        warmup=0.1,
        beta1=0.9062,
        beta2=rung.beta2,
        epsilon=rung.epsilon,
        max_grad_norm=1.0,
        lr_schedule="linear",
        decay=None,
    )


def validate_same_region(cache_prefix: str, output_prefix: str, region: str) -> None:
    """Reject cross-region data reads and checkpoint writes before submission."""
    cache_bucket = _gcs_bucket(cache_prefix)
    output_bucket = _gcs_bucket(output_prefix)
    if cache_bucket != output_bucket:
        raise ValueError(
            f"cache bucket {cache_bucket!r} and output bucket {output_bucket!r} must match; "
            "cross-region tokenizer comparison runs are not allowed"
        )
    bucket_region = _MARIN_BUCKET_REGIONS.get(cache_bucket)
    if bucket_region is None:
        raise ValueError(f"unknown region for Marin bucket {cache_bucket!r}")
    compute_region = _base_region(region)
    if compute_region != bucket_region:
        raise ValueError(
            f"TPU region {compute_region!r} does not match cache bucket {cache_bucket!r} in {bucket_region!r}"
        )


def _gcs_bucket(path: str) -> str:
    parsed = urlparse(path)
    if parsed.scheme != "gs" or not parsed.netloc:
        raise ValueError(f"expected a gs:// path, got {path!r}")
    return parsed.netloc


def _base_region(location: str) -> str:
    parts = location.rsplit("-", 1)
    if len(parts) == 2 and len(parts[1]) == 1 and parts[1].isalpha():
        return parts[0]
    return location


def _completed_cache_dirs(root: str, expected_sources: int) -> list[tuple[str, str, float]]:
    fs, root_path = fsspec.core.url_to_fs(root)
    try:
        entries = sorted(fs.ls(root_path, detail=False))
    except FileNotFoundError as exc:
        raise ValueError(f"tokenized cache root does not exist: {root}") from exc

    completed = []
    for entry in entries:
        source_name = os.path.basename(entry.rstrip("/"))
        if source_name.startswith("."):
            continue
        stats_path = f"{entry.rstrip('/')}/train/.stats.json"
        if not fs.exists(stats_path):
            continue
        with fs.open(stats_path) as stats_file:
            stats = json.load(stats_file)
        total_tokens = float(stats.get("total_tokens") or 1.0)
        completed.append((source_name, fs.unstrip_protocol(entry), total_tokens))

    if len(completed) != expected_sources:
        raise ValueError(f"expected {expected_sources} completed cache sources under {root}, found {len(completed)}")
    return completed


def _adopt_caches(
    *,
    config: TokenizerMoeComparisonConfig,
    cell: ComparisonCell,
    label: str,
    root: str,
) -> tuple[dict[ArtifactStep[TokenizedCache], float], list[ArtifactStep[TokenizedCache]]]:
    tokenizer = f"marin-community/{config.tokenizer_run_id}-{cell.family}-{cell.vocab_size // 1024}k"
    entries = _completed_cache_dirs(root, config.expected_sources)
    handles = []
    weights = {}
    for source_name, source, total_tokens in entries:
        artifact_name = (
            f"data/tokenizer-comparison/{config.tokenizer_run_id}/{label}/"
            f"{cell.family}-{cell.vocab_size // 1024}k/{source_name}"
        )
        handle = ArtifactStep.adopt(
            name=artifact_name,
            version=config.version,
            source=source,
            kind=TokenizedCache,
            config={
                "tokenizer": tokenizer,
                "format": {"text_key": "text"},
                "tags": [label, source_name],
            },
        )
        handles.append(handle)
        weights[handle] = total_tokens
    return weights, handles


def comparison_run(
    config: TokenizerMoeComparisonConfig,
    cell: ComparisonCell,
) -> ArtifactStep[LevanterCheckpoint]:
    """Build one tokenizer/rung training checkpoint."""
    cache_base = config.cache_prefix.rstrip("/")
    cache_name = f"{cell.family}-{cell.vocab_size // 1024}k"
    train, _ = _adopt_caches(
        config=config,
        cell=cell,
        label="train",
        root=f"{cache_base}/{config.train_cache_label}/{cache_name}",
    )
    _, holdout = _adopt_caches(
        config=config,
        cell=cell,
        label="holdout",
        root=f"{cache_base}/{cache_name}",
    )
    resources = ResourceConfig.with_tpu(
        config.tpu_type,
        regions=[config.region],
        preemptible=config.preemptible,
    )

    def build_config(ctx: StepContext) -> GrugMoeLaunchConfig:
        return GrugMoeLaunchConfig(
            model=grug_model(cell),
            data=mixture(ctx, train, validation=holdout),
            output_path=ctx.output_path,
            run_id=f"tokenizer-moe-{cell.name}-{config.version}",
            resources=ctx.runtime_arg("train_resources"),
            steps=cell.rung.steps,
            batch_size=BATCH_SIZE,
            seed=0,
            mp="params=float32,compute=bfloat16,output=bfloat16",
            tracker=WandbConfig(
                project="marin_moe",
                tags=["tokenizer-comparison", cell.family, f"{cell.vocab_size // 1024}k", f"d{cell.rung.hidden_dim}"],
                group=f"tokenizer-moe-{config.version}",
                name=None,
            ),
            optimizer=grug_optimizer(cell.rung),
            grug_trainer=GrugTrainerConfig(
                z_loss_weight=1e-4,
                ema_beta=None,
                log_every=1,
                expert_axis_size=4,
            ),
            eval=GrugEvalConfig(
                eval_batch_size=512,
                steps_per_eval=1_000,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
                compute_bpb=True,
            ),
        )

    return ArtifactStep(
        name=user_namespaced_name(f"tokenizer-comparison/{cell.name}", config.version),
        version=config.version,
        artifact_type=LevanterCheckpoint,
        run=run_grug_moe_trial,
        build_config=build_config,
        deps=(*train, *holdout),
        runtime_args={"train_resources": resources},
    )


def main() -> None:
    config = draccus.parse(TokenizerMoeComparisonConfig)
    config.validate()
    os.environ["MARIN_PREFIX"] = config.output_prefix
    runs = [comparison_run(config, cell).lower() for cell in comparison_cells(config)]
    StepRunner().run(runs, dry_run=config.dry_run, max_concurrent=config.max_concurrent)


if __name__ == "__main__":
    main()

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grug-MoE training harness with simulated epoching over the testbed mixture.

``run_testbed_config`` wraps ``experiments.grug.moe.launch.run_grug_moe_trial``
and applies simulated epoching via ``target_budget``/``experiment_budget`` on
the ``LmDataConfig`` — Grug doesn't go through the generic ``train_lm`` assembler,
so the testbed routes through Grug's own training entry point.

The RFC's canonical testbed dataset targets ~1T raw tokens sampled
proportionally by provenance. At training time we pick a much smaller
compute-optimal ``experiment_budget`` (batch * steps * seq_len) and let
``MixtureDataset`` slice each component by ``experiment_budget / target_budget``
to preserve the per-source shares over the shortened horizon (see
``levanter/data/text/datasets.py:763-770``).
"""

import dataclasses
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.lazy import Dataset, lower, materialized_config
from marin.execution.step_spec import StepSpec
from marin.processing.tokenize import TokenizeConfig, lm_mixture_data_config, tokenize
from rigging.filesystem import marin_prefix

from experiments.datakit.testbed.mixture import read_bucket_weights
from experiments.datakit.testbed.settings import TESTBED_SEQ_LEN, TESTBED_TOKENIZER
from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import GrugMoeLaunchConfig, run_grug_moe_trial
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

logger = logging.getLogger(__name__)


# RFC target: 1T raw tokens sampled proportionally across the testbed.
DEFAULT_TARGET_BUDGET_TOKENS: int = 1_000_000_000_000

# Grug MoE baseline — matches the values used by ``experiments/grug/moe/launch.py``.
# We reuse them so the testbed ranking protocol starts at the same compute point as
# the existing Grug runs.
DEFAULT_COMPUTE_BUDGET_FLOPS: float = 1e18
DEFAULT_HIDDEN_DIM: int = 1024
DEFAULT_TARGET_STEPS: int = 2**14


def simulated_experiment_budget(*, train_batch_size: int, num_train_steps: int, seq_len: int) -> int:
    """Tokens consumed by one training run at the given schedule.

    Returned as an int so ``dataclasses.replace(data, experiment_budget=...)``
    doesn't surprise with a float in a field typed int.
    """
    return int(train_batch_size) * int(num_train_steps) * int(seq_len)


def _testbed_tokenize_config(sampled: StepSpec, cache_path: str, tokenizer: str) -> TokenizeConfig:
    """The tokenize config for one bucket: tokenize the sample's main parquet shards."""
    return TokenizeConfig(
        train_paths=[f"{sampled.output_path}/outputs/main/*.parquet"],
        validation_paths=[],
        cache_path=cache_path,
        tokenizer=tokenizer,
    )


def _bucket_tokenize_config(bucket: StepSpec, tokenizer: str) -> TokenizeConfig:
    """Rebuild a tokenize bucket's config so it can be wired into a mixture component.

    A bucket's sole dependency is its sampled source; the cache lives at the bucket's
    own output path. Both resolve their storage prefix lazily at call time.
    """
    (sampled,) = bucket.deps
    return _testbed_tokenize_config(sampled, bucket.output_path, tokenizer)


def testbed_tokenize(
    bucket_name: str,
    sampled: StepSpec,
    tokenizer: str = TESTBED_TOKENIZER,
) -> StepSpec:
    """Convert a sample step into a training-ready tokenize ``StepSpec``.

    The step depends on ``sampled`` so the runner tokenizes only after the sample
    shards exist, and reads the sample's resolved output path at run time.
    """

    def fn(output_path: str) -> None:
        tokenize(_testbed_tokenize_config(sampled, output_path, tokenizer))

    return StepSpec(
        name=os.path.join("data/datakit", "tokenized", bucket_name),
        deps=[sampled],
        hash_attrs={"tokenizer": tokenizer},
        fn=fn,
    )


@dataclass(frozen=True)
class TestbedTrialConfig:
    """Wraps a ``GrugMoeLaunchConfig`` plus a path to a runtime-computed weights file.

    ``weights_path`` points at the output of ``tokenized_bucket_weights_step``. The
    runner reads the JSON file, patches ``train_weights`` on the embedded data config,
    and dispatches to ``run_grug_moe_trial``.
    """

    weights_path: str
    grug_config: GrugMoeLaunchConfig


def run_testbed_trial(config: TestbedTrialConfig) -> None:
    """Read bucket weights, splice into the data config, then run Grug-MoE."""
    weights = read_bucket_weights(config.weights_path)
    data = dataclasses.replace(config.grug_config.data, train_weights=weights)
    grug = dataclasses.replace(config.grug_config, data=data)
    run_grug_moe_trial(grug)


def run_testbed_config(
    *,
    name: str,
    tokenized_buckets: dict[str, StepSpec],
    weights_step: StepSpec,
    validation: Sequence[Dataset],
    compute_budget_flops: float = DEFAULT_COMPUTE_BUDGET_FLOPS,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    target_steps: int = DEFAULT_TARGET_STEPS,
    target_budget_tokens: int = DEFAULT_TARGET_BUDGET_TOKENS,
    tokenizer: str = TESTBED_TOKENIZER,
    wandb_group: str = "datakit-testbed",
    wandb_tags: Sequence[str] = ("datakit-testbed", "moe"),
    seed: int = 0,
    tpu: str = "v5p-8",
) -> StepSpec:
    """Assemble a Grug-MoE training step for one testbed configuration.

    Args:
        name: Config name — forms the step name and wandb run id. Use e.g.
            ``"baseline"`` for the trivial no-dedup run.
        tokenized_buckets: Per-source tokenize steps. The caller builds these from
            its own bucketed view of the sampled data — baseline buckets by
            provenance (one tokenize per source); other configs may bucket
            differently (e.g. by quality tier).
        weights_step: Step that produces a ``weights.json`` artifact (see
            :func:`tokenized_bucket_weights_step`). Real weights are read at training
            time, so the training DAG can be validated without the tokenize outputs
            existing yet.
        validation: Validation ``Dataset`` handles added to the mixture at weight 0.
        compute_budget_flops: FLOP budget fed to ``build_from_heuristic``.
        hidden_dim: Model hidden dimension for the heuristic.
        target_steps: Heuristic target steps; default ``2**14`` matches Grug.
        target_budget_tokens: Denominator for simulated-epoching slicing — "how many
            tokens would the full run consume". Default 1T per RFC.
        tokenizer: Tokenizer used across every component. Must match the training
            model's tokenizer; defaults to ``TESTBED_TOKENIZER``.
        wandb_group: Groups this run with siblings in the ranking protocol.
        wandb_tags: Tags attached to the wandb run.
        seed: RNG seed for the trainer.
        tpu: Fray TPU request string (e.g. ``"v5p-8"``).

    Returns:
        A ``StepSpec`` whose ``fn`` is ``run_testbed_trial``. The step runs inline and
        ``run_grug_moe_trial`` dispatches the training job to Fray.
    """
    if not tokenized_buckets:
        raise ValueError("tokenized_buckets must be non-empty")

    model_cfg, opt_cfg, batch_size, steps = build_from_heuristic(
        budget=compute_budget_flops,
        hidden_dim=hidden_dim,
        target_steps=target_steps,
    )

    experiment_budget = simulated_experiment_budget(
        train_batch_size=batch_size,
        num_train_steps=steps,
        seq_len=TESTBED_SEQ_LEN,
    )

    logger.info(
        "testbed config %s: budget=%.2e flops, H=%d, batch=%d, steps=%d, "
        "experiment_budget=%.2eB tokens over target_budget=%.2eB",
        name,
        compute_budget_flops,
        hidden_dim,
        batch_size,
        steps,
        experiment_budget / 1e9,
        target_budget_tokens / 1e9,
    )

    buckets = dict(tokenized_buckets)
    validation = list(validation)

    def fn(output_path: str) -> None:
        prefix = marin_prefix()
        # Placeholder uniform weights — only the structure of LmDataConfig matters
        # here; ``run_testbed_trial`` overwrites ``train_weights`` from the on-disk
        # weights.json before dispatching to ``run_grug_moe_trial``. Validation
        # handles are padded to weight 0 (missing_weights_are_validation).
        components: dict[str, TokenizeConfig] = {
            bucket: _bucket_tokenize_config(step, tokenizer) for bucket, step in buckets.items()
        }
        for handle in validation:
            components[handle.name] = materialized_config(handle, prefix)
        placeholder_weights = {bucket: 1.0 for bucket in buckets}

        data = lm_mixture_data_config(components, placeholder_weights)
        data = dataclasses.replace(
            data,
            target_budget=target_budget_tokens,
            experiment_budget=experiment_budget,
        )

        grug_config = GrugMoeLaunchConfig(
            model=model_cfg,
            data=data,
            output_path=output_path,
            run_id=f"datakit-testbed-{name}",
            resources=ResourceConfig.with_tpu(tpu),
            steps=steps,
            batch_size=batch_size,
            seed=seed,
            mp="params=float32,compute=bfloat16,output=bfloat16",
            tracker=WandbConfig(
                project="marin",
                tags=list(wandb_tags),
                group=wandb_group,
                name=None,
            ),
            optimizer=opt_cfg,
            grug_trainer=GrugTrainerConfig(
                z_loss_weight=1e-4,
                ema_beta=None,
                log_every=10,
            ),
            eval=GrugEvalConfig(
                eval_batch_size=512,
                steps_per_eval=1000,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
            ),
        )
        run_testbed_trial(TestbedTrialConfig(weights_path=weights_step.output_path, grug_config=grug_config))

    deps = [weights_step, *buckets.values(), *(lower(handle) for handle in validation)]
    return StepSpec(
        name=f"data/datakit/train/{name}",
        deps=deps,
        hash_attrs={
            "compute_budget_flops": compute_budget_flops,
            "hidden_dim": hidden_dim,
            "target_steps": target_steps,
            "target_budget_tokens": target_budget_tokens,
            "tokenizer": tokenizer,
            "seed": seed,
        },
        fn=fn,
    )

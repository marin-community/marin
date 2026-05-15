# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grug-MoE training harness with simulated epoching over the testbed mixture.

``run_testbed_config`` wraps ``experiments.grug.moe.launch.run_grug_moe_trial``
and applies simulated epoching via ``target_budget``/``experiment_budget`` on
the ``LmDataConfig`` — matching the contract in
``experiments.defaults.simulated_epoching_train`` but routed through Grug's
own training entry point (Grug doesn't go through ``default_train``).

The RFC's canonical testbed dataset targets ~1T raw tokens sampled
proportionally by provenance. At training time we pick a much smaller
compute-optimal ``experiment_budget`` (batch * steps * seq_len) and let
``MixtureDataset`` slice each component by ``experiment_budget / target_budget``
to preserve the per-source shares over the shortened horizon (see
``levanter/data/text/datasets.py:763-770``).
"""

from __future__ import annotations

import dataclasses
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, output_path_of, this_output_path, versioned
from marin.execution.step_spec import StepSpec
from marin.processing.tokenize import (
    TokenizeConfig,
    add_validation_sets_to_mixture,
    lm_mixture_data_config,
    tokenize,
)
from marin.processing.tokenize.data_configs import TokenizerStep

from experiments.datakit_testbed.mixture import read_bucket_weights
from experiments.datakit_testbed.settings import TESTBED_SEQ_LEN, TESTBED_TOKENIZER
from experiments.defaults import default_validation_sets
from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import GrugMoeLaunchConfig, run_grug_moe_trial
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

logger = logging.getLogger(__name__)


# RFC target: 1T raw tokens sampled proportionally across the testbed.
DEFAULT_TARGET_BUDGET_TOKENS: int = 1_000_000_000_000

# Grug MoE baseline — matches the values used by
# ``experiments/grug/moe/launch.py:120-127``. We reuse them so the testbed
# ranking protocol starts at the same compute point as the existing Grug runs.
DEFAULT_COMPUTE_BUDGET_FLOPS: float = 1e18
DEFAULT_HIDDEN_DIM: int = 1024
DEFAULT_TARGET_STEPS: int = 2**14


def simulated_experiment_budget(*, train_batch_size: int, num_train_steps: int, seq_len: int) -> int:
    """Tokens consumed by one training run at the given schedule.

    Returned as an int so ``dataclasses.replace(data, experiment_budget=...)``
    doesn't surprise with a float in a field typed int.
    """
    return int(train_batch_size) * int(num_train_steps) * int(seq_len)


def testbed_tokenize(
    bucket_name: str,
    sampled: StepSpec,
    tokenizer: str = TESTBED_TOKENIZER,
) -> TokenizerStep:
    """Convert a sample ``StepSpec`` into a training-ready ``TokenizerStep``.

    Mirrors ``experiments/pretraining_datasets/nemotron_v2.py:65-76``: build
    a real ``ExecutorStep[TokenizeConfig]`` that references the sample
    output via ``InputName`` so the path is resolved lazily (the sample
    doesn't need to be materialized at tokenize-step-construction time)
    and the executor preserves dep tracking.
    """
    sampled_exec = sampled.as_executor_step()
    return ExecutorStep(
        name=os.path.join("data/datakit", "tokenized", bucket_name),
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=[sampled_exec / "outputs/main/*.parquet"],
            validation_paths=versioned([]),
            cache_path=this_output_path(),
            tokenizer=versioned(tokenizer),
        ),
    )


@dataclass(frozen=True)
class TestbedTrialConfig:
    """Wraps a ``GrugMoeLaunchConfig`` plus a path to a runtime-computed weights file.

    ``weights_path`` is an ``InputName`` at construction (output of
    ``tokenized_bucket_weights_step``) and a concrete path at runtime. The
    runner reads the JSON file, patches ``train_weights`` on the embedded
    data config, and dispatches to ``run_grug_moe_trial``.
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
    tokenized_buckets: dict[str, TokenizerStep],
    weights_step: ExecutorStep,
    compute_budget_flops: float = DEFAULT_COMPUTE_BUDGET_FLOPS,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    target_steps: int = DEFAULT_TARGET_STEPS,
    target_budget_tokens: int = DEFAULT_TARGET_BUDGET_TOKENS,
    tokenizer: str = TESTBED_TOKENIZER,
    wandb_group: str = "datakit-testbed",
    wandb_tags: Sequence[str] = ("datakit-testbed", "moe"),
    seed: int = 0,
    tpu: str = "v5p-8",
) -> ExecutorStep:
    """Assemble a Grug-MoE training step for one testbed configuration.

    Args:
        name: Config name — forms the executor step name and wandb run id.
            Use e.g. ``"baseline"`` for the trivial no-dedup run.
        tokenized_buckets: Per-source tokenize ExecutorSteps. The caller
            builds these from its own bucketed view of the sampled data
            — baseline buckets by provenance (one tokenize per source);
            other configs may bucket differently (e.g. by quality tier).
        weights_step: ExecutorStep that produces a ``weights.json`` artifact
            (see :func:`tokenized_bucket_weights_step`). Real weights are
            read at training time, so the training DAG can be validated
            via ``--dry_run`` without the tokenize outputs existing yet.
        compute_budget_flops: FLOP budget fed to ``build_from_heuristic``.
        hidden_dim: Model hidden dimension for the heuristic.
        target_steps: Heuristic target steps; default ``2**14`` matches Grug.
        target_budget_tokens: Denominator for simulated-epoching slicing —
            "how many tokens would the full run consume". Default 1T per RFC.
        tokenizer: Tokenizer used across every component. Must match the
            training model's tokenizer; defaults to ``TESTBED_TOKENIZER``.
        wandb_group: Groups this run with siblings in the ranking protocol.
        wandb_tags: Tags attached to the wandb run.
        seed: RNG seed for the trainer.
        tpu: Fray TPU request string (e.g. ``"v5p-8"``).

    Returns:
        An ``ExecutorStep`` whose ``fn`` is ``run_testbed_trial``. Pass to
        ``executor_main`` to validate the DAG (dry-run) or actually train.
    """
    if not tokenized_buckets:
        raise ValueError("tokenized_buckets must be non-empty")

    model_cfg, opt_cfg, batch_size, steps = build_from_heuristic(
        budget=compute_budget_flops,
        hidden_dim=hidden_dim,
        target_steps=target_steps,
    )

    # Placeholder uniform weights — only the structure of LmDataConfig matters
    # at construction; ``run_testbed_trial`` overwrites ``train_weights`` from
    # the on-disk weights.json before dispatching to ``run_grug_moe_trial``.
    placeholder_weights = {bucket: 1.0 for bucket in tokenized_buckets}
    data = lm_mixture_data_config(components=tokenized_buckets, weights=placeholder_weights)
    data = add_validation_sets_to_mixture(
        data,
        default_validation_sets(tokenizer=tokenizer),
    )

    experiment_budget = simulated_experiment_budget(
        train_batch_size=batch_size,
        num_train_steps=steps,
        seq_len=TESTBED_SEQ_LEN,
    )
    data = dataclasses.replace(
        data,
        target_budget=target_budget_tokens,
        experiment_budget=experiment_budget,
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

    grug_config = GrugMoeLaunchConfig(
        model=versioned(model_cfg),
        data=data,
        output_path=this_output_path(),
        run_id=f"datakit-testbed-{name}",
        resources=versioned(ResourceConfig.with_tpu(tpu)),
        steps=versioned(steps),
        batch_size=versioned(batch_size),
        seed=versioned(seed),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=list(wandb_tags),
            group=wandb_group,
            name=None,
        ),
        optimizer=versioned(opt_cfg),
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=1e-4,
                ema_beta=None,
                log_every=10,
            )
        ),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=512,
                steps_per_eval=1000,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
            )
        ),
    )

    return ExecutorStep(
        name=f"data/datakit/train/{name}",
        fn=run_testbed_trial,
        config=TestbedTrialConfig(
            weights_path=output_path_of(weights_step),
            grug_config=grug_config,
        ),
    )

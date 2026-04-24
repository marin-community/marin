# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Datakit Testbed ferry DAG builder.

Composes the canonical Datakit stages into one multi-source pipeline:

    <source.normalize_steps> ─► sample[source]

Each :class:`DatakitSource` already carries its full ``(download, ..., normalize)``
:class:`StepSpec` chain (see :mod:`marin.datakit.sources`). This module just
appends the testbed-specific sample stage on top of every source's terminal
normalize step.

Key shape choices:
* **Sampling happens post-normalize.** Normalize produces uniform-size shards
  (``target_partition_bytes``), making "first K by filename" both byte-fair
  and content-fair. Sample is a one-time cost cached by ``override_output_path``.
* **Downloads intentionally duplicate when sources share staging.** The
  executor dedupes content-identical ``StepSpec``s at run time via
  ``override_output_path``, so no in-DAG grouping is needed.

The ferry stops at ``sample``. Tokenize runs in the training executor graph
(see ``experiments/datakit_testbed/train.py``), not the ferry, because
``lm_mixture_data_config`` needs real ``ExecutorStep[TokenizeConfig]``
components — not the ``StepSpec`` instances this ferry produces.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from marin.datakit.sources import DatakitSource, all_sources
from marin.execution.executor import ExecutorStep
from marin.execution.step_runner import check_cache
from marin.execution.step_spec import StepSpec

from experiments.datakit_testbed.sampler import (
    proportional_sample_fractions,
    sample_normalized_shards_step,
)
from experiments.datakit_testbed.settings import RAW_TARGET_TOTAL_TOKENS_B, TESTBED_TOKENIZER
from experiments.datakit_testbed.train import run_testbed_config

logger = logging.getLogger(__name__)


def _sample_step_for(
    src: DatakitSource,
    normalized: StepSpec,
    sample_fraction: float,
    base: str,
) -> StepSpec:
    """Per-source post-normalize sampler. Copies first ceil(N * fraction) shards."""
    return sample_normalized_shards_step(
        name=f"datakit-testbed/sample/{src.name}",
        normalized=normalized,
        sample_fraction=sample_fraction,
        override_output_path=f"{base}/{src.name}",
    )


def build_testbed_steps(
    run_id: str,
    sources: Sequence[DatakitSource] | None = None,
    target_total_tokens_b: float = RAW_TARGET_TOTAL_TOKENS_B,
) -> list[StepSpec]:
    """Build the full Datakit Testbed ferry DAG.

    Args:
        run_id: Per-run identifier; sample output paths land under
            ``datakit_testbed/{run_id}/sample/...``. Normalize outputs land
            at canonical run-independent paths (``normalized/<name>-<hash>``)
            so they're reused across runs.
        sources: DatakitSource list to ferry. ``None`` auto-selects every
            entry from :func:`all_sources` whose normalize output is
            already cached on GCS, matching the run_source_sampling script.
            Pass an explicit list to bypass this check.
        target_total_tokens_b: Target total token count (in billions) across
            the sampled set. Drives per-source sample fractions via
            :func:`proportional_sample_fractions`. Default is
            :data:`RAW_TARGET_TOTAL_TOKENS_B`.

    Returns:
        Flat list of :class:`StepSpec` covering every normalize chain plus
        one sample step per source. Ready to hand to ``StepRunner().run()``.
    """
    if sources is None:
        # TODO (rav): remove the check_cache when ready?
        sources = tuple(s for s in all_sources().values() if check_cache(s.normalized.output_path))
    if not sources:
        raise ValueError("build_testbed_steps requires at least one source")

    base = f"datakit_testbed/{run_id}"

    fractions = proportional_sample_fractions(sources, target_total_tokens_b=target_total_tokens_b)

    all_steps: list[StepSpec] = []
    for src in sources:
        # Each source contributes its own download → [preprocess] → normalize
        # chain. Duplicates across sources are dedup'd by the executor at run time.
        all_steps.extend(src.normalize_steps)
        all_steps.append(_sample_step_for(src, src.normalized, fractions[src.name], base))

    logger.info(
        "Built testbed DAG: %d sources, %d steps (normalize chains + sample), target %.0fB tokens",
        len(sources),
        len(all_steps),
        target_total_tokens_b,
    )

    return all_steps


_SAMPLE_STEP_PREFIX = "datakit-testbed/sample/"


def baseline(
    steps: list[StepSpec],
    *,
    name: str = "baseline",
    tokenizer: str = TESTBED_TOKENIZER,
    **run_config_kwargs,
) -> ExecutorStep:
    """Assemble the baseline (control-arm) training step off *steps*.

    The testbed's ranking protocol runs every experiment alongside a
    baseline where the pipeline's middle stages are deliberate no-ops:

    * **no-op dedup** — every sampled doc survives (no fuzzy/exact cut)
    * **constant-quality filter** — all docs tagged equal quality
    * **bucket by provenance** — the sample output IS already the bucket
      (one shard set per source), so bucketing is an identity op

    With all three as no-ops, the sampled parquet that ``build_testbed_steps``
    produces is also the bucket, so this function wires one tokenize
    ExecutorStep per sample output, builds the proportional mixture over
    them, and hands the result to :func:`run_testbed_config` to assemble
    the full Grug-MoE training ExecutorStep.

    Args:
        steps: Return value of :func:`build_testbed_steps`.
        name: Config name — forms the executor step name and wandb run id.
        tokenizer: Tokenizer to use across every component; must match
            the training model's tokenizer.
        **run_config_kwargs: Forwarded to :func:`run_testbed_config`
            (compute_budget_flops, hidden_dim, target_steps, weights,
            tpu, wandb_group, etc.).

    Returns:
        An ``ExecutorStep`` whose ``fn`` is ``run_grug_moe_trial``. Pass
        to ``executor_main`` to actually train.
    """
    sampled_by_source = {
        s.name.removeprefix(_SAMPLE_STEP_PREFIX): s for s in steps if s.name.startswith(_SAMPLE_STEP_PREFIX)
    }
    if not sampled_by_source:
        raise ValueError("no sample steps found in the DAG (expected names under 'datakit-testbed/sample/...')")
    return run_testbed_config(
        name=name,
        sampled_by_source=sampled_by_source,
        tokenizer=tokenizer,
        **run_config_kwargs,
    )

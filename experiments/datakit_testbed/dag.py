# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Datakit Testbed ferry DAG builder.

Composes the canonical Datakit stages into one multi-source pipeline:

    <source.normalize_steps> ─► sample[source] ─┐
                                                 ├─► noop_dedup (all sampled)
                                                 │
    sample[source] ────────────── consolidate[source]

Each :class:`DatakitSource` already carries its full ``(download, ..., normalize)``
:class:`StepSpec` chain (see :mod:`marin.datakit.sources`). This module just
appends the testbed-specific stages (sample → dedup → consolidate) on top of
every source's terminal normalize step.

Key shape choices:
* **Sampling happens post-normalize.** Normalize produces uniform-size shards
  (``target_partition_bytes``), making "first K by filename" both byte-fair
  and content-fair. Downstream stages (dedup, consolidate, tokenize) pay
  O(sampled) per experiment; normalize is a one-time cost cached by
  ``override_output_path``.
* **Downloads intentionally duplicate when sources share staging.** The
  executor dedupes content-identical ``StepSpec``s at run time via
  ``override_output_path``, so no in-DAG grouping is needed.
* Dedup is a single step with one dep per sampled source, emitting
  per-source attr directories.
* Consolidate fans back out to one step per ``DatakitSource`` so each
  mixture component has its own filtered parquet output.

The ferry stops at ``consolidate``. Tokenize runs in the training executor
graph (see ``experiments/datakit_testbed/train.py``), not the ferry, because
``lm_mixture_data_config`` needs real ``ExecutorStep[TokenizeConfig]``
components — not the ``StepSpec`` instances this ferry produces.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from fray import ResourceConfig
from marin.datakit.normalize import NormalizedData
from marin.datakit.sources import DatakitSource, all_sources
from marin.execution.artifact import Artifact
from marin.execution.step_spec import StepSpec
from marin.processing.classification.consolidate import (
    FilterConfig,
    FilterType,
    consolidate,
)
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData

from experiments.datakit_testbed.noop_dedup import compute_noop_dedup_attrs_step
from experiments.datakit_testbed.sampler import (
    proportional_sample_fractions,
    sample_normalized_shards_step,
)
from experiments.datakit_testbed.settings import RAW_TARGET_TOTAL_TOKENS_B

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TestbedDAG:
    """Handle to the built ferry DAG.

    All steps must be passed to ``StepRunner().run(...)`` for execution.
    ``consolidated_by_source`` is exposed separately so the training harness
    can reference each source's consolidate step when building the tokenize
    ExecutorSteps that feed the mixture.
    """

    all_steps: list[StepSpec]
    consolidated_by_source: dict[str, StepSpec]


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
        override_output_path=f"{base}/sample/{src.name}",
    )


def _consolidate_step_for(
    src: DatakitSource,
    sampled: StepSpec,
    deduped: StepSpec,
    base: str,
) -> StepSpec:
    """Per-source consolidate. Resolves attr_dir at runtime via Artifact.load."""
    return StepSpec(
        name=f"datakit-testbed/consolidate/{src.name}",
        deps=[sampled, deduped],
        fn=lambda output_path: consolidate(
            input_path=Artifact.load(sampled, NormalizedData).main_output_dir,
            output_path=output_path,
            filetype="parquet",
            filters=[
                FilterConfig(
                    type=FilterType.KEEP_DOC,
                    attribute_path=Artifact.load(deduped, FuzzyDupsAttrData)
                    .sources[Artifact.load(sampled, NormalizedData).main_output_dir]
                    .attr_dir,
                    name="is_cluster_canonical",
                    attribute_filetype="parquet",
                    keep_if_missing=True,
                ),
            ],
            worker_resources=ResourceConfig(cpu=1, ram="8g"),
        ),
        override_output_path=f"{base}/consolidate/{src.name}",
    )


def build_testbed_steps(
    run_id: str,
    sources: Sequence[DatakitSource] | None = None,
    target_total_tokens_b: float = RAW_TARGET_TOTAL_TOKENS_B,
) -> TestbedDAG:
    """Build the full Datakit Testbed ferry DAG.

    Args:
        run_id: Per-run identifier; sample/dedup/consolidate output paths land
            under ``datakit-testbed/{run_id}/...``. Normalize outputs land at
            canonical run-independent paths (``normalized/<name>-<hash>``) so
            they're reused across runs.
        sources: DatakitSource list to ferry. ``None`` means every entry in
            :func:`all_sources` (the canonical active registry).
        target_total_tokens_b: Target total token count (in billions) across
            the sampled set. Drives per-source sample fractions via
            :func:`proportional_sample_fractions`. Default is
            :data:`RAW_TARGET_TOTAL_TOKENS_B` (1000B = 1T per RFC).

    Returns:
        A ``TestbedDAG`` whose ``all_steps`` list is safe to pass directly to
        ``StepRunner().run(...)``.
    """
    if sources is None:
        sources = tuple(all_sources().values())
    if not sources:
        raise ValueError("build_testbed_steps requires at least one source")

    base = f"datakit-testbed/{run_id}"

    fractions = proportional_sample_fractions(sources, target_total_tokens_b=target_total_tokens_b)

    all_steps: list[StepSpec] = []
    sampled: dict[str, StepSpec] = {}
    for src in sources:
        # Each source contributes its own download → [preprocess] → normalize
        # chain. Duplicates across sources are dedup'd by the executor at run time.
        all_steps.extend(src.normalize_steps)
        sample = _sample_step_for(src, src.normalized, fractions[src.name], base)
        sampled[src.name] = sample
        all_steps.append(sample)

    deduped = compute_noop_dedup_attrs_step(
        name="datakit-testbed/noop_dedup",
        normalized_steps=list(sampled.values()),
        override_output_path=f"{base}/noop_dedup",
    )
    all_steps.append(deduped)

    consolidated: dict[str, StepSpec] = {
        src.name: _consolidate_step_for(src, sampled[src.name], deduped, base) for src in sources
    }
    all_steps.extend(consolidated.values())

    logger.info(
        "Built testbed DAG: %d sources, %d steps (normalize chains + sample + "
        "1 noop_dedup + consolidate), target %.0fB tokens",
        len(sources),
        len(all_steps),
        target_total_tokens_b,
    )

    return TestbedDAG(all_steps=all_steps, consolidated_by_source=consolidated)

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Datakit Testbed ferry DAG builder.

Composes the canonical Datakit stages into one multi-source pipeline:

    download[hf]  ─┐
                   ├─► normalize[source] ─► sample[source] ─┐
    download[hf]  ─┘                                         ├─► noop_dedup (all sampled)
                                                             │
    sample[source] ────────────────── consolidate[source]

Key shape choices:
* Downloads are grouped by ``(hf_dataset_id, revision)`` so Nemotron v2.1
  subsets that share a family dump do not re-download.
* **Sampling happens post-normalize.** Normalize produces uniform-size shards
  (``target_partition_bytes``), making "first K by filename" both byte-fair
  and content-fair. Downstream stages (dedup, consolidate, tokenize) pay
  O(sampled) per experiment; normalize is a one-time cost cached by
  ``override_output_path``.
* Dedup is a single step with one dep per sampled source, emitting
  per-source attr directories.
* Consolidate fans back out to one step per ``DatakitSource`` so each
  mixture component has its own filtered parquet output.

The ferry stops at ``consolidate`` on purpose. Tokenize runs in the training
executor graph (see ``experiments/datakit_testbed/train.py``), not the ferry,
because ``lm_mixture_data_config`` needs real ``ExecutorStep[TokenizeConfig]``
components — not the ``StepSpec`` instances this ferry produces. The training
harness converts each consolidate ``StepSpec`` to an ``ExecutorStep`` via
``.as_executor_step()`` and builds a proper ``TokenizerStep`` on top, which
preserves cross-layer dep tracking for the executor.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from fray import ResourceConfig
from marin.datakit.normalize import NormalizedData
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
from marin.datakit.sources import DatakitSource, pinned_sources

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


DownloadKey = tuple[str, str, str | None, tuple[str, ...] | None]


def _download_key(src: DatakitSource) -> DownloadKey:
    """Group sources that map to the same physical download.

    Some HF repos (e.g. ``bigcode/StarCoder2-Extras``) are downloaded per-subset
    with distinct ``override_output_path`` values, so the same
    ``(hf_repo, revision)`` can correspond to multiple physical download steps —
    hence ``staged_path`` and ``hf_urls_glob`` are part of the key.
    """
    return (src.hf_dataset_id, src.revision or "", src.staged_path, src.hf_urls_glob)


def _build_downloads(sources: Sequence[DatakitSource]) -> dict[DownloadKey, StepSpec]:
    """One download StepSpec per unique ``(hf_repo, revision, staged_path, urls_glob)``."""
    by_key: dict[DownloadKey, StepSpec] = {}
    for src in sources:
        key = _download_key(src)
        if key not in by_key:
            by_key[key] = src.download_step()
    return by_key


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
    normalized: StepSpec,
    deduped: StepSpec,
    base: str,
) -> StepSpec:
    """Per-source consolidate. Resolves attr_dir at runtime via Artifact.load."""
    return StepSpec(
        name=f"datakit-testbed/consolidate/{src.name}",
        deps=[normalized, deduped],
        fn=lambda output_path: consolidate(
            input_path=Artifact.load(normalized, NormalizedData).main_output_dir,
            output_path=output_path,
            filetype="parquet",
            filters=[
                FilterConfig(
                    type=FilterType.KEEP_DOC,
                    attribute_path=Artifact.load(deduped, FuzzyDupsAttrData)
                    .sources[Artifact.load(normalized, NormalizedData).main_output_dir]
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
        run_id: Per-run identifier; output paths are ``datakit-testbed/{run_id}/...``
            under ``MARIN_PREFIX`` so reruns are isolated.
        sources: DatakitSource list to ferry. ``None`` means the default set
            produced by :func:`pinned_sources` — every registry entry that
            has a pinned HF revision and a non-empty repo. Pass explicitly if
            you need to include unpinned or API-sourced entries with a custom
            download wiring.
        target_total_tokens_b: Target total token count (in billions) across
            the sampled set. Drives per-source sample fractions via
            :func:`proportional_sample_fractions`. Default is
            :data:`RAW_TARGET_TOTAL_TOKENS_B` (1000B = 1T per RFC).

    Returns:
        A ``TestbedDAG`` whose ``all_steps`` list is safe to pass directly to
        ``StepRunner().run(...)``.
    """
    if sources is None:
        sources = tuple(pinned_sources().values())
    if not sources:
        raise ValueError("build_testbed_steps requires at least one source")

    base = f"datakit-testbed/{run_id}"

    fractions = proportional_sample_fractions(sources, target_total_tokens_b=target_total_tokens_b)

    downloads = _build_downloads(sources)
    normalized: dict[str, StepSpec] = {}
    sampled: dict[str, StepSpec] = {}
    for src in sources:
        normalized[src.name] = src.normalize_step(downloads[_download_key(src)])
        sampled[src.name] = _sample_step_for(src, normalized[src.name], fractions[src.name], base)

    deduped = compute_noop_dedup_attrs_step(
        name="datakit-testbed/noop_dedup",
        normalized_steps=list(sampled.values()),
        override_output_path=f"{base}/noop_dedup",
    )

    consolidated: dict[str, StepSpec] = {
        src.name: _consolidate_step_for(src, sampled[src.name], deduped, base) for src in sources
    }

    all_steps: list[StepSpec] = []
    all_steps.extend(downloads.values())
    all_steps.extend(normalized.values())
    all_steps.extend(sampled.values())
    all_steps.append(deduped)
    all_steps.extend(consolidated.values())

    logger.info(
        "Built testbed DAG: %d sources, %d downloads, %d samplers (target %.0fB tokens), "
        "1 noop_dedup, %d consolidated outputs",
        len(sources),
        len(downloads),
        len(sampled),
        target_total_tokens_b,
        len(consolidated),
    )

    return TestbedDAG(all_steps=all_steps, consolidated_by_source=consolidated)

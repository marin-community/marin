# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deduplication and eval-contamination attributes for rendered SFT conversations."""

import argparse
import json
import logging
from dataclasses import dataclass, replace

from marin.datakit.decon import DropSetSource, all_source_drop_sets_step, build_eval_bloom_step, decon_step
from marin.datakit.normalize import NormalizedData
from marin.datakit.sft_sources import DatakitChatSource, all_sft_sources
from marin.execution.artifact import read_artifact
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData
from marin.processing.classification.deduplication.fuzzy_minhash import MinHashAttrData
from marin.processing.classification.deduplication.fuzzy_verification import FuzzyVerificationParams
from marin.processing.classification.deduplication.verify_fuzzy_dups import (
    REFERENCE_LOCAL_REPRESENTATIVE_PARAMS,
    VERIFIED_FUZZY_DUPS_ATTR_DATA_VERSION,
    FuzzyVerificationStoreConfig,
    verify_fuzzy_dups,
)
from rigging.log_setup import configure_logging
from zephyr.context import ZephyrContext
from zephyr.runners import SubprocessRunner

from experiments.datakit.reference_pipeline import (
    AA_BENCHMARK_NAMES,
    AA_MANIFEST_PATH,
    DECON_EXCLUDED_EVAL_TASKS,
    DEFAULT_SCALE,
    ESTIMATED_DOC_COUNT,
    EVAL_CORPUS_VERSION,
    EVAL_ROOT,
    FALSE_POSITIVE_RATE,
    FLAGGED_SAMPLE_SIZE,
    GLOBAL_DF_COMMON_MIN_ABS,
    GLOBAL_DF_COMMON_MIN_SOURCES,
    GLOBAL_DF_SAMPLE_DOCS,
    LMH_MANIFEST_PATH,
    NGRAM_LENGTH,
    OVERLAP_THRESHOLD,
    SMOKE_SCALE,
    SOURCE_DF_COMMON_FRAC,
    SOURCE_DF_COMMON_MIN_ABS,
    SOURCE_DF_SAMPLE_DOCS,
    PipelineScale,
    zephyr_datakit_steps,
)


@dataclass(frozen=True)
class SftFilterSteps:
    """Attribute outputs keyed to normalized rendered-text IDs."""

    normalized: dict[str, StepSpec]
    exact_dedup: StepSpec
    verified_dedup: StepSpec
    decontam: dict[str, StepSpec]

    @property
    def targets(self) -> list[StepSpec]:
        return [self.exact_dedup, self.verified_dedup, *self.decontam.values()]


def decontamination_steps(
    sources: dict[str, StepSpec],
    scale: PipelineScale = DEFAULT_SCALE,
    zephyr_context: ZephyrContext | None = None,
) -> tuple[StepSpec, StepSpec, dict[str, StepSpec]]:
    """Build a shared eval bloom, frequency drop sets, and per-source marks."""
    # One combined decontam bloom (no merge step); every per-source decon
    # consumes it directly. Same name/params as the testbed decon arm, so runs
    # sharing a prefix share the built bloom.
    decon_bloom_step = build_eval_bloom_step(
        name="datakit/bloom/_combined_fixed",
        eval_data_sources=[EVAL_ROOT],
        ngram_length=NGRAM_LENGTH,
        overlap_threshold=OVERLAP_THRESHOLD,
        estimated_doc_count=ESTIMATED_DOC_COUNT,
        false_positive_rate=FALSE_POSITIVE_RATE,
        exclude_eval_dirs=DECON_EXCLUDED_EVAL_TASKS,
        required_eval_manifest_path=AA_MANIFEST_PATH,
        required_eval_corpus_version=EVAL_CORPUS_VERSION,
        required_eval_names=AA_BENCHMARK_NAMES,
        best_effort_eval_manifest_path=LMH_MANIFEST_PATH,
        best_effort_eval_corpus_version=EVAL_CORPUS_VERSION,
    )
    # Count eval-ngram document frequency across normalized sources before
    # marking. Each decon consumes its source-local set and the global set.
    decon_drop_sets = all_source_drop_sets_step(
        name="datakit/decon_drop/_combined",
        sources=[
            DropSetSource(
                name=source_name,
                data_path=f"{normalize_step.output_path.rstrip('/')}/outputs/main",
                dependency=normalize_step,
            )
            for source_name, normalize_step in sources.items()
        ],
        prebuilt_bloom=decon_bloom_step,
        ngram_length=NGRAM_LENGTH,
        sample_docs=SOURCE_DF_SAMPLE_DOCS,
        common_frac=SOURCE_DF_COMMON_FRAC,
        common_min_abs=SOURCE_DF_COMMON_MIN_ABS,
        global_sample_docs=GLOBAL_DF_SAMPLE_DOCS,
        global_common_min_abs=GLOBAL_DF_COMMON_MIN_ABS,
        global_common_min_sources=GLOBAL_DF_COMMON_MIN_SOURCES,
        worker_resources=scale.pool.worker,
        max_workers=scale.pool.n_workers,
        zephyr_context=zephyr_context,
    )

    marks = {}
    for name, normalize_step in sources.items():
        decontam = decon_step(
            name=f"datakit/decontam/{name}",
            normalized=normalize_step,
            prebuilt_bloom=decon_bloom_step,
            drop_sets=decon_drop_sets,
            drop_set_source=name,
            ngram_length=NGRAM_LENGTH,
            overlap_threshold=OVERLAP_THRESHOLD,
            estimated_doc_count=ESTIMATED_DOC_COUNT,
            false_positive_rate=FALSE_POSITIVE_RATE,
            flagged_sample_size=FLAGGED_SAMPLE_SIZE,
            worker_resources=scale.pool.worker,
            zephyr_context=zephyr_context,
        )

        marks[name] = decontam
    return decon_bloom_step, decon_drop_sets, marks


def verified_dedup_step(
    sources: dict[str, StepSpec],
    minhash: dict[str, StepSpec],
    dedup: StepSpec,
    scale: PipelineScale = DEFAULT_SCALE,
) -> StepSpec:
    """Verify fuzzy candidates against the normalized source text."""
    verification_params = FuzzyVerificationParams()
    verification_store_config = FuzzyVerificationStoreConfig(
        recovery_timeout=1_800,
        ready_timeout=1_800,
        lookup_batch_size=128,
    )
    verified_dedup = StepSpec(
        name="datakit/verify_fuzzy_dups",
        deps=[*sources.values(), *minhash.values(), dedup],
        hash_attrs={
            "artifact_version": VERIFIED_FUZZY_DUPS_ATTR_DATA_VERSION,
            "verification": verification_params.model_dump(mode="json"),
            "local_representatives": REFERENCE_LOCAL_REPRESENTATIVE_PARAMS.model_dump(mode="json"),
        },
        fn=lambda op: verify_fuzzy_dups(
            normalized_sources={name: read_artifact(step.output_path, NormalizedData) for name, step in sources.items()},
            minhash_sources={name: read_artifact(step.output_path, MinHashAttrData) for name, step in minhash.items()},
            candidates=read_artifact(dedup.output_path, FuzzyDupsAttrData),
            output_path=op,
            verification_params=verification_params,
            local_representative_params=REFERENCE_LOCAL_REPRESENTATIVE_PARAMS,
            store_config=verification_store_config,
            max_workers=scale.pool.n_workers,
            worker_resources=scale.pool.worker,
        ),
    )

    return verified_dedup


def sft_filter_steps(
    sources: dict[str, DatakitChatSource],
    scale: PipelineScale = SMOKE_SCALE,
    zephyr_context: ZephyrContext | None = None,
) -> SftFilterSteps:
    """Build exact/fuzzy dedup and decontamination over rendered conversations.

    Source normalization depends on the canonical render step, so existing
    renders are reused by StepRunner. Only attribute stages are targeted;
    filtering rows into a training dataset is a separate downstream operation.
    """
    if not sources:
        raise ValueError("Select at least one SFT source")
    normalized = {name: source.normalized for name, source in sorted(sources.items())}
    dedup = zephyr_datakit_steps(normalized, scale, zephyr_context)
    _, _, decontam = decontamination_steps(normalized, scale, zephyr_context)
    verified = verified_dedup_step(normalized, dedup.minhash, dedup.fuzzy_dedup, scale)
    return SftFilterSteps(normalized, dedup.exact_dedup, verified, decontam)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", help="SFT registry names; omit for all sources")
    parser.add_argument("--execute", action="store_true", help="Run the DAG; otherwise print target artifact paths")
    parser.add_argument("--pool-workers", type=int, default=SMOKE_SCALE.pool.n_workers)
    parser.add_argument("--max-concurrent", type=int, default=4)
    args = parser.parse_args()
    if args.pool_workers < 1 or args.max_concurrent < 1:
        parser.error("worker and concurrency limits must be positive")
    configure_logging(logging.INFO)
    registry = all_sft_sources()
    names = args.sources if args.sources is not None else sorted(registry)
    unknown = sorted(set(names) - registry.keys())
    if unknown:
        parser.error(f"Unknown SFT sources: {', '.join(unknown)}")
    sources = {name: registry[name] for name in names}
    scale = replace(SMOKE_SCALE, pool=replace(SMOKE_SCALE.pool, n_workers=args.pool_workers))
    if not args.execute:
        result = sft_filter_steps(sources, scale)
        print(json.dumps({step.name: step.output_path for step in result.targets}, indent=2))
        return
    with ZephyrContext(
        name="datakit-sft-filter",
        resources=scale.pool.worker,
        max_workers=scale.pool.n_workers,
        stage_runner_factory=SubprocessRunner,
    ) as context:
        result = sft_filter_steps(sources, scale, context)
        StepRunner().run(result.targets, max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    main()

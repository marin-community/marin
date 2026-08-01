# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Datakit Testbed fuzzy-dedup variant — non-trivial dedup arm of the ranking protocol.

Shares the sample stage (``build_testbed_steps(...)``) with the
baseline and with other fuzzy-dedup variants so one set of sampled
parquet serves every hyperparam sweep. Each variant then runs global exact
deduplication and MinHash→fuzzy-dups→full-text verification, consolidates the sampled data,
tokenizes the deduped output, and trains.

The whole pipeline lives in one ``StepSpec`` graph that :class:`StepRunner`
walks, scheduling each step once its dependencies are satisfied.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence

from fray.types import ActorConfig, ResourceConfig
from marin.datakit.normalize import NormalizedData
from marin.datakit.source_key import datakit_source_key
from marin.execution.artifact import read_artifact
from marin.execution.lazy import ArtifactStep
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.classification.consolidate import FilterConfig, FilterType, consolidate
from marin.processing.classification.deduplication.fuzzy_dups import (
    FUZZY_DUPS_ATTR_DATA_VERSION,
    FuzzyDupsAttrData,
    compute_fuzzy_dups_attrs,
)
from marin.processing.classification.deduplication.fuzzy_minhash import MinHashAttrData, compute_minhash_attrs
from marin.processing.classification.deduplication.fuzzy_verification import FuzzyVerificationParams
from marin.processing.classification.deduplication.verify_fuzzy_dups import (
    REFERENCE_LOCAL_REPRESENTATIVE_PARAMS,
    VERIFIED_FUZZY_DUPS_ATTR_DATA_VERSION,
    FuzzyVerificationStoreConfig,
    VerifiedFuzzyDupsAttrData,
    verify_fuzzy_dups,
)
from marin.processing.tokenize.tokenize import TokenizedCache
from rigging.filesystem import prefix_join
from rigging.log_setup import configure_logging
from zephyr.execution import ZephyrExecutionResult

from experiments.datakit.global_exact_dedup import (
    GLOBAL_EXACT_DEDUP_DATA_VERSION,
    GlobalExactDedupData,
    global_exact_deduplicate,
)
from experiments.datakit.testbed.mixture import tokenized_bucket_weights_step
from experiments.datakit.testbed.sampler import build_testbed_steps
from experiments.datakit.testbed.settings import TESTBED_TOKENIZER
from experiments.datakit.testbed.train import run_testbed_config, testbed_tokenize
from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.uncheatable import uncheatable_datasets

logger = logging.getLogger(__name__)

STAGING_PREFIX = "gs://marin-us-central1"
TARGET_TOTAL_TOKENS_B = 1000.0
MAX_STEP_CONCURRENCY = 20

_SAMPLE_STEP_PREFIX = "data/datakit/normalized/"
_EXACT_DUPS_MAX_PARALLELISM = 128
_FUZZY_DUPS_MAX_PARALLELISM = 128
_EXACT_DUPS_WORKER_RESOURCES = ResourceConfig(cpu=2, ram="5g")
_MINHASH_WORKER_RESOURCES = ResourceConfig(cpu=2, ram="5g")
_FUZZY_DUPS_WORKER_RESOURCES = ResourceConfig(cpu=2, ram="5g")
_FUZZY_VERIFICATION_WORKER_RESOURCES = ResourceConfig(cpu=2, ram="8g")
_FUZZY_VERIFICATION_STORE_CONFIG = FuzzyVerificationStoreConfig(
    max_actors=32,
    actor_resources=ResourceConfig(cpu=2, ram="8g"),
    actor_config=ActorConfig(max_concurrency=32, max_task_retries=1_000),
    recovery_timeout=1_800,
    ready_timeout=1_800,
    lookup_batch_size=128,
)
_CONSOLIDATE_WORKER_RESOURCES = ResourceConfig(cpu=2, ram="5g")


def _minhash_step(src_name: str, sampled: StepSpec, **params: int) -> StepSpec:
    """MinHash bucket attrs for one sampled source."""
    return StepSpec(
        name=f"data/datakit/minhash/{src_name}",
        deps=[sampled],
        hash_attrs={
            "num_perms": params["num_perms"],
            "num_bands": params["num_bands"],
            "ngram_size": params["ngram_size"],
            "seed": params["seed"],
        },
        fn=lambda output_path, sampled=sampled: compute_minhash_attrs(
            source=read_artifact(sampled.output_path, NormalizedData),
            output_path=output_path,
            num_perms=params["num_perms"],
            num_bands=params["num_bands"],
            ngram_size=params["ngram_size"],
            seed=params["seed"],
            worker_resources=_MINHASH_WORKER_RESOURCES,
        ),
    )


def _exact_dups_step(sampled_by_source: dict[str, StepSpec]) -> StepSpec:
    """Mark later copies of each normalized content ID."""
    source_names = sorted(sampled_by_source)
    return StepSpec(
        name="data/datakit/global_exact_dedup",
        deps=[sampled_by_source[name] for name in source_names],
        hash_attrs={"sources": source_names, "v": GLOBAL_EXACT_DEDUP_DATA_VERSION},
        fn=lambda output_path: global_exact_deduplicate(
            sources={name: read_artifact(sampled_by_source[name].output_path, NormalizedData) for name in source_names},
            output_path=output_path,
            worker_resources=_EXACT_DUPS_WORKER_RESOURCES,
            max_workers=_EXACT_DUPS_MAX_PARALLELISM,
        ),
    )


def _fuzzy_dups_step(minhash_steps: list[StepSpec], cc_max_iterations: int) -> StepSpec:
    """Global fuzzy-dup cluster attrs across every source's MinHash."""
    return StepSpec(
        name="data/datakit/fuzzy_dups",
        deps=list(minhash_steps),
        hash_attrs={"artifact_version": FUZZY_DUPS_ATTR_DATA_VERSION, "cc_max_iterations": cc_max_iterations},
        fn=lambda output_path: compute_fuzzy_dups_attrs(
            inputs=[read_artifact(mh.output_path, MinHashAttrData) for mh in minhash_steps],
            output_path=output_path,
            cc_max_iterations=cc_max_iterations,
            max_parallelism=_FUZZY_DUPS_MAX_PARALLELISM,
            worker_resources=_FUZZY_DUPS_WORKER_RESOURCES,
        ),
    )


def _fuzzy_verification_step(
    sampled_by_source: dict[str, StepSpec],
    minhash_by_source: dict[str, StepSpec],
    fuzzy_dups: StepSpec,
) -> StepSpec:
    """Verify candidate members against retained local representatives."""
    params = FuzzyVerificationParams()
    return StepSpec(
        name="data/datakit/verify_fuzzy_dups",
        deps=[*sampled_by_source.values(), *minhash_by_source.values(), fuzzy_dups],
        hash_attrs={
            "artifact_version": VERIFIED_FUZZY_DUPS_ATTR_DATA_VERSION,
            "verification": params.model_dump(mode="json"),
            "local_representatives": REFERENCE_LOCAL_REPRESENTATIVE_PARAMS.model_dump(mode="json"),
        },
        fn=lambda output_path: verify_fuzzy_dups(
            normalized_sources={
                name: read_artifact(step.output_path, NormalizedData) for name, step in sampled_by_source.items()
            },
            minhash_sources={
                name: read_artifact(step.output_path, MinHashAttrData) for name, step in minhash_by_source.items()
            },
            candidates=read_artifact(fuzzy_dups.output_path, FuzzyDupsAttrData),
            output_path=output_path,
            verification_params=params,
            local_representative_params=REFERENCE_LOCAL_REPRESENTATIVE_PARAMS,
            store_config=_FUZZY_VERIFICATION_STORE_CONFIG,
            worker_resources=_FUZZY_VERIFICATION_WORKER_RESOURCES,
        ),
    )


def _consolidate_deduped(
    *,
    output_path: str,
    sampled: StepSpec,
    exact_dups: StepSpec,
    verified_dups: StepSpec,
) -> ZephyrExecutionResult:
    normalized = read_artifact(sampled.output_path, NormalizedData)
    source_key = datakit_source_key(normalized.main_output_dir)
    exact = read_artifact(exact_dups.output_path, GlobalExactDedupData)
    verified = read_artifact(verified_dups.output_path, VerifiedFuzzyDupsAttrData)
    return consolidate(
        input_path=normalized.main_output_dir,
        output_path=prefix_join(output_path, "outputs/main"),
        filetype="parquet",
        filters=[
            FilterConfig(
                type=FilterType.REMOVE_DOC,
                attribute_path=exact.sources[source_key].attr_dir,
                name="dup_doc",
                attribute_filetype="parquet",
                keep_if_missing=True,
            ),
            FilterConfig(
                type=FilterType.REMOVE_DOC,
                attribute_path=verified.attr_dir_for_source(normalized.main_output_dir),
                name="dup_doc",
                attribute_filetype="parquet",
                keep_if_missing=True,
            ),
        ],
        worker_resources=_CONSOLIDATE_WORKER_RESOURCES,
    )


def _deduped_step(src_name: str, sampled: StepSpec, exact_dups: StepSpec, verified_dups: StepSpec) -> StepSpec:
    """Per-source consolidate: remove exact and directly verified duplicates.

    Writes to ``{output_path}/outputs/main/part-*.parquet`` so the downstream
    tokenize's ``outputs/main/*.parquet`` glob picks it up unchanged.
    """
    return StepSpec(
        name=f"data/datakit/deduped/{src_name}",
        deps=[sampled, exact_dups, verified_dups],
        fn=lambda output_path, sampled=sampled: _consolidate_deduped(
            output_path=output_path,
            sampled=sampled,
            exact_dups=exact_dups,
            verified_dups=verified_dups,
        ),
    )


def dedup(
    steps: list[StepSpec],
    *,
    name: str,
    tokenizer: str,
    validation: Sequence[ArtifactStep[TokenizedCache]],
    fuzzy_dedup_num_perms: int = 286,
    fuzzy_dedup_num_bands: int = 26,
    fuzzy_dedup_ngram_size: int = 5,
    fuzzy_dedup_seed: int = 42,
    fuzzy_dedup_cc_max_iterations: int = 10,
) -> StepSpec:
    """Assemble the fuzzy-dedup training step off a testbed DAG.

    Defaults for ``fuzzy_dedup_*`` match
    :func:`marin.processing.classification.deduplication.fuzzy_minhash.compute_minhash_attrs`
    and :func:`marin.processing.classification.deduplication.fuzzy_dups.compute_fuzzy_dups_attrs`.
    """
    sampled_by_source = {
        s.name.removeprefix(_SAMPLE_STEP_PREFIX): s for s in steps if s.name.startswith(_SAMPLE_STEP_PREFIX)
    }
    if not sampled_by_source:
        raise ValueError("no sample steps found in the DAG (expected names under 'data/datakit/normalized/...')")

    minhash_params = {
        "num_perms": fuzzy_dedup_num_perms,
        "num_bands": fuzzy_dedup_num_bands,
        "ngram_size": fuzzy_dedup_ngram_size,
        "seed": fuzzy_dedup_seed,
    }
    minhash_by_source = {
        src_name: _minhash_step(src_name, sampled, **minhash_params) for src_name, sampled in sampled_by_source.items()
    }
    exact_dups = _exact_dups_step(sampled_by_source)
    fuzzy_dups = _fuzzy_dups_step(list(minhash_by_source.values()), fuzzy_dedup_cc_max_iterations)
    verified_dups = _fuzzy_verification_step(sampled_by_source, minhash_by_source, fuzzy_dups)
    deduped_by_source = {
        src_name: _deduped_step(src_name, sampled, exact_dups, verified_dups)
        for src_name, sampled in sampled_by_source.items()
    }

    logger.info(
        "fuzzy-dedup variant %s: %d sources → exact + minhash → fuzzy_dups → verification → consolidate. "
        "params=%s, cc_max=%d",
        name,
        len(sampled_by_source),
        minhash_params,
        fuzzy_dedup_cc_max_iterations,
    )

    tokenized_buckets = {
        src_name: testbed_tokenize(src_name, deduped, tokenizer) for src_name, deduped in deduped_by_source.items()
    }
    weights_step = tokenized_bucket_weights_step(name, tokenized_buckets)
    return run_testbed_config(
        name=name,
        tokenized_buckets=tokenized_buckets,
        weights_step=weights_step,
        validation=validation,
        tokenizer=tokenizer,
    )


def main() -> None:
    """Build the fuzzy-dedup DAG and run it."""
    os.environ.setdefault("MARIN_PREFIX", STAGING_PREFIX)

    tokenizer = TESTBED_TOKENIZER
    run_id = "fuzzy_dedup"
    validation = [*paloma_datasets(tokenizer=tokenizer).values(), *uncheatable_datasets(tokenizer=tokenizer).values()]

    testbed_steps = build_testbed_steps(target_total_tokens_b=TARGET_TOTAL_TOKENS_B)
    training_step = dedup(testbed_steps, name=run_id, tokenizer=tokenizer, validation=validation)
    StepRunner().run([training_step], max_concurrent=MAX_STEP_CONCURRENCY)


if __name__ == "__main__":
    configure_logging()
    main()

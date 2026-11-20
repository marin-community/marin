# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One required evaluation-corpus binding for Datakit PT and SFT preparation."""

from dataclasses import dataclass
from typing import TypedDict

from marin.datakit.decon import BestEffortEvalSuite, RequiredEvalManifest, RequiredEvalSuite
from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.storage_path import prefix_join

from experiments.datakit.decontam.pinned_references import (
    COVERAGE_MANIFEST,
    COVERAGE_RELATIVE,
    COVERAGE_VERSION,
    reference_catalog,
    reference_sources_sha256,
)
from experiments.datakit.decontam.prepare_eval_corpus import (
    AA_BENCHMARK_NAMES,
    AA_MANIFEST_RELATIVE,
    EVAL_CORPUS_VERSION,
    EVALS_RELATIVE,
    LMH_MANIFEST_RELATIVE,
    UNCHEATABLE_BENCHMARK_NAMES,
    UNCHEATABLE_MANIFEST_RELATIVE,
    required_lmh_tasks,
)


class BloomBindingOptions(TypedDict):
    required_eval_suites: tuple[RequiredEvalSuite, ...]
    required_eval_corpus_version: str
    best_effort_eval_suite: BestEffortEvalSuite
    additional_required_eval_manifests: tuple[RequiredEvalManifest, ...]


@dataclass(frozen=True)
class RequiredDatakitEvals:
    root: str
    core: tuple[RequiredEvalSuite, ...]
    version: str
    lmh: BestEffortEvalSuite
    additional: tuple[RequiredEvalManifest, ...]

    def bloom_options(self) -> BloomBindingOptions:
        return {
            "required_eval_suites": self.core,
            "required_eval_corpus_version": self.version,
            "best_effort_eval_suite": self.lmh,
            "additional_required_eval_manifests": self.additional,
        }


def required_eval_binding() -> RequiredDatakitEvals:
    root = prefix_join(marin_prefix(), EVALS_RELATIVE)
    catalog = reference_catalog()
    return RequiredDatakitEvals(
        root=root,
        version=EVAL_CORPUS_VERSION,
        core=(
            RequiredEvalSuite(prefix_join(root, AA_MANIFEST_RELATIVE), AA_BENCHMARK_NAMES),
            RequiredEvalSuite(prefix_join(root, UNCHEATABLE_MANIFEST_RELATIVE), UNCHEATABLE_BENCHMARK_NAMES),
        ),
        lmh=BestEffortEvalSuite(prefix_join(root, LMH_MANIFEST_RELATIVE), EVAL_CORPUS_VERSION, required_lmh_tasks()),
        additional=(
            RequiredEvalManifest(
                path=prefix_join(prefix_join(marin_prefix(), COVERAGE_RELATIVE), COVERAGE_MANIFEST),
                version=COVERAGE_VERSION,
                names=tuple(source.name for source in catalog),
                source_catalog_sha256=reference_sources_sha256(catalog),
            ),
        ),
    )

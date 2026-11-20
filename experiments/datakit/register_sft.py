# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Issue a Datakit SFT receipt from completed, pinned preparation artifacts."""

import argparse
import hashlib
from functools import partial

from finestore.sft import DatasetVariantInput, SftEvidenceReference, SftPreparationReceipt, SftStage
from marin.datakit.sft_registration import (
    SftEvalRequirement,
    SftRegistrationPlan,
    qualification_implementation_sha256,
    qualify_sft_variant,
)
from marin.datakit.sft_sources import DatakitChatSource, all_sft_sources
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.storage_path import StoragePath, prefix_join

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


def _pin(uri: str) -> SftEvidenceReference:
    with StoragePath(uri).open("rb") as stream:
        return SftEvidenceReference(uri=uri, sha256=hashlib.file_digest(stream, "sha256").hexdigest())


def required_sft_evals() -> dict[str, SftEvalRequirement]:
    """Bind the same required corpus and reference catalog as Datakit preparation."""
    root = prefix_join(marin_prefix(), EVALS_RELATIVE)
    catalog = reference_catalog()
    return {
        "aa": SftEvalRequirement(
            manifest=_pin(prefix_join(root, AA_MANIFEST_RELATIVE)), version=EVAL_CORPUS_VERSION, names=AA_BENCHMARK_NAMES
        ),
        "uncheatable": SftEvalRequirement(
            manifest=_pin(prefix_join(root, UNCHEATABLE_MANIFEST_RELATIVE)),
            version=EVAL_CORPUS_VERSION,
            names=UNCHEATABLE_BENCHMARK_NAMES,
        ),
        "lm_eval": SftEvalRequirement(
            manifest=_pin(prefix_join(root, LMH_MANIFEST_RELATIVE)),
            version=EVAL_CORPUS_VERSION,
            required_tasks=required_lmh_tasks(),
        ),
        "atlas_references": SftEvalRequirement(
            manifest=_pin(prefix_join(prefix_join(marin_prefix(), COVERAGE_RELATIVE), COVERAGE_MANIFEST)),
            version=COVERAGE_VERSION,
            names=tuple(source.name for source in catalog),
            source_catalog_sha256=reference_sources_sha256(catalog),
        ),
    }


def _register(plan: SftRegistrationPlan, output_path: str) -> SftPreparationReceipt:
    if plan.evals != required_sft_evals():
        raise ValueError("SFT registration does not use the current required eval corpus and reference catalog")
    return qualify_sft_variant(plan, prefix_join(output_path, "receipt.json"))


def registered_download(source: DatakitChatSource, identity: DatasetVariantInput) -> StepSpec:
    pending = [source.chat_normalized]
    downloads = {}
    seen = set()
    while pending:
        step = pending.pop()
        if step.output_path in seen:
            continue
        seen.add(step.output_path)
        if (step.hash_attrs.get("hf_dataset_id"), step.hash_attrs.get("revision")) == (
            identity.repository,
            identity.revision,
        ):
            downloads[step.output_path] = step
        pending.extend(step.deps)
    if len(downloads) != 1:
        raise ValueError("SFT source must have one download for the selected repository and revision")
    return next(iter(downloads.values()))


def registration_step(plan: SftRegistrationPlan, *, output_prefix: str | None = None) -> StepSpec:
    """Add qualification after the registered full-source normalization graph."""
    source = all_sft_sources()[plan.registry_source]
    expected = {
        SftStage.DOWNLOAD: registered_download(source, plan.input),
        SftStage.CHAT: source.chat_normalized,
        SftStage.RENDER: source.rendered,
        SftStage.NORMALIZE: source.normalized,
    }
    for stage, step in expected.items():
        if plan.stages[stage].uri != prefix_join(step.output_path, ".artifact.json"):
            raise ValueError(f"SFT {stage.value} evidence is not the registered source's artifact")
    return StepSpec(
        name=f"datakit/sft-registration/{plan.registry_source}",
        deps=[source.normalized],
        hash_attrs={
            "plan": plan.model_dump(mode="json"),
            "implementation_sha256": qualification_implementation_sha256(),
        },
        fn=partial(_register, plan),
        output_path_prefix=output_prefix,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="Pinned registration-plan JSON")
    parser.add_argument("--output-prefix")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    with StoragePath(args.plan).open() as stream:
        plan = SftRegistrationPlan.model_validate_json(stream.read())
    step = registration_step(plan, output_prefix=args.output_prefix)
    print(step.output_path)
    if args.run:
        StepRunner().run([step], max_concurrent=1)


if __name__ == "__main__":
    main()

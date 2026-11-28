# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a registered full source, then issue its Datakit SFT receipt."""

import argparse
from functools import partial

from finestore.sft import SftEvalFamily, SftEvidenceReference, SftPreparationReceipt, SftRecord, SftStage
from marin.datakit.decon import build_eval_bloom_step, decon_step
from marin.datakit.sft_registration import (
    SFT_MIN_MATCHED_FEATURES,
    SFT_NGRAM_LENGTH,
    TRAINING_MANIFEST_FILENAME,
    PreparedSftData,
    SftEvalRequirement,
    SftPreparationInput,
    SftRegistrationPlan,
    pin_file,
    prepare_sft_training,
    qualification_implementation_sha256,
    qualify_sft_variant,
)
from marin.datakit.sft_sources import DatakitChatSource, all_sft_sources
from marin.execution.artifact import RECORD_FILENAME, read_record
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.datakit.decontam.required_binding import required_eval_binding


class SftRegistrationRequest(SftRecord):
    registry_source: str
    source_id: str
    variant_id: str
    repository: str
    revision: str
    task_specs_sha256: str | None = None
    tokenizer: SftEvidenceReference
    chat_template: SftEvidenceReference


def required_sft_evals() -> dict[str, SftEvalRequirement]:
    binding = required_eval_binding()
    result = {
        family: SftEvalRequirement(
            manifest=pin_file(StoragePath(suite.manifest_path)), version=binding.version, names=suite.names
        )
        for family, suite in zip((SftEvalFamily.AA, SftEvalFamily.UNCHEATABLE), binding.core, strict=True)
    }
    result[SftEvalFamily.LM_EVAL] = SftEvalRequirement(
        manifest=pin_file(StoragePath(binding.lmh.manifest_path)),
        version=binding.lmh.corpus_version,
        required_tasks=binding.lmh.required_tasks,
    )
    supplement = binding.additional[0]
    result[SftEvalFamily.ATLAS_REFERENCES] = SftEvalRequirement(
        manifest=pin_file(StoragePath(supplement.path)),
        version=supplement.version,
        names=supplement.names,
        source_catalog_sha256=supplement.source_catalog_sha256,
    )
    return result


def registered_download(source: DatakitChatSource, repository: str, revision: str) -> StepSpec:
    pending = [source.chat_normalized]
    downloads = {}
    seen = set()
    while pending:
        step = pending.pop()
        if step.output_path in seen:
            continue
        seen.add(step.output_path)
        if (step.hash_attrs.get("hf_dataset_id"), step.hash_attrs.get("revision")) == (repository, revision):
            downloads[step.output_path] = step
        pending.extend(step.deps)
    if len(downloads) != 1:
        raise ValueError("SFT source must have one download for the selected repository and revision")
    return next(iter(downloads.values()))


def _training_output(filter_path: str, output_path: str) -> dict:
    prepared = PreparedSftData.model_validate(read_record(filter_path).result)
    (StoragePath(output_path) / TRAINING_MANIFEST_FILENAME).write_bytes(
        StoragePath(prepared.output_manifest.uri).read_bytes(), auto_mkdir=True
    )
    return prepared.model_dump(mode="json")


def _register(
    request: SftRegistrationRequest, stages: dict[SftStage, StepSpec], output_path: str
) -> SftPreparationReceipt:
    prepared = PreparedSftData.model_validate(read_record(stages[SftStage.FILTER].output_path).result)
    plan = SftRegistrationPlan(
        input=prepared.input,
        registry_source=request.registry_source,
        input_manifest=prepared.input_manifest,
        stages={stage: pin_file(StoragePath(step.output_path) / RECORD_FILENAME) for stage, step in stages.items()},
        stage_files=prepared.stage_files,
        evals=required_sft_evals(),
        output_manifest=prepared.output_manifest,
        tokenizer=request.tokenizer,
        chat_template=request.chat_template,
    )
    return qualify_sft_variant(plan, prefix_join(output_path, "receipt.json"))


def registration_step(request: SftRegistrationRequest, *, output_prefix: str | None = None) -> StepSpec:
    source = all_sft_sources()[request.registry_source]
    stages = {
        SftStage.DOWNLOAD: registered_download(source, request.repository, request.revision),
        SftStage.CHAT: source.chat_normalized,
        SftStage.RENDER: source.rendered,
        SftStage.NORMALIZE: source.normalized,
    }
    binding = required_eval_binding()
    bloom = build_eval_bloom_step(
        name="datakit/sft-eval-bloom",
        eval_data_sources=[binding.root],
        ngram_length=SFT_NGRAM_LENGTH,
        min_matched_features=SFT_MIN_MATCHED_FEATURES,
        **binding.bloom_options(),
        output_path_prefix=output_prefix,
    )
    marks = decon_step(
        name=f"datakit/sft-decontamination/{request.registry_source}",
        normalized=source.normalized,
        prebuilt_bloom=bloom,
        ngram_length=SFT_NGRAM_LENGTH,
        min_matched_features=SFT_MIN_MATCHED_FEATURES,
        output_path_prefix=output_prefix,
    )
    stages.update({SftStage.EVAL_BLOOM: bloom, SftStage.DECONTAM: marks})
    preparation = SftPreparationInput(
        source_id=request.source_id,
        variant_id=request.variant_id,
        repository=request.repository,
        revision=request.revision,
        task_specs_sha256=request.task_specs_sha256,
        stage_paths={stage: step.output_path for stage, step in stages.items()},
        tokenizer=request.tokenizer,
        chat_template=request.chat_template,
    )
    filtering = StepSpec(
        name=f"datakit/sft-filter/{request.registry_source}",
        deps=list(stages.values()),
        hash_attrs={"request": request.model_dump(mode="json"), "implementation": qualification_implementation_sha256()},
        fn=partial(prepare_sft_training, preparation),
        output_path_prefix=output_prefix,
    )
    training = StepSpec(
        name=f"datakit/sft-training/{request.registry_source}",
        deps=[filtering],
        hash_attrs={"request": request.model_dump(mode="json")},
        fn=partial(_training_output, filtering.output_path),
        output_path_prefix=output_prefix,
    )
    stages.update({SftStage.FILTER: filtering, SftStage.TRAINING: training})
    return StepSpec(
        name=f"datakit/sft-registration/{request.registry_source}",
        deps=[training],
        hash_attrs={"request": request.model_dump(mode="json"), "implementation": qualification_implementation_sha256()},
        fn=partial(_register, request, stages),
        output_path_prefix=output_prefix,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, help="Source identity and pinned tokenizer/template JSON")
    parser.add_argument("--output-prefix")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    request = SftRegistrationRequest.model_validate_json(StoragePath(args.request).read_bytes())
    step = registration_step(request, output_prefix=args.output_prefix)
    print(step.output_path)
    if args.run:
        StepRunner().run([step], max_concurrent=1)


if __name__ == "__main__":
    main()

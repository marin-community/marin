# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.sft import (
    REQUIRED_SFT_EVAL_FAMILIES,
    DatasetVariantInput,
    SftEvidenceReference,
    SftPreparationReceipt,
    SftStage,
)
from marin.datakit.chat_normalize import ChatChannel, _normalize_chat_record
from marin.datakit.chat_render import render_chat_record
from marin.datakit.normalize import _make_normalize_fn
from marin.datakit.sft_registration import (
    SftEvalRequirement,
    SftPreparationInput,
    SftRegistrationPlan,
    prepare_sft_training,
    qualify_sft_variant,
)
from marin.execution.artifact import ArtifactRecord
from marin.execution.step_status import StatusFile
from openai_harmony import Message, Role
from rigging.filesystem.step_status import STATUS_FAILED, STATUS_SUCCESS
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


def pin(path: Path) -> SftEvidenceReference:
    return SftEvidenceReference(uri=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def json_file(path: Path, value) -> SftEvidenceReference:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")
    return pin(path)


def parquet(path: Path, rows: list[dict]) -> SftEvidenceReference:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return pin(path)


def artifact(path: Path, value: dict | None, deps=(), config=None) -> SftEvidenceReference:
    record = ArtifactRecord(
        name=path.name, output_path=str(path), dep_paths=[str(dep) for dep in deps], result=value, config=config
    )
    StatusFile(str(path), worker_id="test").write_status(STATUS_SUCCESS)
    return json_file(path / ".artifact.json", record.model_dump(mode="json"))


def _tokenizer_fixture(tmp_path: Path) -> tuple[SftEvidenceReference, SftEvidenceReference]:
    token_dir = tmp_path / "tokenizer"
    backend = Tokenizer(
        WordLevel({"[UNK]": 0, "Question": 1, "Answer": 2, "Different": 3, "Excluded": 4}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.save_pretrained(token_dir)
    template_path = tmp_path / "template.jinja"
    template_path.write_text(
        "{% for message in messages %}{% if message['role'] == 'assistant' %}"
        "{% generation %}{{message['content']}}{% endgeneration %}"
        "{% else %}{{message['content']}} {% endif %}{% endfor %}"
    )
    template = pin(template_path)
    tokenizer_manifest = json_file(
        tmp_path / "tokenizer.json", {"files": [pin(f).model_dump() for f in sorted(token_dir.iterdir())], "rows": 0}
    )
    return tokenizer_manifest, template


def _eval_fixtures(tmp_path: Path) -> dict:
    evals = {}
    for family in sorted(REQUIRED_SFT_EVAL_FAMILIES):
        root = tmp_path / "evals" / family
        data = parquet(root / "data.parquet", [{"id": "reference", "text": "one two three four five six seven"}])
        if family == "lm_eval":
            manifest = json_file(
                root / "manifest.json",
                {
                    "corpus_version": "fixture-v1",
                    "required": False,
                    "status": "complete",
                    "included_leaf_tasks": ["core"],
                    "required_tasks": {"core": ["core"]},
                    "artifacts": [{"task": "core", "artifact": "data.parquet", "records": 1}],
                },
            )
            evals[family] = SftEvalRequirement(manifest=manifest, version="fixture-v1", required_tasks={"core": 1})
        else:
            manifest = json_file(
                root / "manifest.json",
                {
                    "corpus_version": "fixture-v1",
                    "required": True,
                    "status": "complete",
                    "benchmarks": [
                        {"name": family, "artifact": "data.parquet", "expected_records": 1, "sha256": data.sha256}
                    ],
                },
            )
            evals[family] = SftEvalRequirement(manifest=manifest, version="fixture-v1", names=(family,))
    return evals


@dataclass(frozen=True)
class RegistrationFixture:
    plan: SftRegistrationPlan
    root: Path
    training_row: dict


@pytest.fixture(params=["parquet", "jsonl"])
def registration(tmp_path, request):
    raw = parquet(
        tmp_path / "download/data.parquet",
        [{"prompt": "Question", "answer": "Answer"}, {"prompt": "Different", "answer": "Excluded"}],
    )
    if request.param == "jsonl":
        path = tmp_path / "download/data.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in pq.read_table(raw.uri).to_pylist()) + "\n")
        Path(raw.uri).unlink()
        raw = pin(path)
    input_manifest = json_file(tmp_path / "input.json", {"files": [raw.model_dump()], "rows": 2})
    identity = DatasetVariantInput(
        source_id="Fixture:source",
        variant_id="chat-v1",
        repository="fixture/source",
        revision="a" * 40,
        data_sha256=input_manifest.sha256,
        task_specs_sha256="b" * 64,
    )
    chats = []
    for question, answer in [("Question", "Answer"), ("Different", "Excluded")]:
        messages = [
            Message.from_role_and_content(Role.USER, question),
            Message.from_role_and_content(Role.ASSISTANT, answer).with_channel(ChatChannel.FINAL),
        ]
        chats.append(_normalize_chat_record({"messages": [m.to_dict() for m in messages]}, "messages", "id"))
    parquet(tmp_path / "chat/outputs/main/part-000.parquet", chats)
    normalized = [_make_normalize_fn("text", "id")(render_chat_record(row)) for row in chats]
    parquet(tmp_path / "normalize/outputs/main/part-000.parquet", normalized)
    parquet(
        tmp_path / "decontam/outputs/main/part-000.parquet",
        [{"id": row["id"], "contaminated": index == 1} for index, row in enumerate(normalized)],
    )
    prepared = {**chats[0], "normalized_id": normalized[0]["id"], "input_ids": [1, 2], "assistant_masks": [0, 1]}
    training_file = parquet(tmp_path / "training/part-000.parquet", [prepared])
    output = json_file(tmp_path / "output.json", {"files": [training_file.model_dump()], "rows": 1})
    tokenizer_manifest, template = _tokenizer_fixture(tmp_path)
    evals = _eval_fixtures(tmp_path)
    paths = {stage: tmp_path / stage.name.lower() for stage in SftStage}
    paths[SftStage.CHAT] = tmp_path / "chat"
    paths[SftStage.NORMALIZE] = tmp_path / "normalize"
    paths[SftStage.DECONTAM] = tmp_path / "decontam"
    paths[SftStage.TRAINING] = tmp_path / "training"
    policy = {
        "text_field": "text",
        "ngram_length": 5,
        "overlap_threshold": 0.5,
        "min_matched_features": 1,
        "paragraph_delimiter": "\n\n",
        "feature_filter_version": "fixture-v1",
    }
    stage_files = {
        stage: json_file(
            tmp_path / f"{stage.name}-files.json",
            {"files": [pin(paths[stage] / "outputs/main/part-000.parquet").model_dump()], "rows": 2},
        )
        for stage in [SftStage.CHAT, SftStage.NORMALIZE, SftStage.DECONTAM]
    }
    stages = {
        SftStage.DOWNLOAD: artifact(
            paths[SftStage.DOWNLOAD], None, config={"hf_dataset_id": identity.repository, "revision": identity.revision}
        ),
        SftStage.CHAT: artifact(
            paths[SftStage.CHAT],
            {
                "main_output_dir": str(tmp_path / "chat/outputs/main"),
                "dup_output_dir": str(tmp_path / "chat/dups"),
                "counters": {"conversion": {}, "normalization": {"normalize_chat/records_validated": 2}},
            },
            [paths[SftStage.DOWNLOAD]],
        ),
        SftStage.RENDER: artifact(paths[SftStage.RENDER], None, [paths[SftStage.CHAT]]),
        SftStage.NORMALIZE: artifact(
            paths[SftStage.NORMALIZE],
            {
                "main_output_dir": str(tmp_path / "normalize/outputs/main"),
                "dup_output_dir": str(tmp_path / "normalize/dups"),
                "counters": {},
            },
            [paths[SftStage.RENDER]],
        ),
        SftStage.EVAL_BLOOM: artifact(
            paths[SftStage.EVAL_BLOOM],
            {
                "bloom_dir": str(paths[SftStage.EVAL_BLOOM]),
                "bloom_path": "fixture.bin",
                "eval_hash_index_path": "fixture.parquet",
                "estimated_doc_count": 4,
                "false_positive_rate": 0.000001,
                "n_eval_records": 4,
            },
            config={
                **policy,
                "required_eval_suites": [["aa"], ["uncheatable"]],
                "required_eval_corpus_version": "fixture-v1",
                "best_effort_eval_corpus_version": "fixture-v1",
                "best_effort_required_tasks": [["core", 1]],
                "additional_required_eval_manifests": [
                    [evals["atlas_references"].manifest.uri, "fixture-v1", ["atlas_references"], None]
                ],
            },
        ),
        SftStage.DECONTAM: artifact(
            paths[SftStage.DECONTAM],
            {
                "main_output_dir": str(tmp_path / "decontam/outputs/main"),
                "flagged_output_dir": "",
                "num_partitions": 1,
                "eval_hash_index_path": "",
                "counters": {},
            },
            [paths[SftStage.NORMALIZE], paths[SftStage.EVAL_BLOOM]],
            config=policy,
        ),
        SftStage.FILTER: artifact(
            paths[SftStage.FILTER],
            {
                "input": identity.model_dump(mode="json"),
                "input_rows": 2,
                "filtered_rows": 1,
                "output_manifest": output.model_dump(),
                "input_manifest": input_manifest.model_dump(),
                "stage_files": {stage.value: ref.model_dump() for stage, ref in stage_files.items()},
            },
            [paths[SftStage.DECONTAM]],
        ),
    }
    stages[SftStage.TRAINING] = artifact(
        paths[SftStage.TRAINING],
        json.loads(Path(stages[SftStage.FILTER].uri).read_text())["result"],
        [paths[SftStage.FILTER]],
    )
    plan = SftRegistrationPlan(
        input=identity,
        registry_source="fixture",
        input_manifest=input_manifest,
        stages=stages,
        stage_files=stage_files,
        evals=evals,
        output_manifest=output,
        tokenizer=tokenizer_manifest,
        chat_template=template,
    )
    return RegistrationFixture(plan, tmp_path, prepared)


def test_registration_writes_only_content_bound_complete_receipt(registration):
    plan, root = registration.plan, registration.root
    result = qualify_sft_variant(plan, str(root / "receipt.json"))
    loaded = SftPreparationReceipt.model_validate_json((root / "receipt.json").read_text())
    assert loaded == result
    assert (
        loaded.input_rows,
        loaded.training_rows,
        loaded.filtered_rows,
        loaded.training_tokens,
        loaded.supervised_tokens,
    ) == (2, 1, 1, 2, 1)
    assert qualify_sft_variant(plan, str(root / "receipt.json")) == loaded
    with pytest.raises(ValueError, match="selected dataset variant"):
        loaded.require_input(plan.input.model_copy(update={"task_specs_sha256": "c" * 64}))


@pytest.mark.parametrize(
    "defect",
    [
        "unfinished_stage",
        "changed_content",
        "wrong_mask",
        "unmeasured_row",
        "contaminated_row",
        "dropped_clean",
        "overwritten_marks",
        "wrong_population",
    ],
)
def test_incomplete_or_changed_preparation_never_writes_receipt(registration, defect):
    plan, root = registration.plan, registration.root
    prepared = dict(registration.training_row)
    if defect == "unfinished_stage":
        StatusFile(str(root / "filter"), worker_id="test").write_status(STATUS_FAILED)
    elif defect == "changed_content":
        Path(json.loads(Path(plan.input_manifest.uri).read_text())["files"][0]["uri"]).write_bytes(b"changed input")
    elif defect == "wrong_population":
        input_value = json.loads(Path(plan.input_manifest.uri).read_text())
        input_value["rows"] = 9
        reference = json_file(Path(plan.input_manifest.uri), input_value)
        plan = plan.model_copy(
            update={
                "input_manifest": reference,
                "input": plan.input.model_copy(update={"data_sha256": reference.sha256}),
            }
        )
    elif defect in {"contaminated_row", "overwritten_marks"}:
        path = root / "decontam/outputs/main/part-000.parquet"
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            row["contaminated"] = defect == "contaminated_row"
        parquet(path, rows)
    elif defect == "dropped_clean":
        path = root / "decontam/outputs/main/part-000.parquet"
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            row["contaminated"] = False
        parquet(path, rows)
    else:
        if defect == "wrong_mask":
            prepared["assistant_masks"] = [1, 0]
        else:
            prepared["normalized_id"] = "unmeasured-text-id"
        parquet(root / "training/part-000.parquet", [prepared])
        output = json_file(
            root / "output.json", {"files": [pin(root / "training/part-000.parquet").model_dump()], "rows": 1}
        )
        stages = dict(plan.stages)
        record = json.loads(Path(stages[SftStage.FILTER].uri).read_text())
        record["result"]["output_manifest"] = output.model_dump()
        stages[SftStage.FILTER] = json_file(Path(stages[SftStage.FILTER].uri), record)
        plan = plan.model_copy(update={"output_manifest": output, "stages": stages})
    if defect in {"contaminated_row", "dropped_clean"}:
        stage_files = dict(plan.stage_files)
        stage_files[SftStage.DECONTAM] = json_file(
            root / "new-marks-manifest.json",
            {"files": [pin(root / "decontam/outputs/main/part-000.parquet").model_dump()], "rows": 2},
        )
        plan = plan.model_copy(update={"stage_files": stage_files})
    if defect in {"wrong_mask", "unmeasured_row", "contaminated_row", "dropped_clean"}:
        stages = dict(plan.stages)
        record = json.loads(Path(stages[SftStage.FILTER].uri).read_text())
        record["result"]["output_manifest"] = plan.output_manifest.model_dump(mode="json")
        record["result"]["stage_files"] = {
            stage.value: ref.model_dump(mode="json") for stage, ref in plan.stage_files.items()
        }
        stages[SftStage.FILTER] = json_file(Path(stages[SftStage.FILTER].uri), record)
        training = json.loads(Path(stages[SftStage.TRAINING].uri).read_text())
        training["result"] = record["result"]
        stages[SftStage.TRAINING] = json_file(Path(stages[SftStage.TRAINING].uri), training)
        plan = plan.model_copy(update={"stages": stages})
    expected = {
        "wrong_mask": "tokens or assistant masks",
        "unmeasured_row": "unmeasured/contaminated text",
        "contaminated_row": "unmeasured/contaminated text",
        "dropped_clean": "discarded a distinct clean conversation",
    }.get(defect)
    with pytest.raises(ValueError, match=expected):
        qualify_sft_variant(plan, str(root / "receipt.json"))
    assert not (root / "receipt.json").exists()


def test_preparation_produces_training_selection_and_qualifies_it(registration):
    plan, root = registration.plan, registration.root
    configuration = SftPreparationInput(
        source_id=plan.input.source_id,
        variant_id=plan.input.variant_id,
        repository=plan.input.repository,
        revision=plan.input.revision,
        task_specs_sha256=plan.input.task_specs_sha256,
        stage_paths={stage: str(Path(ref.uri).parent) for stage, ref in plan.stages.items()},
        tokenizer=plan.tokenizer,
        chat_template=plan.chat_template,
    )
    result = prepare_sft_training(configuration, str(root / "actual-preparation"))
    stages = dict(plan.stages)
    stages[SftStage.FILTER] = artifact(
        root / "filter", result.model_dump(mode="json"), [Path(stages[SftStage.DECONTAM].uri).parent]
    )
    stages[SftStage.TRAINING] = artifact(root / "training", result.model_dump(mode="json"), [root / "filter"])
    completed = plan.model_copy(
        update={
            "input": result.input,
            "input_manifest": result.input_manifest,
            "output_manifest": result.output_manifest,
            "stage_files": result.stage_files,
            "stages": stages,
        }
    )
    receipt = qualify_sft_variant(completed, str(root / "completed-receipt.json"))
    assert (receipt.input_rows, receipt.training_rows, receipt.filtered_rows) == (2, 1, 1)

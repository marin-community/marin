# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rolloutengine.models import RolloutLimits, TaskRolloutSpec
from taskcompendium.models import (
    ConversationInput,
    EnvironmentRequirements,
    Source,
    TaskSpec,
    TextMessage,
    TokenizedInput,
    VerifyitGrader,
)
from taskcompendium.pipeline.fingerprints import semantic_digest
from taskcompendium.runtime.resources import inline_resource
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast


@pytest.mark.manual
def test_reference_panel_preparation_changes_only_outer_deadlines_and_refuses_overwrite(tmp_path):
    from experiments.post_training.snowball_inventory_evaluate_runtime import (  # noqa: PLC0415 - optional frozen runtime
        prepare_reference_panel,
        require_reference_limits,
    )

    rows = []
    for index, context in enumerate(
        [
            ConversationInput(events=(TextMessage(role="user", content="Complete the structure"),)),
            TokenizedInput(
                events=(TextMessage(role="user", content="Exact supplied prefix"),),
                token_ids=(128000, 42, 128005),
                tokenizer_model="pinned/reference-tokenizer",
                tokenizer_revision="a" * 40,
            ),
        ]
    ):
        task = TaskSpec(
            id=f"paired-{index}",
            context=context,
            environment_requirements=EnvironmentRequirements(),
            answer_type="text",
            answer_format={"kind": "plain_text"},
            source=Source(dataset="original", revision="frozen", row=str(index), importer_revision="1"),
            grader=VerifyitGrader(
                mode="structure_closure",
                parameters={"prefix": '{"x":', "expected": '{"x":1}', "format": "json", "terminal_markers": []},
            ),
            resources={"verifier": (inline_resource("private-reference.json", b'{"hidden":"unaltered"}'),)},
        )
        spec = TaskRolloutSpec(
            task=task,
            harness="token_prefix" if index else "shellbox",
            task_machine=None,
            verifier_machine=None,
            limits=RolloutLimits(
                max_turns=1,
                max_total_tokens=None,
                model_turn_timeout=180,
                tool_turn_timeout=30,
                total_turn_timeout=240,
                attempt_timeout=300,
                verifier_timeout=900,
                cleanup_timeout=30,
            ),
        )
        rows.append(
            {
                "task_id": task.id,
                "task_rollout_spec": spec.model_dump_json(),
                "contract_id": semantic_digest(task, True),
                "public_content_id": semantic_digest(task, False),
                "source_contract_id": f"original-contract-{index}",
                "evaluation_family": "structural",
                "lowered_task_spec": "byte-exact original source payload",
            }
        )
    source, derived = tmp_path / "original.parquet", tmp_path / "derived.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source)
    original_bytes = source.read_bytes()
    report = prepare_reference_panel(
        source,
        derived,
        source_sha256=hashlib.sha256(original_bytes).hexdigest(),
        family_counts={"structural": 2},
    )
    written = pq.read_table(derived).to_pylist()
    for before, after in zip(rows, written, strict=True):
        original = TaskRolloutSpec.model_validate_json(before["task_rollout_spec"])
        updated = TaskRolloutSpec.model_validate_json(after["task_rollout_spec"])
        assert updated.task == original.task
        assert updated.limits.tool_turn_timeout == original.limits.tool_turn_timeout
        assert updated.limits.verifier_timeout == original.limits.verifier_timeout
        assert updated.limits.cleanup_timeout == original.limits.cleanup_timeout
        assert updated.limits.model_turn_timeout == 1800
        assert updated.limits.total_turn_timeout == 2400
        assert updated.limits.attempt_timeout == 3600
        require_reference_limits(updated)
        with pytest.raises(ValueError, match="frozen panel"):
            require_reference_limits(original)
        assert {key: value for key, value in after.items() if key != "task_rollout_spec"} == {
            key: value for key, value in before.items() if key != "task_rollout_spec"
        }
    assert source.read_bytes() == original_bytes
    assert report["row_count"] == 2
    assert [row["task_id"] for row in report["runtime_mappings"]] == ["paired-0", "paired-1"]
    with pytest.raises(FileExistsError):
        prepare_reference_panel(
            source,
            derived,
            source_sha256=report["source_sha256"],
            family_counts={"structural": 2},
        )


@pytest.mark.manual
def test_cpu_preflight_tokenizer_loads_without_staged_model(tmp_path):
    # Run this optional-dependency regression in the frozen MarinSkyRL CPU environment.
    from omegaconf import OmegaConf  # noqa: PLC0415 - optional frozen runtime dependency

    from experiments.post_training.snowball_inventory_evaluate_runtime import preflight_tokenizer  # noqa: PLC0415

    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1, "world": 2, "[EOS]": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    source = tmp_path / "pinned-tokenizer"
    PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]").save_pretrained(source)
    staged = tmp_path / "not-yet-staged-model"
    config = OmegaConf.create(
        {
            "inputs": {"model": {"tokenizer_uri": str(source), "tokenizer_revision": "a" * 40}},
            "skyrl": {
                "trainer": {
                    "disable_fast_tokenizer": False,
                    "policy": {"model": {"tokenizer_path": str(staged), "tokenizer_revision": "a" * 40}},
                }
            },
        }
    )
    original = OmegaConf.to_container(config, resolve=True)

    tokenizer = preflight_tokenizer(config)

    assert tokenizer.encode("hello world", add_special_tokens=False) == [1, 2]
    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token_id == 3
    assert not staged.exists()
    assert OmegaConf.to_container(config, resolve=True) == original

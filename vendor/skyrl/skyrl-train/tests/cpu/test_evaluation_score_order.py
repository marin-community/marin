# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import hashlib
from itertools import permutations
from types import SimpleNamespace
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.rl.strata import MembershipState, SelectionDefinition, SelectionTaskRow
from finestore.schema import arrow_schema
from taskcompendium.identity import canonical_sha256
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from rolloutengine.models import ModelTurn
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import ExitCodeReward, ScriptGrader

from skyrl_train.batch_sampling import filter_trajectory_batch
from skyrl_train.config.utils import get_default_config
from skyrl_train.dataset.tasks import write_tasks
from skyrl_train.entrypoints.main_base import BasePPOExp
from skyrl_train.entrypoints.terminal_bench import TerminalBenchExp
from skyrl_train.evaluation_profiles import resolve_evaluation_profiles
from skyrl_train.rollouts.task_projections import StepTaskProjection, WholeTaskProjection
from skyrl_train.rollouts.training_record import TrainingRollout, TrainingStep, TrainingTransition
from skyrl_train.trajectory_runners.trajectory_processing import (
    concatenate_trajectory_batches,
    get_metrics_from_trajectory_batch,
)
from skyrl_train.trajectory_runners.types import TaskRequestBatch, TrajectoryID
from skyrl_train.utils.trainer_utils import evaluation_response_metrics
from tests.cpu.task_specs import NATIVE_ENVIRONMENT, lowered_task, task_input, text_task
from tests.cpu.tiny_training.tiny_model import build_tiny_policy


@pytest.mark.parametrize("experiment_type", [BasePPOExp, TerminalBenchExp])
def test_final_profiles_build_primary_and_named_datasets_without_periodic_evaluation(tmp_path, experiment_type):
    config = get_default_config()
    primary, named = tmp_path / "primary" / "tasks.parquet", tmp_path / "named" / "tasks.parquet"
    task = text_task("shared").model_copy(update={"tags": ("agentic",)})
    digest = canonical_sha256(task.model_dump(mode="json"))
    for path in (primary, named):
        path.parent.mkdir()
        if experiment_type is BasePPOExp and path == primary:
            pq.write_table(
                pa.table(
                    {
                        "task_spec": [task.model_dump_json()],
                        "family": ["primary"],
                        "aai_category": ["general"],
                        "source": ["declared-source"],
                        "topic": ["algebra"],
                        "difficulty": ["hard"],
                    }
                ),
                path,
            )
        else:
            write_tasks(path, [lowered_task(task)])
        pq.write_table(
            pa.Table.from_pylist([{"task_spec": task.model_dump_json()}]), path.parent / "validation.parquet"
        )
        row = SelectionTaskRow(
            task_id=task.id,
            task_digest=digest,
            family=path.parent.name,
            source="declared-source",
            aai_category="general",
            topic="algebra",
            difficulty="hard",
            agentic=True,
            state=MembershipState.ACTIVE,
        )
        pq.write_table(
            pa.Table.from_pylist([row.model_dump(mode="json")], schema=arrow_schema(SelectionTaskRow)),
            path.parent / "task_manifest.parquet",
        )
        (path.parent / "summary.json").write_text(
            json.dumps(
                {
                    "selection_strata": SelectionDefinition(
                        name=path.parent.name, version="1", sampling="all", task_files=("validation.parquet",)
                    ).model_dump(mode="json")
                }
            )
        )
    config.data.val_data = [str(primary)]
    config.data.task_cache_dir = str(tmp_path / "task-cache")
    if experiment_type is TerminalBenchExp:
        OmegaConf.update(config, "terminal_bench_config", {}, force_add=True)
    config.trainer.eval_interval = 0
    config.trainer.callbacks = [
        {
            "type": "evaluation",
            "eval_steps": 0,
            "additional_evaluations": {
                "primary_final": {"schedule": "final"},
                "named_final": {"schedule": "final", "data_files": [str(named)]},
            },
        }
    ]
    experiment = experiment_type.__new__(experiment_type)
    experiment.cfg = config
    original_config = OmegaConf.to_container(config, resolve=True)
    experiment.evaluation_profiles = resolve_evaluation_profiles(config)
    experiment.tokenizer = AutoTokenizer.from_pretrained(build_tiny_policy(tmp_path / "model"))

    primary_dataset = experiment.get_eval_dataset()
    named_datasets = experiment.get_named_eval_datasets()

    assert primary_dataset is not None
    assert [row["input"].spec.task.id for row in primary_dataset.collate_fn(list(primary_dataset))] == ["shared"]
    named_dataset = named_datasets["named_final"]
    assert [row["input"].spec.task.id for row in named_dataset.collate_fn(list(named_dataset))] == ["shared"]
    if experiment_type is TerminalBenchExp:
        assert (primary_dataset.uid(0), named_dataset.uid(0)) == ("shared", "shared")

    identities = [
        dataset.collate_fn(list(dataset))[0]["input"].metadata for dataset in (primary_dataset, named_dataset)
    ]
    assert [
        (
            item.evaluation_identity.panel_name,
            item.evaluation_identity.strata.family,
            item.evaluation_identity.strata.source,
            item.evaluation_identity.strata.aai_category,
            item.evaluation_identity.strata.topic,
            item.evaluation_identity.strata.difficulty,
        )
        for item in identities
    ] == [
        ("primary", "primary", "declared-source", "general", "algebra", "hard"),
        ("named", "named", "declared-source", "general", "algebra", "hard"),
    ]
    assert all(item.data_source != "declared-source" for item in identities)
    assert all(item.evaluation_identity.strata.agentic for item in identities)
    assert all(item.evaluation_identity.task_digest == digest for item in identities)
    assert OmegaConf.to_container(config, resolve=True) == original_config
    config.data.val_data = [str(primary), str(named)]
    with pytest.raises(ValueError, match="conflicting selection identities"):
        experiment.get_eval_dataset()
    if experiment_type is TerminalBenchExp:
        package = tmp_path / "packages" / "shared"
        (package / "environment").mkdir(parents=True)
        (package / "tests").mkdir()
        (package / "tests/test.sh").write_text("#!/bin/sh\nexit 0\n")
        (package / "instruction.md").write_text("Write an answer.")
        (package / "task.toml").write_text(
            f'[verifier]\nenvironment_mode = "separate"\n[environment]\ndocker_image = "{NATIVE_ENVIRONMENT.docker_image}"\n'
        )
        config.trainer.max_prompt_length = 4096
        package_dataset = experiment.get_eval_dataset([str(package.parent)])
        package_input = package_dataset.collate_fn(list(package_dataset))[0]["input"]
        assert package_input.spec.task.id == "shared" and package_input.metadata.evaluation_identity is None
        with pytest.raises(ValueError, match="conflicting selection identities"):
            experiment.get_eval_dataset([str(primary), str(package.parent)])
    else:
        raw = {
            "prompt": [{"role": "user", "content": "Count cats."}],
            "env_class": "cat_count",
            "data_source": "original",
            "extra_info": {"n": 1},
        }
        source_file = tmp_path / "raw.jsonl"
        source_file.write_text(json.dumps(raw) + "\n")
        source_dataset = experiment.get_eval_dataset([str(source_file)])
        source_task = source_dataset.collate_fn(list(source_dataset))[0]["input"].spec.task
        payload = json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        assert source_task.source.revision == "sha256:" + hashlib.sha256(payload).hexdigest()
        original = pq.read_table(primary)
        changed = task.model_copy(update={"tags": ()})
        pq.write_table(pa.table({"task_spec": [changed.model_dump_json()]}), primary)
        with pytest.raises(ValueError, match="task content differs from selection manifest"):
            experiment.get_eval_dataset([str(primary)])
        mismatched = original.set_column(
            original.schema.get_field_index("family"), "family", pa.array(["wrong-family"])
        )
        pq.write_table(mismatched, primary)
        with pytest.raises(ValueError, match="row strata differ from selection manifest"):
            experiment.get_eval_dataset([str(primary)])
        pq.write_table(original, primary)
        membership_path = primary.parent / "validation.parquet"
        membership = pq.read_table(membership_path)
        changed_task = json.loads(membership["task_spec"][0].as_py())
        changed_task["id"] = "edited"
        pq.write_table(pa.table({"task_spec": [json.dumps(changed_task)]}), membership_path)
        with pytest.raises(ValueError, match="manifest population differs"):
            experiment.get_eval_dataset([str(primary)])
        pq.write_table(membership, membership_path)
    config.data.val_data = [str(primary)]
    (primary.parent / "summary.json").write_text('{"selection_strata": null}')
    with pytest.raises(ValueError, match="SelectionDefinition"):
        experiment.get_eval_dataset()


@pytest.mark.parametrize("per_token", [False, True])
def test_evaluation_mean_survives_cancellation_and_completion_order(per_token):
    means = set()
    for values in permutations((1e16, 1.0, -1e16, 2.0)):
        rewards = [list(values)] if per_token else list(values)
        batch = {"rewards": rewards, "unshaped_rewards": [1.0] * len(rewards)}
        mean, passed = get_metrics_from_trajectory_batch(batch, [str(i) for i in range(len(rewards))])
        means.add(mean)
        assert passed == 1.0
    assert means == ({3.0} if per_token else {0.75})


def test_task_timings_survive_projection_concatenation_and_row_selection():
    config = get_default_config().generator
    tokenizer = SimpleNamespace(eos_token_id=2)
    inputs, rollouts = [], []
    turn = ModelTurn(
        message={"role": "assistant", "content": "answer"},
        prompt_token_ids=(1,),
        response_token_ids=(2,),
        logprobs=(-0.1,),
        stop_reason="stop",
    )
    for index, duration in enumerate((1.0, 3.0, 5.0, 101.0, None)):
        turn_count = 1 if duration == 101.0 else 2
        task = text_task(str(index)).model_copy(
            update={"grader": ScriptGrader(argv=("true",), environment=NATIVE_ENVIRONMENT, reward=ExitCodeReward())}
        )
        inputs.append(
            task_input(
                lowered_task(task, verifier_backend="local"),
                metric_class="taskcompendium" if index % 2 else "other",
            )
        )
        turns = (turn,) * turn_count
        rollouts.append(
            TrainingRollout(
                task_id=task.id,
                messages=(),
                prompt_token_ids=(1,),
                response_token_ids=(2,) * turn_count,
                loss_mask=(1,) * turn_count,
                logprobs=(-0.1,) * turn_count,
                grade=GradeResult(Outcome.GRADED, 1.0),
                stop_reason="stop",
                steps=tuple(
                    TrainingStep(
                        turn,
                        TrainingTransition(done=i == turn_count - 1, reward=float(i == turn_count - 1)),
                        i,
                        (),
                    )
                    for i, turn in enumerate(turns)
                ),
                metrics={} if duration is None else {"timing/grading": duration, "timing/setup": duration - 1},
            )
        )
    request = TaskRequestBatch(
        rows=tuple(inputs), trajectory_ids=tuple(TrajectoryID(str(i), 0) for i in range(len(inputs)))
    )
    whole = WholeTaskProjection(config, tokenizer)
    batches = [
        whole.project(rollouts[:1], replace(request, rows=request.rows[:1], trajectory_ids=request.trajectory_ids[:1])),
        whole.project(rollouts[1:], replace(request, rows=request.rows[1:], trajectory_ids=request.trajectory_ids[1:])),
    ]
    combined = concatenate_trajectory_batches(batches, tis_lcs_alert_threshold=0.05)
    combined = concatenate_trajectory_batches([combined], tis_lcs_alert_threshold=0.05)
    stepped = StepTaskProjection(config, tokenizer).project(rollouts, request)
    for batch in (combined, stepped):
        for metrics in (batch["rollout_metrics"], evaluation_response_metrics(batch)):
            assert {key: metrics[f"timing/grading/{key}"] for key in ("mean", "p50", "p95", "max")} == pytest.approx(
                {"mean": 27.5, "p50": 4.0, "p95": 86.6, "max": 101.0}
            )
            assert metrics["timing/setup/mean"] == 26.5
            assert metrics["timing/setup/p50"] == 3.0
            assert metrics["timing/setup/p95"] == pytest.approx(85.6)
            assert not any(key.startswith("environment/timing/") for key in metrics)
            assert not {"timing/grading", "timing/setup"} & metrics.keys()
    first_turns = filter_trajectory_batch(
        stepped, [index for index, last in enumerate(stepped["is_last_step"]) if not last]
    )
    assert not any(key.startswith(("timing/grading/", "timing/setup/")) for key in first_turns["rollout_metrics"])

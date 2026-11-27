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
from finestore.rl.run_records import read_evaluation_traces, read_evaluations
from finestore.rl.strata import TaskEvaluationIdentity, TaskStrata
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer
from rolloutengine.models import ModelTurn
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import ExitCodeReward, GraderMachine, ScriptGrader

from skyrl_train.batch_sampling import filter_trajectory_batch
from skyrl_train.config.utils import get_default_config
from skyrl_train.dataset.tasks import write_tasks
from skyrl_train.entrypoints.main_base import BasePPOExp
from skyrl_train.evaluate import evaluate
from skyrl_train.evaluation_pool import EvaluationPool
from skyrl_train.evaluation_profiles import resolve_evaluation_profiles
from skyrl_train.rollouts.task_projections import StepTaskProjection, WholeTaskProjection
from skyrl_train.rollouts.training_record import TrainingRollout, TrainingStep, TrainingTransition
from skyrl_train.trajectory_runners.trajectory_processing import (
    concatenate_trajectory_batches,
    get_metrics_from_trajectory_batch,
)
from skyrl_train.trajectory_runners.types import TaskRequestBatch, TrajectoryID
from skyrl_train.utils.trainer_utils import evaluation_response_metrics
from tests.cpu.task_specs import NATIVE_ENVIRONMENT, lowered_task, task_input, text_task, write_selection_panel
from tests.cpu.tiny_training.tiny_model import build_tiny_policy


def test_final_profiles_build_primary_and_additional_datasets_without_periodic_evaluation(tmp_path):
    config = get_default_config()
    primary, named = tmp_path / "primary" / "tasks.parquet", tmp_path / "named" / "tasks.parquet"
    task = text_task("shared").model_copy(update={"tags": ("agentic",)})
    digest = canonical_sha256(task.model_dump(mode="json"))
    for path in (primary, named):
        path.parent.mkdir()
        if path == primary:
            pq.write_table(
                pa.table(
                    {
                        "task_spec": [task.model_dump_json()],
                        "family": ["primary"],
                        "aai_category": ["general"],
                        "source": ["declared-source"],
                        "topic": ["algebra"],
                        "difficulty": ["hard"],
                        "agentic": pa.array([None], type=pa.bool_()),
                    }
                ),
                path,
            )
        else:
            write_tasks(path, [lowered_task(task)])
        write_selection_panel(
            path.parent, [task], path.parent.name, "1",
            lambda _: {"family": path.parent.name, "source": "declared-source",
                       "aai_category": "general", "topic": "algebra", "difficulty": "hard", "agentic": True},
        )
    config.data.val_data = [str(primary)]
    manifest_path = primary.parent / "summary.json"
    faults = [(manifest_path, b'{"selection_strata": null}', "SelectionDefinition")]
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
    experiment = BasePPOExp.__new__(BasePPOExp)
    experiment.cfg = config
    original_config = OmegaConf.to_container(config, resolve=True)
    experiment.evaluation_profiles = resolve_evaluation_profiles(config)
    experiment.tokenizer = AutoTokenizer.from_pretrained(build_tiny_policy(tmp_path / "model"))

    primary_dataset = experiment.get_eval_dataset()
    additional_datasets = experiment.get_additional_eval_datasets()

    assert primary_dataset is not None
    assert [row["input"].spec.task.id for row in primary_dataset.collate_fn(list(primary_dataset))] == ["shared"]
    additional_dataset = additional_datasets["named_final"]
    assert [row["input"].spec.task.id for row in additional_dataset.collate_fn(list(additional_dataset))] == ["shared"]

    identities = [
        dataset.collate_fn(list(dataset))[0]["input"].metadata for dataset in (primary_dataset, additional_dataset)
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
    assert all(item.subset_id == "declared-source" for item in identities)
    assert all(item.evaluation_identity.strata.agentic for item in identities)
    assert all(item.evaluation_identity.task_digest == digest for item in identities)
    assert OmegaConf.to_container(config, resolve=True) == original_config
    config.data.val_data = [str(primary), str(named)]
    with pytest.raises(ValueError, match="conflicting selection identities"):
        experiment.get_eval_dataset()
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
    mismatched = original.set_column(original.schema.get_field_index("family"), "family", pa.array(["wrong-family"]))
    false_label = original.set_column(original.schema.get_field_index("agentic"), "agentic", pa.array([False]))
    membership_path = primary.parent / "validation.parquet"
    membership = pq.read_table(membership_path)
    changed_task = json.loads(membership["task_spec"][0].as_py())
    changed_task["id"] = "edited"
    faults.extend(
        [
            (primary, mismatched, "row strata differ from selection manifest"),
            (primary, false_label, "row strata differ from selection manifest"),
            (membership_path, pa.table({"task_spec": [json.dumps(changed_task)]}), "manifest population differs"),
        ]
    )
    config.data.val_data = [str(primary)]
    for path, replacement, expected in faults:
        original = path.read_bytes()
        if isinstance(replacement, bytes):
            path.write_bytes(replacement)
        else:
            pq.write_table(replacement, path)
        try:
            with pytest.raises(ValueError, match=expected):
                experiment.get_eval_dataset()
        finally:
            path.write_bytes(original)


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("keep_traces", "redact_fields"),
    [(False, []), (True, []), (True, ["response.messages", "verification_result.reason"])],
    ids=["scores-only", "full-traces", "redacted-traces"],
)
async def test_evaluation_collector_keeps_task_id_and_scoring_metadata(tmp_path, keep_traces, redact_fields):
    config = get_default_config()
    config.trainer.export_path = str(tmp_path)
    config.trainer.dump_eval_results = keep_traces
    config.trainer.dump_grader_report = True
    config.generator.trajectory_retention.redact_fields = redact_fields
    config.generator.eval_n_samples_per_prompt = 1
    OmegaConf.update(config, "runtime", {"attempt_id": "eval-attempt", "model": "test-policy"}, force_add=True)
    task = text_task("validation-task")
    row = task_input(lowered_task(task), metric_class="script", data_source="validation")
    identity = TaskEvaluationIdentity(strata=TaskStrata(family="math", source="fixture"))
    row = replace(
        row, metadata=replace(row.metadata, evaluation_identity=identity, task_id=task.id, subset_id="math")
    )
    loader = StatefulDataLoader([{"input": row}], collate_fn=list)
    tokenizer = SimpleNamespace(eos_token_id=2, decode=lambda _tokens: "answer")
    projection = WholeTaskProjection(config.generator, tokenizer)
    response = TrainingRollout(
        task_id="validation-task",
        messages=({"role": "assistant", "content": "answer"},),
        prompt_token_ids=(1,),
        response_token_ids=(2,),
        loss_mask=(1,),
        logprobs=(-0.1,),
        grade=GradeResult(Outcome.GRADED, 1.0),
        stop_reason="stop",
    )

    async def run(request):
        return projection.project([response], request)

    async with EvaluationPool(concurrency=1) as pool:
        await evaluate(loader, SimpleNamespace(run=run), config, 0, tokenizer, evaluation_pool=pool)
        named_loader = StatefulDataLoader([{"input": row}], collate_fn=list)
        await evaluate(
            named_loader, SimpleNamespace(run=run), config, 0, tokenizer, val_set_name="named", evaluation_pool=pool
        )

    [saved] = read_evaluations(str(tmp_path), attempt_id="eval-attempt", step=0, evaluation="eval/")
    assert saved.task_id == task.id
    assert saved.data_source == "validation"
    assert saved.strata == identity.strata
    assert saved.model == "test-policy"
    [answer] = saved.answers
    assert answer.response_token_count == 1
    traces = read_evaluation_traces(str(tmp_path), attempt_id="eval-attempt", step=0)
    if keep_traces:
        assert [trace.evaluation for trace in traces] == ["eval/", "eval/named/"]
        for trace in traces:
            assert trace.task_id == task.id
            if redact_fields:
                assert trace.messages_json is None
                assert trace.input_prompt is None and trace.output_response is None
                assert trace.prompt_token_ids is None and trace.response_token_ids is None
                assert trace.verification is None
            else:
                assert (trace.prompt_token_ids, trace.response_token_ids) == ([1], [2])
                assert trace.messages_json is not None
                assert json.loads(trace.messages_json) == [{"role": "assistant", "content": "answer"}]
                assert trace.verification is not None and trace.verification.score == 1.0
    else:
        assert not traces
    report = json.loads((tmp_path / "grader_report.json").read_text())
    assert set(report["evaluation"]) == {"0", "named/0"}
    assert report["evaluation"]["named/0"][0]["subset_id"] == "math"
    assert {
        key: [(row["task_id"], row["success_rate"]) for row in rows]
        for key, rows in report["evaluation_tasks"].items()
    } == {"0": [(task.id, 1.0)], "named/0": [(task.id, 1.0)]}


def test_task_timings_survive_projection_concatenation_and_row_selection():
    projection_args = get_default_config().generator, SimpleNamespace(eos_token_id=2)
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
            update={
                "grader": ScriptGrader(
                    argv=("true",), placement=GraderMachine(environment=NATIVE_ENVIRONMENT), reward=ExitCodeReward()
                )
            }
        )
        inputs.append(
            task_input(
                lowered_task(task, verifier_backend="local"),
                metric_class="taskcompendium" if index % 2 else "other",
            )
        )
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
                        TrainingTransition(done=last, reward=float(last)),
                        i,
                        (),
                    )
                    for i, last in enumerate((False,) * (turn_count - 1) + (True,))
                ),
                metrics={} if duration is None else {"timing/grading": duration, "timing/setup": duration - 1},
            )
        )
    ids = tuple(TrajectoryID(str(i), 0) for i in range(len(inputs)))
    request = TaskRequestBatch(rows=tuple(inputs), trajectory_ids=ids)
    whole = WholeTaskProjection(*projection_args)
    batches = [
        whole.project(rollouts[part], replace(request, rows=request.rows[part], trajectory_ids=ids[part]))
        for part in (slice(0, 1), slice(1, None))
    ]
    combined = concatenate_trajectory_batches(batches)
    combined = concatenate_trajectory_batches([combined])
    stepped = StepTaskProjection(*projection_args).project(rollouts, request)
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

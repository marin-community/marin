"""
uv run --isolated --group dev --extra cpu pytest tests/cpu/test_eval.py
"""

import json

import fsspec
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf
from skyrl_gym.verification import VerificationResult

from skyrl_train.io import io
from marinskyrl.resource_locator import join_resource_path
from skyrl_train.evaluate import _calculate_eval_metrics, evaluate, evaluate_step_wise
from tests.cpu.trajectory_runners.fixture_runner import FixtureRunner
from skyrl_train.trajectory_runners.types import TrajectoryBatch
from skyrl_train.trajectory_runners.trajectory_retention import (
    TrajectorySink,
    execute_publication,
    parse_trajectory_retention_config,
)
from skyrl_train.trajectory_runners.trajectory_retention_publisher import InlineTrajectoryPublisher
from tests.cpu.util import example_dummy_config
from tests.cpu.task_specs import lowered_task, text_task, task_input
from skyrl_train.rollouts.group_grader import GroupGraderSpec


@pytest.fixture
def dummy_config():
    return example_dummy_config()


@pytest.mark.parametrize(
    "ungraded",
    [
        VerificationResult.error("verifier timed out", diagnostics={"exception_type": "VerifierTimeoutError"}),
        VerificationResult.unavailable("environment failed to start"),
        VerificationResult.skipped("grading disabled"),
    ],
)
def test_eval_verifier_metrics_exclude_ungraded_results_and_keep_policy_zero(ungraded):
    batch: TrajectoryBatch = {
        "response_ids": [[1], [2], [3]],
        "rewards": [1.0, 0.0, 0.0],
        "verification_results": [
            VerificationResult.verified(1.0, passed=True),
            VerificationResult.verified(0.0, passed=False, diagnostics={"exception_type": "AgentTimeoutError"}),
            ungraded,
        ],
    }

    metrics = _calculate_eval_metrics(batch, ["correct", "policy-failed", "ungraded"], ["bfcl"] * 3, 1)

    assert metrics["eval/all/avg_verifier_score"] == 0.5
    assert metrics["eval/all/verifier_score_coverage"] == pytest.approx(2 / 3)
    assert metrics["eval/all/num_attempted"] == 3
    assert metrics["eval/all/num_scored"] == 2
    assert metrics["eval/all/avg_score"] == pytest.approx(1 / 3)
    assert batch["rewards"] == [1.0, 0.0, 0.0]


def configure_eval(cfg, tmp_path):
    cfg.generator.backend = "vllm"
    cfg.generator.eval_sampling_params = OmegaConf.create(
        {
            "max_generate_length": 20,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "logprobs": None,
            "stop": None,
        }
    )
    cfg.generator.eval_n_samples_per_prompt = 1
    cfg.environment = OmegaConf.create({"env_class": "gsm8k"})
    cfg.trainer.dump_eval_results = False
    cfg.trainer.export_path = str(tmp_path)
    return cfg


class DummyStatefulDataLoader:
    def __init__(self, batches):
        self._batches = batches

    def __len__(self):
        return len(self._batches)

    def __iter__(self):
        return iter(self._batches)


class DummyRunner(FixtureRunner):
    def __init__(self, outputs: dict[str, TrajectoryBatch]):
        self.outputs = outputs
        self.seen_inputs = []

    async def _run(self, input_batch):
        self.seen_inputs.append(input_batch)
        return {**self.outputs[input_batch.metadata[0].uid], "trajectory_ids": list(input_batch.trajectory_ids)}


def test_eval_reports_normalized_verifier_score_alongside_raw_reward():
    batch: TrajectoryBatch = {
        "response_ids": [[1], [2]],
        "rewards": [5.0, 0.0],
        "verification_results": [
            VerificationResult.verified(5.0, score_min=1.0, score_max=5.0),
            VerificationResult.verified(0.0),
        ],
    }

    metrics = _calculate_eval_metrics(batch, ["a", "b"], ["genrm", "math"], 1)

    assert metrics["eval/all/avg_score"] == 2.5
    assert metrics["eval/all/avg_verifier_score"] == 0.5
    assert metrics["eval/all/verifier_score_coverage"] == 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize(("step_wise", "cloud_output"), [(False, False), (True, False), (False, True)])
async def test_evaluate_exports_public_results_and_expected_metrics(
    dummy_config, tmp_path, monkeypatch, step_wise, cloud_output
):
    cfg = configure_eval(dummy_config, tmp_path)
    cfg.trainer.dump_eval_results = True
    if cloud_output:
        cfg.trainer.export_path = f"s3://eval-results/{tmp_path.name}"
        filesystem = fsspec.filesystem("memory")
        local_filesystem = io._get_filesystem
        open_file = io.open_file
        monkeypatch.setattr(
            io, "_get_filesystem", lambda path: filesystem if io.is_cloud_path(path) else local_filesystem(path)
        )

        def open_local_output(path, mode="rb"):
            if io.is_cloud_path(path) and "w" in mode:
                raise OSError("This object store supports directory publication")
            return open_file(path, mode)

        monkeypatch.setattr(io, "open_file", open_local_output)

    prompts_batch = [
        {
            "input": task_input(
                lowered_task(text_task("uid-1", prompt="question-1", source="dataset/a")),
                metric_class="gsm8k",
                group_grader=GroupGraderSpec(
                    name="private-reference", parameters_json='{"api_key":"private-judge-key"}'
                ),
            ),
            "uid": "uid-1",
        },
        {
            "input": task_input(
                lowered_task(text_task("uid-2", prompt="question-2", source="dataset/b")), metric_class="custom_env"
            ),
            "uid": "uid-2",
        },
    ]
    eval_dataloader = DummyStatefulDataLoader([prompts_batch])

    trajectory_batch: TrajectoryBatch = {
        "prompt_token_ids": [[101], [102]],
        "response_ids": [[201], [202]],
        "rewards": [1.0, 0.0],
        "loss_masks": [[1], [1]],
        "stop_reasons": ["stop", "stop"],
        "rollout_logprobs": None,
        "env_classes": ["gsm8k", "custom_env"],
        "env_metrics": [{"truncated": 1}, {"truncated": 0}],
        "verification_results": [
            VerificationResult.verified(1.0, passed=True, diagnostics={"reference": "private-verifier-reference"}),
            VerificationResult.unavailable(
                "Request timed out.",
                diagnostics={
                    "api_key": "private-key",
                    "cause_error_type": "DaytonaRateLimitError",
                    "operation": "grade",
                    "machine_operation": "run",
                    "provider_status_code": 429,
                    "provider_sdk_call": "daytona_api_client_async.api.sandbox_api.create_sandbox",
                    "provider_rate_limits": {"retry-after-sandbox-create": 5},
                    "provider_error_code": "private-error-code",
                },
            ),
        ],
        "is_last_step": [True, True],
    }
    runner = DummyRunner(
        {
            row["uid"]: {
                key: [value[index]] if isinstance(value, list) else value for key, value in trajectory_batch.items()
            }
            for index, row in enumerate(prompts_batch)
        }
    )

    tokenizer = MagicMock()
    tokenizer.decode.side_effect = lambda tokens: "decoded"
    # The run's shared sink is a Ray actor; an in-process sink with the same configuration keeps this test off Ray.
    sink = TrajectorySink(
        parse_trajectory_retention_config(cfg.generator.trajectory_retention),
        tokenizer,
        publisher=InlineTrajectoryPublisher(execute_publication),
    )

    evaluation = evaluate_step_wise if step_wise else evaluate
    metrics = await evaluation(
        eval_dataloader=eval_dataloader,
        trajectory_runner=runner,
        cfg=cfg,
        global_step=5,
        tokenizer=tokenizer,
        trajectory_sink=sink,
    )

    expected_metrics = {
        "eval/dataset_a/avg_score": 1.0,
        "eval/dataset_a/pass_at_1": 1.0,
        "eval/dataset_b/avg_score": 0.0,
        "eval/dataset_b/pass_at_1": 0.0,
        "eval/all/avg_score": 0.5,
        "eval/all/pass_at_1": 0.5,
        "eval/all/environment/gsm8k/truncated": 1.0,
        "eval/all/environment/custom_env/truncated": 0.0,
        "eval/dataset_a/environment/gsm8k/truncated": 1.0,
        "eval/dataset_b/environment/custom_env/truncated": 0.0,
    }

    for key, expected_value in expected_metrics.items():
        assert metrics[key] == pytest.approx(expected_value)

    dump = tmp_path / "dumped_evals/global_step_5_evals"
    if cloud_output:
        filesystem.get(join_resource_path(cfg.trainer.export_path, "dumped_evals/global_step_5_evals"), str(dump), recursive=True)
    saved = "".join(path.read_text() for path in dump.glob("*.jsonl"))
    assert saved
    assert "private-reference" not in saved
    assert "private-judge-key" not in saved
    assert "private-verifier-reference" not in saved
    assert "private-key" not in saved
    assert "private-error-code" not in saved
    records = [json.loads((dump / f"dataset_{name}.jsonl").read_text()) for name in ("a", "b")]
    assert records[0]["input_prompt"] == records[1]["input_prompt"]
    assert records[0]["uid"] != records[1]["uid"]
    assert [record["prompt_token_ids"] for record in records] == [[101], [102]]
    assert [record["response_token_ids"] for record in records] == [[201], [202]]
    assert [record["env_metrics"]["truncated"] for record in records] == [1, 0]
    assert records[0]["verification"] == {
        "status": "verified",
        "reason": None,
        "score": 1.0,
        "passed": True,
        "score_min": 0.0,
        "score_max": 1.0,
        "diagnostics": {},
    }
    assert records[1]["verification"]["reason"] == "Request timed out."
    assert records[1]["verification"]["status"] == "unavailable"
    assert records[1]["verification"]["score"] is None
    assert records[1]["verification"]["diagnostics"] == {
        "cause_error_type": "DaytonaRateLimitError",
        "operation": "grade",
        "machine_operation": "run",
        "provider_status_code": 429,
        "provider_sdk_call": "daytona_api_client_async.api.sandbox_api.create_sandbox",
        "provider_rate_limits": {"retry-after-sandbox-create": 5},
    }

    assert [prompt[0]["content"] for batch in runner.seen_inputs for prompt in batch.public_prompts] == [
        "question-1",
        "question-2",
    ]
    assert [row.metric_class for batch in runner.seen_inputs for row in batch.metadata] == ["gsm8k", "custom_env"]
    assert tuple(row for batch in runner.seen_inputs for row in batch.rows) == tuple(
        prompt["input"] for prompt in prompts_batch
    )
    assert all(batch.batch_metadata.training_phase == "eval" for batch in runner.seen_inputs)

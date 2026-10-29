import asyncio
import json
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from multiprocessing.context import ForkServerContext

import fsspec
import pytest
import skyrl_gym
import skyrl_train
import torch
import zstandard
from omegaconf import OmegaConf
from taskcompendium.grading_result import Outcome
from skyrl_train.evaluate import evaluation_dump_dir
from skyrl_train.dataset.tasks import source_row_task, write_tasks
from examples.cat_count.cpu_canary import PROMPT
from skyrl_train.metric_names import CORRECTION_WEIGHT_MEAN_METRIC
from skyrl_train.config.utils import get_default_config
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.trajectory_runners.model_clients import DirectModelClient

from tests.cpu.cat_count_policy import POLICY_PREFIX, download_policy
from tests.cpu.tiny_training.cat_count import (
    FAST_STEPS,
    RUN_TIMEOUT_SECONDS,
    cat_count_config,
    run_cat_count,
    run_cat_count_generate,
)
from tests.cpu.tiny_training.cpu_backend import CPUInferenceEngine
from tests.cpu.tiny_training.experiment import read_metrics

pytestmark = pytest.mark.slow


def test_policy_download_rejects_a_corrupted_manifest(tmp_path, monkeypatch):
    filesystem = fsspec.filesystem("memory")
    filesystem.pipe(f"{POLICY_PREFIX}/MANIFEST.json", b'{"files": {}}')
    monkeypatch.setattr("tests.cpu.cat_count_policy.create_s3_filesystem", lambda **_kwargs: filesystem)
    with pytest.raises(ValueError, match="manifest SHA256"):
        download_policy(tmp_path)


@pytest.fixture
def cat_count_session(monkeypatch):
    repository = Path(__file__).resolve().parents[3]
    assert Path(skyrl_train.__file__).resolve().is_relative_to(repository / "skyrl-train")
    assert Path(skyrl_gym.__file__).resolve().is_relative_to(repository / "skyrl-gym")
    rows = []

    class Sink(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = self.rfile.read(int(self.headers["Content-Length"]))
            if self.headers.get("Content-Encoding") == "zstd":
                payload = zstandard.ZstdDecompressor().decompress(payload, max_output_size=1 << 28)
            batch = json.loads(payload)
            rows.extend(batch["records"])
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"batch_id": batch["batch_id"], "status": "accepted"}).encode())

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Sink)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    telemetry_env = {
        "SKYRL_TELEMETRY_ENDPOINT": f"http://127.0.0.1:{server.server_port}/v1/ingest",
        "SKYRL_RUN_ID": "cat-count-cpu",
        "SKYRL_EXECUTION_UID": "cat-count-cpu-attempt",
    }
    for key, value in telemetry_env.items():
        monkeypatch.setenv(key, value)
    try:
        yield rows
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def train(
    forkserver_context: ForkServerContext,
    root: Path,
    model: Path,
    *,
    flipped: bool = False,
    steps: int = FAST_STEPS,
    staleness: int = 0,
    resume: bool = False,
    eval_interval: int | None = None,
    seed: int = 0,
    callbacks: list[dict] | None = None,
):
    cfg = cat_count_config(
        root, model, steps=steps, staleness=staleness, resume=resume, eval_interval=eval_interval, seed=seed
    )
    if callbacks is not None:
        OmegaConf.update(cfg, "trainer.callbacks", callbacks, force_add=True)
    if flipped:
        cfg.trainer.algorithm.advantage_estimator = "cat_count_flipped_grpo"
    return run_process(forkserver_context, root, cfg)


def run_process(forkserver_context, root, cfg, runner=run_cat_count):
    worker_env = {key: os.environ[key] for key in ("SKYRL_TELEMETRY_ENDPOINT", "SKYRL_RUN_ID", "SKYRL_EXECUTION_UID")}
    start = time.perf_counter()
    process = forkserver_context.Process(target=runner, args=(cfg, worker_env))
    process.start()
    process.join(RUN_TIMEOUT_SECONDS)
    if process.exitcode is None:
        process.kill()
        process.join()
        pytest.fail(f"the run did not finish within {RUN_TIMEOUT_SECONDS} seconds")
    assert process.exitcode == 0
    print(f"CAT_COUNT_CPU run={root.name} seconds={time.perf_counter() - start:.3f}")
    return read_metrics(root)


def scores(records):
    evaluations = [row for row in records if "eval/train/avg_score" in row]
    return [(row["eval/train/avg_score"], row["eval/heldout/avg_score"]) for row in evaluations]


@pytest.mark.asyncio
async def test_cpu_sampling_preserves_trajectory_rng_and_minimum_tokens(cat_count_policy):
    engine = CPUInferenceEngine(str(cat_count_policy), seed=0)
    tokenizer = engine.tokenizer
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": PROMPT.format(N=7),
            }
        ],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
    )
    sampling = {"temperature": 1.0, "max_tokens": 64, "min_tokens": 0, "logprobs": 0}
    try:
        batch = await engine.generate(
            {"prompt_token_ids": [prompt] * 3, "session_ids": ["a", "b", "c"], "sampling_params": sampling}
        )
        reordered = await asyncio.gather(
            *(
                engine.generate({"prompt_token_ids": [prompt], "session_ids": [identity], "sampling_params": sampling})
                for identity in ("c", "b", "a")
            )
        )
        for index, result in enumerate(reversed(reordered)):
            assert result["response_ids"][0] == batch["response_ids"][index]
            assert result["response_logprobs"][0] == pytest.approx(batch["response_logprobs"][index], abs=1e-5)
        assert len({tuple(ids) for ids in batch["response_ids"]}) > 1
        for index, identity in enumerate(("a", "b", "c")):
            singleton = await engine.generate(
                {"prompt_token_ids": [prompt], "session_ids": [identity], "sampling_params": sampling}
            )
            assert singleton["response_ids"][0] == batch["response_ids"][index]

        config = get_default_config()
        config.generator.enable_http_endpoint = False
        client = DirectModelClient(InferenceEngineClient([engine], tokenizer, config))
        for index in (2, 0, 1):
            chat = await client.generate(
                {
                    "prompts": [[{"role": "user", "content": PROMPT.format(N=7)}]],
                    "chat_completion_params": [{}],
                    "session_ids": [("a", "b", "c")[index]],
                    "sampling_params": sampling,
                }
            )
            assert chat["response_ids"][0] == batch["response_ids"][index]
            assert chat["response_logprobs"][0] == pytest.approx(batch["response_logprobs"][index], abs=1e-5)

        eos_prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": PROMPT.format(N=1),
                }
            ],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
        ) + tokenizer.encode("cat", add_special_tokens=False)
        mixed = await engine.generate(
            {"prompt_token_ids": [eos_prompt, prompt], "session_ids": ["eos-short", "a"], "sampling_params": sampling}
        )
        assert mixed["response_ids"][0] == [tokenizer.eos_token_id]
        assert mixed["response_ids"][1] == batch["response_ids"][0]
        for minimum in (0, None, 3):
            params = {"temperature": 0.0, "max_tokens": 8, "logprobs": 0}
            if minimum is not None:
                params["min_tokens"] = minimum
            result = await engine.generate(
                {"prompt_token_ids": [eos_prompt], "session_ids": ["eos"], "sampling_params": params}
            )
            ids = result["response_ids"][0]
            if minimum == 0:
                assert ids == [tokenizer.eos_token_id]
            else:
                assert len(ids) >= (minimum or 1)
                assert tokenizer.eos_token_id not in ids[: minimum or 1]
            assert all(math.isfinite(value) for value in result["response_logprobs"][0])
    finally:
        await engine.teardown()


def assert_cpu_learning(positive, root: Path, telemetry):
    before, after = scores(positive)
    assert sum(after) / 2 >= sum(before) / 2 + 0.1
    assert all(final > initial for initial, final in zip(before, after, strict=True))
    training = [row for row in positive if "policy/raw_grad_norm" in row]
    assert len(training) == FAST_STEPS
    assert all(math.isfinite(row["policy/mismatch/pooled/log_ratio_abs_mean"]) for row in training)
    # The engine stamps every trained token with the step that sampled it.
    assert all(row["async/staleness_tokens_stamped"] == 1 for row in training)
    assert all(math.isfinite(row["policy/policy_loss"]) for row in training)
    assert all(0 < row[CORRECTION_WEIGHT_MEAN_METRIC] <= 2 for row in training)
    assert any(row["policy/ppo_clip_ratio"] > 0 for row in training)
    assert all("environment/exact" in row for row in training)
    assert any("environment/exact_n20" in row for row in training)
    assert all(sum(row[f"generate/outcome/{outcome.value}"] for outcome in Outcome) == 32 for row in training)
    checkpoint = torch.load(root / f"ckpts/global_step_{FAST_STEPS}/policy/rank_0.pt", weights_only=False)
    optimizer_steps = [state["step"].item() for state in checkpoint["optimizer"]["state"].values()]
    assert optimizer_steps and set(optimizer_steps) == {2 * FAST_STEPS}
    names = {row["name"] for row in telemetry}
    assert {"policy_step", "work_completed", "phase_duration_seconds", "weight_sync_completed"} <= names
    assert any(
        row["name"] == "training_metric_value" and row["attributes"].get("metric") == "environment/exact"
        for row in telemetry
    )

    evaluations = [row for row in positive if "eval/train/avg_score" in row]
    assert [int(row["step"]) for row in evaluations] == [0, FAST_STEPS]
    for row in evaluations:
        assert row["eval/reporting/avg_score"] == pytest.approx(
            (row["eval/train/avg_score"] + row["eval/heldout/avg_score"]) / 2
        )
    assert evaluations[0]["eval/reporting/avg_score_improvement"] == 0
    assert evaluations[-1]["eval/reporting/avg_score_improvement"] >= 0.1
    greedy_rows = [
        json.loads(line)
        for line in (Path(evaluation_dump_dir(str(root / "exports"), 0)) / "train.jsonl").read_text().splitlines()
    ]
    sampled_rows = [
        json.loads(line)
        for line in (Path(evaluation_dump_dir(str(root / "exports/sampled"), 0)) / "train.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(sampled_rows) == 8 * len(greedy_rows)
    for prefix, rows in (("eval/train", greedy_rows), ("eval/sampled/train", sampled_rows)):
        exact = sum((sum(row["score"]) if isinstance(row["score"], list) else row["score"]) == 1.0 for row in rows)
        assert evaluations[0][f"{prefix}/environment/cat_count/exact"] == pytest.approx(exact / len(rows))
    assert any(
        row["name"] == "training_metric_value"
        and row["attributes"].get("metric") == "eval/sampled/train/environment/cat_count/exact"
        for row in telemetry
    )


def train_positive(forkserver_context, root, model, seed=0):
    return train(
        forkserver_context,
        root,
        model,
        seed=seed,
        steps=FAST_STEPS + 2,
        eval_interval=FAST_STEPS,
        callbacks=[
            {"type": "checkpoint", "save_steps": FAST_STEPS},
            {
                "type": "evaluation",
                "eval_steps": FAST_STEPS,
                "additional_evaluations": {
                    "sampled": {"sampling_params": {"temperature": 1.0, "seed": 42}, "n_samples_per_prompt": 8}
                },
                "metric_groups": {"eval/reporting/avg_score": ["eval/train/avg_score", "eval/heldout/avg_score"]},
                "stop_when": {"eval/reporting/avg_score": {"min_improvement": 0.1}},
            },
        ],
    )


def test_cat_count_cpu_learns(tmp_path, cat_count_policy, cat_count_session, forkserver_context):
    root = tmp_path / "positive"
    positive = train_positive(forkserver_context, root, cat_count_policy)
    assert_cpu_learning(positive, root, cat_count_session)
    before, after = scores(positive)
    print(f"CAT_COUNT_CPU seed=0 positive_scores={before}->{after}")


@pytest.mark.timeout(2 * RUN_TIMEOUT_SECONDS)
# Step-wise training has distinct estimator forwarding and evaluation projection branches.
@pytest.mark.parametrize("step_wise", [False, True])
def test_cat_count_rloo_n_preserves_metrics_across_resume(
    tmp_path, cat_count_policy, cat_count_session, forkserver_context, step_wise
):
    root = tmp_path / "rloo-n"
    for steps, resume in ((2, False), (4, True)):
        cfg = cat_count_config(root, cat_count_policy, steps=steps, resume=resume, eval_interval=3)
        cfg.trainer.algorithm.advantage_estimator = "rloo_n"
        cfg.trainer.step_wise_training = step_wise
        cfg.trainer.algorithm.group_advantage_min_size = 2
        cfg.trainer.train_batch_size = 5
        cfg.trainer.policy_mini_batch_size = 5
        cfg.trainer.placement.policy_num_gpus_per_node = 3
        cfg.trainer.micro_train_batch_size_per_gpu = 1
        cfg.trainer.micro_forward_batch_size_per_gpu = 1
        cfg.generator.n_samples_per_prompt = 2
        for filename in (cfg.data.train_data[0], cfg.data.val_data[0]):
            path = Path(filename)
            row = json.loads(path.read_text().splitlines()[0])
            row["prompt"][0]["content"] = "cat " * 200
            with path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
        records = run_process(forkserver_context, root, cfg)
        training = [row for row in records if "policy/raw_grad_norm" in row]
        assert [row["step"] for row in training] == list(range(1, steps + 1))
        assert all(row["generate/prompts_dropped_too_long"] == 1 for row in training)
        assert all(
            row["reward/zero_std_group_fraction"] == row["reward/rloo_n_filtered_groups"] / 5 for row in training
        )
        assert all(row["reward/rloo_n_total_groups"] == 5 for row in training)
        assert any(row["reward/rloo_n_filtered_groups"] > 0 for row in training)
        assert sum(records[-1][f"eval/all/outcome/{outcome.value}"] for outcome in Outcome) == 20


@pytest.mark.timeout(RUN_TIMEOUT_SECONDS)
def test_cat_count_generate_preserves_task_ids_and_token_metrics(
    tmp_path, cat_count_policy, cat_count_session, forkserver_context
):
    root = tmp_path / "generate"
    cfg = cat_count_config(root, cat_count_policy)
    environment_configs = OmegaConf.to_container(cfg.environment.task_sessions, resolve=True)

    def task(n, identifier, extra_info):
        row = {
            "prompt": [{"role": "user", "content": PROMPT.format(N=n)}],
            "env_class": "cat_count",
            "data_source": "train",
            "extra_info": {"n": n, "data_source": "train", **extra_info},
        }
        lowered = source_row_task(row, 0, source_name="cat-count", environment_configs=environment_configs)
        spec = lowered.task.model_copy(update={"id": identifier})
        return lowered.model_copy(update={"task": spec})

    expected = {f"stable-cat-count-{n}": PROMPT.format(N=n) for n in (7, 1, 4, 2)}
    tasks = [task(n, f"stable-cat-count-{n}", {}) for n in (7, 1, 4, 2)]
    tasks.append(task(2, "rollout-error", {"n": "invalid"}))
    expected["rollout-error"] = PROMPT.format(N=2)
    path = root / "data" / "permuted.parquet"
    overlong = task(2, "overlong", {})
    context = overlong.task.context.model_copy(
        update={"events": (overlong.task.context.events[0].model_copy(update={"content": "cat " * 200}),)}
    )
    spec = overlong.task.model_copy(update={"context": context})
    write_tasks(path, [*reversed(tasks), overlong.model_copy(update={"task": spec})])
    cfg.data.val_data = [str(path)]
    cfg.generator.num_inference_engines = 2
    cfg.generator.eval_n_samples_per_prompt = 2
    cfg.generator.eval_sampling_params.ignore_eos = True
    cfg.generator.eval_sampling_params.max_generate_length = 8
    records = run_process(forkserver_context, root, cfg, run_cat_count_generate)
    assert len(records) == 1
    metrics = records[0]
    rows = [
        json.loads(line)
        for line in (Path(evaluation_dump_dir(str(root / "exports"), None)) / "train.jsonl").read_text().splitlines()
    ]
    assert metrics["generate/prompts_dropped_too_long"] == 1
    assert len(rows) == 2 * len(expected)
    assert {row["task_id"] for row in rows} == expected.keys()
    assert all(expected[row["task_id"]] in row["input_prompt"] and row["uid"] != row["task_id"] for row in rows)
    assert all(len(row["response_token_ids"]) == 8 for row in rows if not row["task_id"].startswith("rollout-error"))
    tokens = sum(len(row["response_token_ids"]) for row in rows)
    assert metrics["generate/response_tokens"] == tokens
    assert metrics["generate/wall_seconds"] > 0
    generated_tokens = 2 * len(expected) * 8
    assert metrics["generate/vllm/generation_tokens"] == generated_tokens
    assert metrics["generate/tokens_per_second_per_gpu"] == pytest.approx(
        generated_tokens / metrics["generate/wall_seconds"] / 2
    )
    assert metrics["generate/vllm/num_engines"] == 2
    assert {outcome.value: metrics[f"generate/outcome/{outcome.value}"] for outcome in Outcome} == {
        "graded": 8,
        "submission_failure": 0,
        "invalid_task": 0,
        "infra_error": 0,
        "unavailable": 2,
    }
    assert metrics["generate/outcome/rollout_failure"] == 2


@pytest.mark.nightly
@pytest.mark.parametrize("seed", [0, 1])
def test_cat_count_flipped_advantage_fails(tmp_path, cat_count_policy, cat_count_session, forkserver_context, seed):
    root = tmp_path / "positive"
    positive = train_positive(forkserver_context, root, cat_count_policy, seed=seed)
    negative = train(forkserver_context, tmp_path / "negative", cat_count_policy, flipped=True, seed=seed)
    assert_cpu_learning(positive, root, cat_count_session)
    before, after = scores(positive)
    negative_before, negative_after = scores(negative)
    assert before == negative_before
    assert sum(negative_after) <= sum(negative_before)
    assert (sum(after) - sum(negative_after)) / 2 >= 0.4
    print(
        f"CAT_COUNT_CPU seed={seed} paired_scores positive={before}->{after} negative={negative_before}->{negative_after}"
    )


@pytest.mark.nightly
def test_cat_count_async_resumes_and_converges(tmp_path, cat_count_policy, cat_count_session, forkserver_context):
    root = tmp_path / "async"
    first = train(forkserver_context, root, cat_count_policy, steps=4, staleness=1)
    before_resume = len(cat_count_session)
    resumed = train(forkserver_context, root, cat_count_policy, steps=100, staleness=1, resume=True, eval_interval=2)
    resumed_steps = {
        int(row["attributes"]["step"])
        for row in cat_count_session[before_resume:]
        if row["name"] == "work_completed" and row["attributes"].get("work_kind") == "policy_step"
    }
    assert resumed_steps == set(range(5, 101))
    assert scores(resumed)[: len(scores(first))] == scores(first)
    assert len([row for row in first if "policy/raw_grad_norm" in row]) == 4
    training = [row for row in resumed if "policy/raw_grad_norm" in row]
    assert [row["trainer/global_step"] for row in training] == list(range(1, 101))
    assert any(row["async/staleness_mean"] > 0 for row in training)
    # Responses in flight across a weight sync carry both steps.
    assert any(row["async/staleness_tokens_spanning"] > 0 for row in training)
    initial_mean = sum(scores(resumed)[0]) / 2
    assert any(sum(pair) / 2 >= max(0.65, initial_mean + 0.3) for pair in scores(resumed))
    checkpoint = torch.load(root / "ckpts/global_step_100/policy/rank_0.pt", weights_only=False)
    assert {state["step"].item() for state in checkpoint["optimizer"]["state"].values()} == {200}

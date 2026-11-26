import asyncio
from collections import Counter
import json
import math
import os
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from multiprocessing.context import ForkServerContext

from finestore.rl.run_records import read_attempts, read_evaluation_traces, read_evaluations
from marinskyrl.environment_contract import RUN_MODEL_ENV
import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import rolloutengine
import skyrl_gym
import skyrl_train
import torch
import zstandard
from omegaconf import OmegaConf
from taskcompendium.grading_result import Outcome
from taskcompendium.grader import verifyit_package
from verifyit.spec import ExactSpec
from skyrl_train.dataset.tasks import TASK_SPEC_COLUMN, source_row_task, write_tasks
from examples.cat_count.cpu_canary import PROMPT
from skyrl_gym.envs.cat_count.reward import TARGET_WORD
from skyrl_train.metric_names import CORRECTION_WEIGHT_MEAN_METRIC, TASK_TIMING_KEYS
from skyrl_train.config.utils import get_default_config
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.trajectory_runners.model_clients import DirectModelClient

from tests.cpu.cat_count_policy import POLICY_PREFIX, download_policy
from tests.cpu.task_specs import lowered_task, text_task, write_selection_panel
from tests.cpu.tiny_training.cat_count import (
    FAST_STEPS,
    RUN_TIMEOUT_SECONDS,
    cat_count_config,
    run_cat_count,
    run_cat_count_generate,
)
from tests.cpu.tiny_training.cpu_backend import CPUInferenceEngine
from tests.cpu.tiny_training.tiny_model import write_gsm8k_dataset
from tests.cpu.tiny_training.experiment import TinyGenerateExp, read_metrics

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
    assert Path(rolloutengine.__file__).resolve().is_relative_to(repository.parents[1] / "lib/rolloutengine/src")
    rows = []

    class Sink(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = self.rfile.read(int(self.headers["Content-Length"]))
            if self.headers.get("Content-Encoding") == "zstd":
                payload = zstandard.ZstdDecompressor().decompress(payload, max_output_size=1 << 28)
            batch = json.loads(payload)
            rows.extend({**record, "resource": batch["resource"]} for record in batch["records"])
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
        RUN_MODEL_ENV: "cat-count-model",
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
    groups_per_family: dict[str, int] | None = None,
):
    cfg = cat_count_config(
        root, model, steps=steps, staleness=staleness, resume=resume, eval_interval=eval_interval, seed=seed
    )
    if callbacks is not None:
        OmegaConf.update(cfg, "trainer.callbacks", callbacks, force_add=True)
    if flipped:
        cfg.trainer.algorithm.advantage_estimator = "cat_count_flipped_grpo"
    if groups_per_family:
        source = Path(cfg.data.train_data[0])
        configs = OmegaConf.to_container(cfg.environment.task_sessions, resolve=True)
        rows = []
        for index, line in enumerate(source.read_text().splitlines()):
            row = json.loads(line)
            task = source_row_task(row, index, source_name="train", environment_configs=configs).task
            family = "odd" if row["extra_info"]["n"] % 2 else "even"
            rows.append({TASK_SPEC_COLUMN: task.model_dump_json(), "family": family})
        parquet = source.with_suffix(".parquet")
        pq.write_table(pa.Table.from_pylist(rows), parquet)
        cfg.data.train_data = [str(parquet)]
        cfg.environment.task_sessions.task_spec.harness_by_dataset = {"train": "cat_count"}
        OmegaConf.update(cfg, "trainer.rollout_buffer.groups_per_family", groups_per_family, force_add=True)
    return run_process(forkserver_context, root, cfg)


def run_process(forkserver_context, root, cfg, runner=run_cat_count):
    worker_env = {
        key: os.environ[key]
        for key in ("SKYRL_TELEMETRY_ENDPOINT", "SKYRL_RUN_ID", "SKYRL_EXECUTION_UID", RUN_MODEL_ENV)
    }
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
    return read_metrics(root) if "records" in cfg.trainer.logger else None


def scores(records):
    evaluations = [row for row in records if "eval/train/avg_score" in row]
    return [(row["eval/train/avg_score"], row["eval/heldout/avg_score"]) for row in evaluations]


@pytest.mark.asyncio
async def test_cpu_sampling_preserves_trajectory_rng_and_minimum_tokens(cat_count_policy):
    engine = CPUInferenceEngine(str(cat_count_policy), seed=0)
    tokenizer = engine.tokenizer

    def messages(n):
        return [{"role": "user", "content": PROMPT.format(N=n)}]

    def tokens(n):
        return tokenizer.apply_chat_template(
            messages(n), tokenize=True, return_dict=False, add_generation_prompt=True
        )

    prompt = tokens(7)
    identities = ("a", "b", "c")
    sampling = {"temperature": 1.0, "max_tokens": 64, "min_tokens": 0, "logprobs": 0}

    async def generate(prompts, sessions):
        return await engine.generate({"prompt_token_ids": prompts, "session_ids": sessions, "sampling_params": sampling})

    try:
        batch = await generate([prompt] * 3, list(identities))
        reordered = await asyncio.gather(*(generate([prompt], [identity]) for identity in reversed(identities)))
        for index, result in enumerate(reversed(reordered)):
            assert result["response_ids"][0] == batch["response_ids"][index]
            assert result["response_logprobs"][0] == pytest.approx(batch["response_logprobs"][index], abs=1e-5)
        assert len({tuple(ids) for ids in batch["response_ids"]}) > 1
        for index, identity in enumerate(identities):
            singleton = await generate([prompt], [identity])
            assert singleton["response_ids"][0] == batch["response_ids"][index]

        config = get_default_config()
        config.generator.enable_http_endpoint = False
        client = DirectModelClient(InferenceEngineClient([engine], tokenizer, config))
        for index in (2, 0, 1):
            chat = await client.generate(
                {
                    "prompts": [messages(7)],
                    "chat_completion_params": [{}],
                    "session_ids": [identities[index]],
                    "sampling_params": sampling,
                }
            )
            assert chat["response_ids"][0] == batch["response_ids"][index]
            assert chat["response_logprobs"][0] == pytest.approx(batch["response_logprobs"][index], abs=1e-5)

        eos_prompt = tokens(1) + tokenizer.encode("cat", add_special_tokens=False)
        mixed = await generate([eos_prompt, prompt], ["eos-short", "a"])
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
    attempt_id = next(reversed(read_attempts(str(root / "exports"))))
    eval_rows = read_evaluation_traces(str(root / "exports"), attempt_id=attempt_id, step=0)
    greedy_rows = [row for row in eval_rows if row.data_source == "train" and row.evaluation == "eval/"]
    sampled_rows = [row for row in eval_rows if row.data_source == "train" and row.evaluation == "eval/sampled/"]
    assert len(sampled_rows) == 8 * len(greedy_rows)
    for prefix, rows in (("eval/train", greedy_rows), ("eval/sampled/train", sampled_rows)):
        exact = sum(row.score == 1.0 for row in rows)
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
    preserved = None
    generated_attempt_id = None
    original_traces = None
    for steps, resume in ((2, False), (4, True)):
        interval = 3 if resume else 2
        cfg = cat_count_config(root, cat_count_policy, steps=steps, resume=resume, eval_interval=3)
        if resume:
            OmegaConf.update(cfg, "runtime", {"run_id": "cpu-rloo-n", "attempt_id": f"attempt-{steps}"}, force_add=True)
        cfg.trainer.algorithm.advantage_estimator = "rloo_n"
        cfg.trainer.step_wise_training = step_wise
        cfg.trainer.algorithm.group_advantage_min_size = 2
        cfg.trainer.train_batch_size = 5
        cfg.trainer.policy_mini_batch_size = 5
        cfg.trainer.placement.policy_num_gpus_per_node = 3
        cfg.trainer.micro_train_batch_size_per_gpu = 1
        cfg.trainer.micro_forward_batch_size_per_gpu = 1
        cfg.generator.n_samples_per_prompt = 2
        cfg.generator.inference_stats_interval = 1
        paths = []
        environment_configs = OmegaConf.to_container(cfg.environment.task_sessions, resolve=True)
        for filename in (cfg.data.train_data[0], *cfg.data.val_data):
            path = Path(filename)
            tasks = []
            for index, line in enumerate(path.read_text().splitlines()):
                row = json.loads(line)
                n = row["extra_info"]["n"]
                lowered = source_row_task(row, index, source_name=path.stem, environment_configs=environment_configs)
                if index % 5 == 0:
                    task = lowered.task.model_copy(
                        update={"grader": verifyit_package(ExactSpec(expected=(" ".join([TARGET_WORD] * n),))).grader}
                    )
                    lowered = lowered_task(task, max_turns=1)
                tasks.append(lowered)
            tasks.append(lowered_task(text_task("overlong", prompt="cat " * 200), max_turns=1))
            paths.append(str(path.with_suffix(".parquet")))
            write_tasks(Path(paths[-1]), tasks)
        cfg.data.train_data, cfg.data.val_data = paths[:1], paths[1:]
        math_file = write_gsm8k_dataset(root / "data/math.jsonl", 1, max_turns=2)
        math_row = json.loads(math_file.read_text())
        math_row["prompt"] = [{"role": "user", "content": PROMPT.format(N=2)}]
        full_task = source_row_task(math_row, 0, source_name="two-turn-math", environment_configs=environment_configs)
        full_task = full_task.model_copy(
            update={"task": full_task.task.model_copy(update={"id": "full-task_set-only"})}
        )
        full_task_set = root / "data/full-task_set.parquet"
        write_tasks(full_task_set, [full_task])
        cfg.generator.max_turns = 2
        cfg.trainer.callbacks = [
            {
                "type": "evaluation",
                "eval_steps": interval,
                "eval_on_train_end": False,
                "additional_evaluations": {
                    "full_task_set": {
                        "data_files": [str(full_task_set)],
                        "schedule": "final",
                        "n_samples_per_prompt": 1,
                        "sampling_params": {"temperature": 0.0, "seed": 42, "max_generate_length": 8},
                    }
                },
            },
            {"type": "checkpoint", "save_steps": steps},
        ]
        if resume:
            cfg.trainer.callbacks.append({"type": "inference_stats", "poll_interval_seconds": 0})
        records = run_process(forkserver_context, root, cfg, runner=run_cat_count)
        engine_rows = [
            row
            for row in cat_count_session
            if row["name"] == "num_requests_running" and row["attributes"].get("step") == str(steps)
        ]
        assert engine_rows
        if resume:
            assert len(engine_rows) == 1
        assert any(row["name"] == "generation_tokens_total" and row["value"] > 0 for row in cat_count_session)
        attempts = read_attempts(str(root / "exports"))
        assert all(record.evaluation_storage_version == 1 for record in attempts.values())
        if not resume:
            (generated_attempt_id,) = attempts
            saved = torch.load(root / "ckpts/global_step_2/trainer_state.pt", map_location="cpu", weights_only=False)
            assert saved["config"].runtime.attempt_id == generated_attempt_id
        else:
            continuation = attempts["attempt-4"]
            assert continuation.parent is not None and continuation.parent.attempt_id == generated_attempt_id
            assert continuation.parent.step == continuation.resumed_step == 2
        current_attempt = "attempt-4" if resume else generated_attempt_id
        elapsed = attempts[current_attempt].elapsed_excluding_eval_seconds
        assert elapsed is not None and elapsed > 0
        task_rows = read_evaluations(
            str(root / "exports"), attempt_id=current_attempt, step=steps, evaluation="eval/full_task_set/"
        )
        assert [row.task_id for row in task_rows] == ["full-task_set-only"]
        if resume:
            original_rows = read_evaluations(
                str(root / "exports"), attempt_id=generated_attempt_id, step=2, evaluation="eval/full_task_set/"
            )
            assert [row.task_id for row in original_rows] == ["full-task_set-only"]
            assert original_rows[0].attempt_id != task_rows[0].attempt_id
        final_metrics = [row for row in records if "eval/full_task_set/all/num_attempted" in row]
        expected_final = [2, 4] if resume else [2]
        assert [row["step"] for row in final_metrics] == expected_final
        assert all(row["eval/full_task_set/all/num_attempted"] == 1 for row in final_metrics)
        full_rows = read_evaluation_traces(
            str(root / "exports"), attempt_id=current_attempt, step=steps, evaluation="eval/full_task_set/"
        )
        assert {row.task_id for row in full_rows} == {"full-task_set-only"}
        finals = [row for row in full_rows if row.is_last_step]
        assert len(finals) == 1
        if step_wise:
            assert len(full_rows) > len(finals)
        assert [answer.ordinal for row in task_rows for answer in row.answers] == [row.ordinal for row in finals]
        assert [answer.native_uid for row in task_rows for answer in row.answers] == [row.native_uid for row in finals]
        boundary_traces = read_evaluation_traces(
            str(root / "exports"), attempt_id=generated_attempt_id, step=2, evaluation="eval/"
        )
        if not resume:
            original_traces = boundary_traces
            assert original_traces
        else:
            assert boundary_traces == original_traces
            at_boundary = [
                read_evaluations(str(root / "exports"), attempt_id=identity, step=2, evaluation="eval/")
                for identity in (generated_attempt_id, current_attempt)
            ]
            assert {row.task_id for row in at_boundary[0]} == {row.task_id for row in at_boundary[1]}
            assert all(
                {row.attempt_id for row in rows} == {identity}
                for identity, rows in zip((generated_attempt_id, current_attempt), at_boundary, strict=True)
            )
            assert all(len(row.answers) == 1 for rows in at_boundary for row in rows)
        for unscheduled_step in (0, 3):
            assert not read_evaluation_traces(
                str(root / "exports"), attempt_id=current_attempt, step=unscheduled_step, evaluation="eval/full_task_set/"
            )
        observed = Counter(
            row["attributes"]["step"]
            for row in cat_count_session
            if row["name"] == "training_metric_value"
            and row["attributes"].get("metric") == "eval/full_task_set/all/num_attempted"
        )
        assert observed == Counter({str(step): 1 for step in expected_final})
        assert [row["step"] for row in records] == list(range(steps + 1))
        if preserved is not None:
            assert records[1] == preserved
        training = [row for row in records if "policy/raw_grad_norm" in row]
        assert [row["step"] for row in training] == list(range(1, steps + 1))
        assert all(row["generate/prompts_dropped_too_long"] == 1 for row in training)
        assert all(0 <= row["reward/zero_std_group_fraction"] <= 1 for row in training)
        assert any(row["reward/zero_std_group_fraction"] > 0 for row in training)
        assert sum(records[-1][f"eval/all/outcome/{outcome.value}"] for outcome in Outcome) == 20
        evaluations = [row for row in records if "eval/all/timing/grading/mean" in row]
        timed_training = [row for row in training if "timing/grading/mean" in row]
        assert evaluations and timed_training
        for rows, prefix in ((timed_training, ""), (evaluations, "eval/all/")):
            for row in rows:
                for raw in TASK_TIMING_KEYS:
                    values = [row[f"{prefix}{raw}/{stat}"] for stat in ("mean", "p50", "p95", "max")]
                    assert all(math.isfinite(value) and value >= 0 for value in values)
                    assert values[1] <= values[2] <= values[3]
                assert row[f"{prefix}timing/grading/mean"] > 0
                assert row[f"{prefix}timing/setup/max"] == 0
        if not resume:
            preserved = records[1]


@pytest.mark.timeout(RUN_TIMEOUT_SECONDS)
def test_cat_count_final_profile_skips_zero_progress(
    tmp_path, cat_count_policy, cat_count_session, forkserver_context
):
    root = tmp_path / "zero-progress"
    cfg = cat_count_config(root, cat_count_policy, steps=1)
    cfg.trainer.epochs = 0
    cfg.generator.inference_stats_interval = 1
    cfg.trainer.callbacks = [
        {
            "type": "evaluation",
            "eval_steps": 1,
            "eval_before_train": True,
            "eval_on_train_end": False,
            "additional_evaluations": {"final_task_set": {"schedule": "final"}},
        }
    ]
    records = run_process(forkserver_context, root, cfg)
    assert any("eval/all/num_attempted" in row for row in records)
    assert all("policy/raw_grad_norm" not in row for row in records)
    assert all(not any(key.startswith("eval/final_task_set/") for key in row) for row in records)
    assert any(row["name"] == "generation_tokens_total" for row in cat_count_session)


@pytest.mark.timeout(RUN_TIMEOUT_SECONDS)
def test_cat_count_generate_preserves_task_ids_and_token_metrics(
    tmp_path, cat_count_policy, cat_count_session, forkserver_context
):
    root = tmp_path / "generate"
    cfg = cat_count_config(root, cat_count_policy)
    cfg.trainer.logger = "records"
    OmegaConf.update(cfg, "runtime.model", "cat-count-model", force_add=True)
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
        if n == 7:
            spec = spec.model_copy(update={"tags": ("agentic",)})
        if n in (7, 1):
            spec = spec.model_copy(
                update={
                    "grader": verifyit_package(
                        ExactSpec(expected=("WRONG",) if n == 7 else (TARGET_WORD,), substring=n == 1)
                    ).grader
                }
            )
            return lowered_task(spec, max_turns=1)
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
    write_tasks(path, [*reversed(tasks), tasks[0], overlong.model_copy(update={"task": spec})])
    write_selection_panel(
        path.parent, [row.task for row in tasks] + [spec], "cat-count", "canary-1",
        lambda task: {"family": "verifier" if task.id.endswith(("7", "1")) else "counts", "source": "cat-count", "agentic": task.id.endswith("7")},
    )
    cfg.data.val_data = [str(path)]
    cfg.generator.num_inference_engines = 2
    cfg.generator.eval_n_samples_per_prompt = 2
    cfg.generator.eval_sampling_params.ignore_eos = True
    cfg.generator.eval_sampling_params.max_generate_length = 8
    cfg.trainer.callbacks = [
        {
            "type": "evaluation",
            "additional_evaluations": {
                "named_profile": {
                    "n_samples_per_prompt": 1,
                    "sampling_params": {"ignore_eos": True, "max_generate_length": 4},
                }
            },
        }
    ]
    swapped = root / "swapped-tokenizer"
    shutil.copytree(cat_count_policy, swapped)
    tokenizer_path = swapped / "tokenizer.json"
    tokenizer_document = json.loads(tokenizer_path.read_text())
    vocabulary = tokenizer_document["model"]["vocab"]
    added = {token["content"] for token in tokenizer_document["added_tokens"]}
    first, second = [token for token in vocabulary if token not in added][:2]
    vocabulary[first], vocabulary[second] = vocabulary[second], vocabulary[first]
    tokenizer_path.write_text(json.dumps(tokenizer_document))
    invalid = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    invalid.trainer.policy.model.tokenizer_path = str(swapped)
    with pytest.raises(ValueError, match="differs from loaded tokenizer") as mismatch:
        TinyGenerateExp(invalid)
    assert str(cat_count_policy) in str(mismatch.value) and str(swapped) in str(mismatch.value)
    missing = root / "model-without-vocabulary"
    missing.mkdir()
    shutil.copyfile(cat_count_policy / "config.json", missing / "config.json")
    invalid.trainer.policy.model.path = str(missing)
    invalid.trainer.policy.model.tokenizer_path = str(cat_count_policy)
    with pytest.raises(ValueError, match="Cannot verify checkpoint vocabulary") as unavailable:
        TinyGenerateExp(invalid)
    assert str(missing) in str(unavailable.value) and str(cat_count_policy) in str(unavailable.value)
    records = run_process(forkserver_context, root, cfg, run_cat_count_generate)
    metrics, = records
    attempts = read_attempts(str(root / "exports"))
    (attempt_id,) = attempts
    assert attempts[attempt_id].evaluation_storage_version == 1
    rows = read_evaluation_traces(str(root / "exports"), attempt_id=attempt_id, step=-1, evaluation="eval/")
    named_rows = read_evaluation_traces(
        str(root / "exports"), attempt_id=attempt_id, step=-1, evaluation="eval/named_profile/"
    )
    assert {row.task_id for row in named_rows} == expected.keys()
    task_rows = read_evaluations(str(root / "exports"), attempt_id=attempt_id, step=-1, evaluation="eval/")
    assert {row.task_id for row in task_rows} == expected.keys()
    assert {row.model for row in task_rows} == {"cat-count-model"}
    panel_rows = pq.read_table(path.parent / "task_manifest.parquet").to_pylist()
    assert {row.task_id: row.task_digest for row in task_rows} == {
        value["task_id"]: value["task_digest"] for value in panel_rows if value["task_id"] in expected
    }
    assert {row.task_id for row in task_rows if row.strata.agentic} == {"stable-cat-count-7"}
    assert {row.task_id: len(row.answers) for row in task_rows} == {
        task_id: 4 if task_id.endswith("7") else 2 for task_id in expected
    }
    assert all(
        (row.panel_name, row.panel_version, row.strata.source) == ("cat-count", "canary-1", "cat-count")
        for row in task_rows
    )
    answers = sorted((answer for row in task_rows for answer in row.answers), key=lambda answer: answer.ordinal)
    assert [answer.ordinal for answer in answers] == list(range(len(rows)))
    assert {row.attempt_id for row in rows} == {attempt_id}
    assert {row.step for row in rows} == {-1} and all(row.is_last_step for row in rows)
    assert math.fsum(answer.optimization_reward for answer in answers) / len(answers) == metrics["eval/all/avg_score"]
    scored = [answer.verifier_score for answer in answers if answer.verifier_score is not None]
    assert sum(scored) / len(scored) == pytest.approx(metrics["eval/all/avg_verifier_score"])
    assert len(scored) / len(answers) == metrics["eval/all/verifier_score_coverage"]
    by_ordinal = {row.ordinal: row for row in rows}
    for task_row in task_rows:
        for answer in task_row.answers:
            trace = by_ordinal[answer.ordinal]
            assert (task_row.task_id, answer.native_uid, answer.response_token_count) == (
                trace.task_id, trace.native_uid, len(trace.response_token_ids)
            )
            assert answer.optimization_reward == trace.score
            verification = json.loads(trace.details_json)["verification"] or {}
            assert (answer.error_type, answer.error_treatment) == (trace.exception_type, trace.error_treatment)
            assert (answer.verification_status, answer.raw_verifier_score, answer.score_min, answer.score_max) == (
                verification.get("status"), verification.get("score"), verification.get("score_min"),
                verification.get("score_max"),
            )
            if verification.get("status") == "verified":
                assert answer.passed == (
                    verification["passed"] if verification["passed"] is not None else answer.optimization_reward > 0
                )
                assert answer.outcome == "graded"
            else:
                assert answer.verifier_score is None and not answer.passed and answer.outcome == "unavailable"
        task_scores = [row.verifier_score for row in task_row.answers if row.verifier_score is not None]
        assert task_row.score == (sum(task_scores) / len(task_scores) if task_scores else None)
    bounded = [
        answer
        for row in task_rows
        if row.task_id in {"stable-cat-count-4", "stable-cat-count-2"}
        for answer in row.answers
    ]
    assert len(bounded) == 4
    assert {
        (answer.raw_verifier_score, answer.score_min, answer.score_max, answer.verifier_score) for answer in bounded
    } == {(3.0, 1.0, 5.0, 0.5)}
    error_task = next(row for row in task_rows if row.task_id == "rollout-error")
    assert error_task.score is None and error_task.outcome == "unavailable"
    verifier_family = [row for row in task_rows if row.strata.family == "verifier"]
    assert sorted(row.score for row in verifier_family) == [0.0, 1.0]
    assert sum(row.score for row in verifier_family) / len(verifier_family) == 0.5
    family_answers = [answer.verifier_score for row in verifier_family for answer in row.answers]
    assert sum(family_answers) / len(family_answers) == pytest.approx(1 / 3)
    assert metrics["generate/prompts_dropped_too_long"] == 1
    assert len(rows) == 2 * (len(expected) + 1)
    assert {row.task_id for row in rows} == expected.keys()
    assert all(expected[row.task_id] in row.input_prompt and row.native_uid != row.task_id for row in rows)
    assert all(len(row.response_token_ids) == 8 for row in rows if row.task_id != "rollout-error")
    tokens = sum(len(row.response_token_ids) for row in rows)
    assert metrics["generate/response_tokens"] == tokens
    assert metrics["generate/wall_seconds"] > 0
    generated_tokens = 2 * (len(expected) + 1) * 8
    assert metrics["generate/vllm/generation_tokens"] == generated_tokens
    assert metrics["generate/tokens_per_second_per_gpu"] == pytest.approx(
        generated_tokens / metrics["generate/wall_seconds"] / 2
    )
    assert metrics["generate/vllm/num_engines"] == 2
    assert {outcome.value: metrics[f"eval/all/outcome/{outcome.value}"] for outcome in Outcome} == {
        "graded": 10,
        "submission_failure": 0,
        "invalid_task": 0,
        "infra_error": 0,
        "unavailable": 2,
    }
    assert metrics["eval/all/outcome/rollout_failure"] == 2
    for raw in TASK_TIMING_KEYS:
        assert all(math.isfinite(metrics[f"eval/all/{raw}/{stat}"]) for stat in ("mean", "p50", "p95", "max"))
    assert metrics["eval/all/timing/grading/mean"] > 0
    assert metrics["eval/all/timing/setup/max"] == 0
    mirrored = {
        row["attributes"]["metric"]
        for row in cat_count_session
        if row["name"] == "training_metric_value" and row["attributes"].get("payload_kind") == "generate"
    }
    assert {"generate/wall_seconds", "generate/response_tokens", "eval/all/num_attempted"} <= mirrored
    assert {
        (row["resource"]["service"], row["resource"]["attributes"].get("model"))
        for row in cat_count_session if row["name"] == "training_metric_value"
    } == {("marinskyrl", "cat-count-model")}
    engine_rows = [row for row in cat_count_session if row["name"] == "generation_tokens_total"]
    assert metrics["eval/named_profile/all/num_attempted"] == len(expected) + 1
    assert len(engine_rows) == 4
    assert sum(row["value"] for row in engine_rows[:2]) == generated_tokens
    assert sum(row["value"] for row in engine_rows[2:]) == generated_tokens + (len(expected) + 1) * 4


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
    groups_per_family = {"odd": 4, "even": 4}
    first = train(forkserver_context, root, cat_count_policy, steps=4, staleness=1, groups_per_family=groups_per_family)
    before_resume = len(cat_count_session)
    resumed = train(
        forkserver_context,
        root,
        cat_count_policy,
        steps=100,
        staleness=1,
        resume=True,
        eval_interval=2,
        groups_per_family=groups_per_family,
    )
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

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import click
import pytest
import yaml
from levanter import model_cache
from marin.execution.artifact import RECORD_FILENAME, Artifact, ArtifactRecord, result_type_name
from marin.execution.lazy import OUT, ArtifactStep, apply, materialized_config, run
from marin.rl import skyrl as skyrl_api
from marin.rl.skyrl import ArtifactHfModel
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath
from rigging.provenance import Provenance

from experiments.post_training import rl as snowball
from experiments.post_training import rl_gate as gate
from experiments.post_training import rl_recipe as recipe
from experiments.post_training import snowball_rl as snowball_model
from experiments.post_training.rl_command import grafana_url
from experiments.post_training.skyrl_test_support import assert_revisioned_native_configs, native_json
from experiments.post_training.task_curation.selection import FULL_VALIDATION_FILENAME, SELECTION_PREFIX

pytestmark = [pytest.mark.usefixtures("source_provenance"), pytest.mark.timeout(180)]


def test_preset_variants_compose(selection, tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_ENTITY", "test-team")
    learning = snowball_model.SNOWBALL.by_name["learning-medium"]
    changed = snowball.build_run(
        learning,
        settings=(
            'trainer={"policy_mini_batch_size":64,"train_batch_size":64,"bf16":false,"algorithm":{"batch_invariant":true}}',
            "generator.model_dtype=float16",
            "generator.n_samples_per_prompt=8",
            "trainer.micro_train_batch_size_per_gpu=2",
            "trainer.policy.optimizer_config.lr=2e-6",
            "generator.engine_init_kwargs.probe={}",
        ),
        table=snowball_model.SNOWBALL,
    ).runs[0]
    promoted = replace(
        learning,
        recipe=replace(
            learning.recipe,
            config=learning.recipe.config.with_settings(
                (
                    "trainer.train_batch_size=64",
                    "trainer.policy_mini_batch_size=64",
                    "generator.n_samples_per_prompt=8",
                    "trainer.micro_train_batch_size_per_gpu=2",
                    "trainer.policy.optimizer_config.lr=2e-6",
                )
            ),
        ),
    )
    checkpoint = StoragePath("s3://marin-test/models/checkpoint")
    export = checkpoint / "hf"
    files = {str(export / name).removeprefix("s3://") for name in ("config.json", "tokenizer_config.json")}
    monkeypatch.setattr("s3fs.S3FileSystem.glob", lambda self, path, **kwargs: [str(export / "config.json")])
    monkeypatch.setattr("s3fs.S3FileSystem.modified", lambda self, path: datetime(2026, 1, 1, tzinfo=UTC))
    monkeypatch.setattr(
        "s3fs.S3FileSystem.exists",
        lambda self, path, **kwargs: str(path).removeprefix("s3://") in files | {str(checkpoint).removeprefix("s3://")},
    )
    metadata = {"train_config": {"data": {"tokenizer": "tests/own-tokenizer"}}}
    monkeypatch.setattr(model_cache, "model_info", lambda *args, **kwargs: SimpleNamespace(sha="a" * 40))
    initializer = ArtifactStep.adopt(
        "tests/checkpoint", "2026.11.10", str(checkpoint), kind=LevanterCheckpoint, config=metadata
    )
    alias = tmp_path / initializer.name / initializer.version
    alias.mkdir(parents=True)
    (alias / RECORD_FILENAME).write_text(
        ArtifactRecord(
            name=initializer.name,
            version=initializer.version,
            result_type=result_type_name(LevanterCheckpoint),
            config=metadata,
            source=str(checkpoint),
        ).model_dump_json()
    )
    cli_initializer = recipe.published_model(f"{initializer.name}@{initializer.version}", preset=learning)
    published = []
    for suffix, config in (
        ("own", {}),
        (
            "explicit",
            {
                "tokenizer_uri": "tests/top-tokenizer",
                "tokenizer_revision": "top-revision",
                "train_config": {"data": {"tokenizer": "tests/nested-ignored"}},
            },
        ),
    ):
        name = "tests/checkpoint-" + suffix
        record = tmp_path / name / initializer.version / RECORD_FILENAME
        record.parent.mkdir(parents=True)
        record.write_text(
            ArtifactRecord(
                name=name,
                version=initializer.version,
                result_type=result_type_name(Artifact),
                config=config,
                source=str(checkpoint),
            ).model_dump_json()
        )
        published.append(recipe.published_model(f"{name}@{initializer.version}", preset=learning))

    revisioned = replace(snowball_model.SNOWBALL.by_name["speed-short"], revision="2026.11.12.2")
    steps = (
        changed,
        snowball.build_run(promoted, table=snowball_model.SNOWBALL).runs[0],
        snowball.build_run(
            learning,
            init=ArtifactHfModel(initializer, "tests/own-tokenizer", "tokenizer-revision"),
            table=snowball_model.SNOWBALL,
        ).runs[0],
        snowball.build_run(
            snowball_model.SNOWBALL.by_name["train-long"],
            settings=("trainer.max_steps=2",),
            table=snowball_model.SNOWBALL,
        ).runs[0],
        snowball.build_run(revisioned, control=True, table=snowball_model.SNOWBALL).runs[0],
        snowball.build_run(learning, init=cli_initializer, table=snowball_model.SNOWBALL).runs[0],
        *(snowball.build_run(learning, init=model, table=snowball_model.SNOWBALL).runs[0] for model in published),
    )
    paths = []
    for index, step in enumerate(steps):
        path = tmp_path / f"variant-{index}.yaml"
        path.write_text(materialized_config(step, str(tmp_path)).launch_config_yaml)
        paths.append(str(path))
    configs = gate.compose_configs(paths)
    for config in configs[:2]:
        trainer, generator = config["skyrl"]["trainer"], config["skyrl"]["generator"]
        assert (
            trainer["train_batch_size"],
            trainer["policy_mini_batch_size"],
            generator["n_samples_per_prompt"],
            trainer["micro_train_batch_size_per_gpu"],
            trainer["micro_forward_batch_size_per_gpu"],
            trainer["rollout_buffer"]["max_in_flight"],
            trainer["policy"]["optimizer_config"]["lr"],
        ) == (64, 64, 8, 2, 2, 64, 2e-6)
        assert (
            trainer["policy"]["optimizer_config"]["weight_decay"],
            trainer["policy"]["optimizer_config"]["adam_betas"],
        ) == (0, [0.9, 0.999])
    trainer, generator = configs[0]["skyrl"]["trainer"], configs[0]["skyrl"]["generator"]
    assert (
        trainer["bf16"],
        generator["model_dtype"],
        trainer["algorithm"]["batch_invariant"],
        generator["engine_init_kwargs"]["probe"],
    ) == (False, "float16", True, {})
    assert configs[2]["inputs"]["model"]["uri"] == str(export)
    assert configs[2]["iris"]["wandb_entity"] == "test-team"
    for config, revision in ((configs[2], "tokenizer-revision"), (configs[5], "a" * 40)):
        assert (config["inputs"]["model"]["tokenizer_uri"], config["inputs"]["model"]["tokenizer_revision"]) == (
            "tests/own-tokenizer",
            revision,
        )
        assert config["inputs"]["model"]["uri"] == str(export)
    assert (configs[6]["inputs"]["model"]["tokenizer_uri"], configs[6]["inputs"]["model"]["tokenizer_revision"]) == (
        None,
        None,
    )
    assert (configs[7]["inputs"]["model"]["tokenizer_uri"], configs[7]["inputs"]["model"]["tokenizer_revision"]) == (
        "tests/top-tokenizer",
        "top-revision",
    )
    callbacks = configs[3]["skyrl"]["trainer"]["callbacks"]
    assert next(row for row in callbacks if row["type"] == "evaluation")["eval_on_train_end"] is True
    assert configs[3]["skyrl"]["trainer"]["max_steps"] == 2
    assert configs[4]["inputs"]["validation_data"][0]["identity"].partition(":")[0].endswith("@" + revisioned.revision)
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))

    def export_model(output):
        destination = Path(output) / "hf"
        destination.mkdir(parents=True)
        for name in ("config.json", "tokenizer_config.json"):
            (destination / name).write_text("{}")
        return Artifact(path=output)

    producer = apply("tests/computed-init", export_model, version="2026.11.17", output=OUT)
    own_model = ArtifactHfModel(producer)
    train = snowball_model.SNOWBALL.by_name["train-long"]

    def reject_export_read(*args, **kwargs):
        raise AssertionError("planning read an unbuilt export")

    with monkeypatch.context() as planning:
        planning.setattr(skyrl_api, "discover_hf_checkpoints", reject_export_read)
        planned = snowball.build_run(train, init=own_model, table=snowball_model.SNOWBALL)
        assert producer in planned.runs[0].deps
        planned.runs[0].lower()
        assert (
            snowball.main(
                snowball_model.SNOWBALL, argv=["--preset", train.name, "--version", train.revision], init=own_model
            )
            == 0
        )
    assert not Path(producer.path()).exists()
    run(producer)
    document = yaml.safe_load(materialized_config(planned.runs[0], str(tmp_path)).launch_config_yaml)
    assert document["inputs"]["model"]["uri"] == str(Path(producer.path()) / "hf")
    assert document["inputs"]["model"]["tokenizer_uri"] is None
    assert document["inputs"]["model"]["tokenizer_revision"] is None
    best = next(row for row in document["skyrl"]["trainer"]["callbacks"] if row["type"] == "best_checkpoint")
    assert best["initial_model_uri"] == document["inputs"]["model"]["uri"]
    with pytest.raises(ValueError, match="require guards"):
        replace(learning, guards=())
    with pytest.raises(ValueError, match="evaluation"):
        replace(snowball_model.SNOWBALL.by_name["train-long"], evaluations=())
    summary = Path(recipe.selection_artifact(learning.tasks).path()) / recipe.SUMMARY_FILENAME
    content = json.loads(summary.read_text())
    content["grader_kind_counts"]["train"]["remote_judge"] = 1
    summary.write_text(json.dumps(content))
    with pytest.raises(click.BadParameter, match="AI-judge"):
        snowball.build_run(learning, table=snowball_model.SNOWBALL)


def test_harbor_coverage_controls_best_checkpoint(selection, tmp_path):
    paths = []
    for index, settings in enumerate(((), ("context_budget.serving_window_tokens=28672",))):
        step = snowball.build_run(
            snowball_model.SNOWBALL.by_name["train-long-harbor"], table=snowball_model.SNOWBALL, settings=settings
        ).runs[0]
        path = tmp_path / f"harbor-{index}.yaml"
        document = yaml.safe_load(materialized_config(step, str(tmp_path)).launch_config_yaml)
        if index:
            callback = next(
                row for row in document["skyrl"]["trainer"]["callbacks"] if row["type"] == "harbor_evaluation"
            )
            for cohort in callback["cohorts"]:
                cohort["job"]["agents"][0]["kwargs"]["model_info"]["max_output_tokens"] = 4096
        path.write_text(yaml.safe_dump(document))
        paths.append(str(path))
    selected = native_json("--isolated", "--group", "harbor-test")(
        """
import asyncio, json, sys
from pathlib import Path
from types import SimpleNamespace
from cloud.iris.launch_config import load_launch_config
from harbor.models.job.config import JobConfig
from omegaconf import OmegaConf
from skyrl_train.callbacks.base import CallbackHandler, TrainerState, TrainerControl
from skyrl_train.callbacks.builtin import HarborEvaluationCallback, create_callbacks_from_config
config, changed = (load_launch_config(path) for path in map(Path, sys.argv[1:]))
windows = []
for launch in (config, changed):
    window = launch.skyrl.generator.engine_init_kwargs.max_model_len
    windows.append(window)
    harbor = next(row for row in launch.skyrl.trainer.callbacks if row.type == "harbor_evaluation")
    for cohort in harbor.cohorts:
        job = JobConfig.model_validate(OmegaConf.to_container(cohort.job, resolve=True))
        model_info = job.agents[0].kwargs["model_info"]
        assert model_info["max_input_tokens"] + model_info["max_output_tokens"] == window
        assert job.retry.max_retries == 1
        assert job.retry.include_exceptions == {"PhaseCancellationError", "TimeoutError", "DaytonaConflictError"}
        if cohort.name == "swebench-verified":
            assert job.verifier.override_timeout_sec == 3000
            agent = job.agents[0]
            assert agent.override_setup_timeout_sec + agent.override_timeout_sec + 3000 < job.trial_attempt_timeout_sec
        else:
            assert job.verifier.override_timeout_sec == 300
            assert job.trial_attempt_timeout_sec == 1800
        assert job.n_concurrent_trials == len(job.tasks)
        attempt_budget = job.trial_attempt_timeout_sec + job.trial_cleanup_grace_sec
        assert 2 * attempt_budget + job.retry.min_wait_sec < harbor.timeout
assert windows[0] != windows[1]
selected = []
for index, (native, cohorts) in enumerate(((99, (27, 27, 27)), (98, (30, 30, 30)), (100, (26, 30, 30)))):
    directory = Path(sys.argv[1]).parent / f"coverage-{index}"
    config.skyrl.trainer.ckpt_path = str(directory)
    metrics = {"eval/all/num_attempted": 100, "eval/all/num_scored": native, "eval/all/avg_verifier_score": 0.5}
    harbor = next(row for row in config.skyrl.trainer.callbacks if row.type == "harbor_evaluation")
    for cohort, scored in zip(harbor.cohorts, cohorts, strict=True):
        prefix = f"eval/harbor/{cohort.name}/"
        metrics.update({prefix + "num_attempted": 30, prefix + "num_scored": scored, prefix + "reward_sum": scored / 2})
    handler = CallbackHandler(create_callbacks_from_config(config.skyrl))
    state = TrainerState(global_step=0, epoch=0, total_steps=100, num_steps_per_epoch=100)
    asyncio.run(handler.call_event_async("on_evaluate", state, TrainerControl(), metrics=metrics,
                                       trainer=SimpleNamespace(cfg=config.skyrl)))
    selected.append((directory / "best_checkpoint.json").exists())

from contextlib import nullcontext
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.client.client import IrisContext, iris_ctx_scope
from iris.cluster.client.job_info import JobInfo, get_job_info, set_job_info
from iris.cluster.endpoints import LOG_SERVER_ENDPOINT_NAME, TELEMETRY_ENDPOINT_PATH
from iris.cluster.types import JobName
from cloud.iris.skyrl_entrypoint import run_config
from cloud.iris.rl_config_translation import registered_rl_entrypoint_module
from skyrl_train.evaluation_pool import EvaluationPool

native_configs = []
module_name = registered_rl_entrypoint_module(str(config.runtime.entrypoint))
original_module = sys.modules.get(module_name)
sys.modules[module_name] = SimpleNamespace(run=native_configs.append)
try:
    run_config(Path(sys.argv[1]))
finally:
    if original_module is None:
        del sys.modules[module_name]
    else:
        sys.modules[module_name] = original_module

worker = r'''
import json, socket, sys, traceback
from harbor.models.job.config import JobConfig
expected = json.loads(sys.argv[2])
connection = socket.socket(fileno=int(sys.argv[1]))
with connection.makefile("rwb") as stream:
    for line in stream:
        message = json.loads(line)
        try:
            job = JobConfig.model_validate(message["cohort"]["job"])
            if expected is None:
                assert job.telemetry is None
            else:
                assert job.telemetry is not None
                actual = {
                    "endpoint": job.telemetry.endpoint.get_secret_value(),
                    "root_run_uid": job.telemetry.root_run_uid,
                    "execution_uid": job.telemetry.execution_uid,
                    "serving_job_id": job.telemetry.serving_job_id,
                }
                assert actual == expected, (actual, expected)
            response = {"id": message["id"], "metrics": {"num_attempted": 1, "num_scored": 1, "reward_sum": 1}}
        except Exception:
            response = {"id": message["id"], "error": traceback.format_exc()}
        stream.write((json.dumps(response) + "\\n").encode())
        stream.flush()
'''

class EndpointClient:
    def __init__(self, missing):
        self.missing = missing

    def resolve_endpoint(self, name):
        assert name == LOG_SERVER_ENDPOINT_NAME
        if self.missing:
            raise ConnectError(Code.NOT_FOUND, "endpoint unavailable")
        return "http://127.0.0.1:4318/iris-capability/"

async def check_runtime_handoff():
    callback = next(item for item in create_callbacks_from_config(config.skyrl)
                    if isinstance(item, HarborEvaluationCallback))
    trainer = SimpleNamespace(
        cfg=native_configs[0],
        inference_engine_client=SimpleNamespace(
            enable_http_endpoint=True, model_name="policy", http_endpoint_host="127.0.0.1", http_endpoint_port=8000,
        ),
    )
    original_exec = asyncio.create_subprocess_exec
    previous = get_job_info()
    expected = None

    async def start_worker(*args, **kwargs):
        return await original_exec(sys.executable, "-c", worker, args[-1], json.dumps(expected), **kwargs)

    asyncio.create_subprocess_exec = start_worker
    try:
        for attempt in (1, 2, None, "missing"):
            trainer.cfg = native_configs[0] if attempt is not None else config.skyrl
            info = JobInfo(task_id=JobName.from_wire(f"/test/native-{attempt}/0"), attempt_uid=f"uid-{attempt}")
            set_job_info(info if attempt is not None else None)
            expected = {
                "endpoint": "http://127.0.0.1:4318/iris-capability" + TELEMETRY_ENDPOINT_PATH,
                "root_run_uid": str(config.run.id),
                "execution_uid": f"iris:uid-{attempt}",
                "serving_job_id": f"/test/native-{attempt}",
            } if attempt not in (None, "missing") else None
            context = IrisContext(job_id=info.job_id, task_attempt=info.task_attempt,
                                  client=EndpointClient(missing=attempt == "missing"))
            scope = iris_ctx_scope(context) if attempt is not None else nullcontext()
            with scope:
                async with EvaluationPool(3) as pool:
                    metrics = await callback.collect_evaluation(state, trainer=trainer, pool=pool)
                assert all(metrics[f"eval/harbor/{cohort['name']}/num_scored"] == len(cohort["job"]["tasks"])
                           for cohort in callback.cohorts)
    finally:
        asyncio.create_subprocess_exec = original_exec
        set_job_info(previous)

import os
os.environ["DAYTONA_EVAL_API_KEY"] = "test-eval-credential"
asyncio.run(check_runtime_handoff())
print(json.dumps(selected))
""",
        paths,
    )
    assert selected == [True, False, False]


def test_presets_match_revisioned_native_configs(selection, tmp_path, monkeypatch):
    revisions = {
        "speed-short": {"2026.11.13.1": "96e0d088", "2026.11.17.1": "363225a9"},
        "data-short": {"2026.11.13.1": "7d9a590f", "2026.11.17.1": "772467a8"},
        "speed-medium": {"2026.11.12.1": "e75349ff", "2026.11.17.1": "5cd852f4"},
        "learning-medium": {"2026.11.12.1": "ba638a12", "2026.11.17.1": "2be6e5e2"},
        "data-medium": {"2026.11.12.1": "ba638a12", "2026.11.17.1": "2be6e5e2"},
        "speed-long": {"2026.11.11.1": "cd0a5f4e", "2026.11.17.1": "f29155cc"},
        "train-long": {"2026.11.11.1": "4945b701", "2026.11.17.1": "b411c765"},
        "train-long-harbor": {"2026.11.17.2": "5a71e7e8", "2026.11.18.1": "7c3c9478"},
    }
    configs = assert_revisioned_native_configs(
        snowball_model.SNOWBALL, revisions, tmp_path=tmp_path, monkeypatch=monkeypatch
    )
    for name, config in zip(revisions, configs, strict=True):
        trainer, generator = config["skyrl"]["trainer"], config["skyrl"]["generator"]
        assert config["iris"]["wandb_entity"] == "marin-community"
        assert config["iris"]["allocation"]["num_nodes"] * config["iris"]["allocation"]["gpus_per_node"] <= 40
        generating = name in ("speed-short", "data-short")
        assert (
            config["runtime"]["training_type"],
            trainer["rollout_buffer"]["max_staleness_steps"],
            trainer["algorithm"]["off_policy_correction"],
            trainer["rollout_buffer"]["batch_policy"],
        ) == ((None, 0, "none", "full_batch") if generating else ("async", 2, "tis", "full_batch"))
        if generating:
            assert config["runtime"]["entrypoint"] == "skyrl_train.entrypoints.main_generate"
            assert config["iris"]["allocation"]["num_nodes"] == 1
            assert config["skyrl"]["trajectory_runner"]["max_concurrent_tasks"] == (
                2048 if name == "speed-short" else 1024
            )
            assert trainer["eval_interval"] > 0 and config["inputs"]["validation_data"]
            sampling = generator["eval_sampling_params"]
            assert (sampling["temperature"], sampling["max_generate_length"], sampling["ignore_eos"]) == (
                1,
                4096 if name == "speed-short" else 8192,
                name == "speed-short",
            )
        else:
            assert generator["apply_overlong_filtering"] is False
        assert (trainer["bf16"], generator["model_dtype"], trainer["algorithm"]["batch_invariant"]) == (
            True,
            "bfloat16",
            False,
        )
        if name in ("speed-short", "learning-medium", "data-medium", "train-long"):
            profile = next(
                (
                    row["additional_evaluations"]["eval_lite"]
                    for row in trainer["callbacks"]
                    if "eval_lite" in row.get("additional_evaluations", {})
                ),
                None,
            )
            for source in config["inputs"].get("evaluation_data", {}).get("eval_lite", []):
                assert (
                    source["identity"].partition(":")[0]
                    == f"{SELECTION_PREFIX}/{snowball_model.SNOWBALL.by_name[name].eval_lite}"
                )
                assert source["relative_path"] == "full_validation.parquet"
            sampling = profile["sampling_params"] if profile else generator["eval_sampling_params"]
            assert (
                sampling["temperature"],
                sampling["ignore_eos"],
                sampling["max_generate_length"],
                profile["n_samples_per_prompt"] if profile else generator["eval_n_samples_per_prompt"],
            ) == (0, False, 8192, 1)
        if name in ("learning-medium", "data-medium"):
            assert next(row for row in trainer["callbacks"] if row["type"] == "evaluation")["eval_before_train"] is True
            assert config["run"]["export_hf"] and "hf_model_save" in [row["type"] for row in trainer["callbacks"]]
        if name == "speed-long":
            assert not config["inputs"]["validation_data"] and not config["run"]["export_hf"]
        if name == "train-long":
            kinds = [row["type"] for row in trainer["callbacks"]]
            assert kinds.index("evaluation") < kinds.index("best_checkpoint")
            assert next(row for row in trainer["callbacks"] if row["type"] == "evaluation")["eval_on_train_end"] is False


def test_run_identity_tracks_code_inputs_and_replicates(selection, monkeypatch, tmp_path, recorded_runs):
    learning = snowball_model.SNOWBALL.by_name["learning-medium"]
    data = snowball_model.SNOWBALL.by_name["data-short"]
    with pytest.raises(click.BadParameter, match="accepts no --set"):
        snowball.build_run(
            replace(learning, accepts_settings=False),
            settings=("trainer.policy.optimizer_config.lr=2e-6",),
            table=snowball_model.SNOWBALL,
        )

    def run(preset=learning, *settings, **kwargs):
        return snowball.build_run(preset, settings=settings, **kwargs, table=snowball_model.SNOWBALL).runs[0]

    def name(preset=learning, *settings, **kwargs):
        return run(preset, *settings, **kwargs).name

    assert name(learning, "generator.engine_init_kwargs.probe={}") != name()
    repeated = snowball.build_run(learning, replicates=3, table=snowball_model.SNOWBALL)
    assert [step.name for step in snowball.build_run(learning, replicates=2, table=snowball_model.SNOWBALL).runs] == [
        step.name for step in repeated.runs[:2]
    ]
    assert len({step.name for step in repeated.runs}) == 3
    initializer = ArtifactStep.adopt("tests/initializer", "2026.10.26", selection.path(), kind=LevanterCheckpoint)
    task_set_name = snowball_model.DEFAULT_EVAL_LITE_NAME
    task_set_version = snowball_model.DEFAULT_EVAL_LITE_VERSION
    new_task_set = tmp_path / task_set_name / "2026.11.05"
    new_task_set.mkdir(parents=True)
    (new_task_set / "summary.json").write_bytes(
        (tmp_path / task_set_name / task_set_version / "summary.json").read_bytes()
    )
    changed_task_set = replace(
        learning,
        eval_lite=f"{task_set_name}@2026.11.05",
    )
    pairs = [
        (name(), name(learning, "trainer.policy.optimizer_config.lr=2e-6")),
        (name(data, tasks=selection), name(data, tasks=replace(selection, version="2026.10.27"))),
        (name(), name(init=ArtifactHfModel(initializer, "tests/own-tokenizer", "tokenizer-revision", relative_path=""))),
        (name(), name(changed_task_set)),
    ]
    assert all(left != right for left, right in pairs), pairs
    assert name(
        learning,
        'trainer.policy.optimizer_config={"lr": 2e-06}',
        'trainer.policy.optimizer_config={"weight_decay": 0.1}',
    ) == name(
        learning, "trainer.policy.optimizer_config.lr=0.000002", "trainer.policy.optimizer_config.weight_decay=0.1"
    )
    assert name(learning, "trainer.policy.optimizer_config.lr=2e-6") == name(
        learning, "trainer.policy.optimizer_config.lr=0.000002"
    )
    pin = gate.control_pin(snowball_model.SNOWBALL.by_name["learning-medium"], table=snowball_model.SNOWBALL)
    for index in range(pin.runs):
        recorded_runs("learning-medium", gate.control_name(pin.preset, pin.commit, index, table=snowball_model.SNOWBALL))
    first = snowball.build_run(learning, requirements=("eval_lite/mean>=0",), table=snowball_model.SNOWBALL)
    second = snowball.build_run(learning, requirements=("eval_lite/mean>=0.1",), table=snowball_model.SNOWBALL)
    assert first.runs[0].name == second.runs[0].name == name()
    assert first.gate.name != second.gate.name
    original_name = name()
    monkeypatch.setenv("WANDB_ENTITY", "another-team")
    assert name() == original_name
    monkeypatch.setattr(
        Provenance, "capture", lambda *args, **kwargs: Provenance("tree-two", "commit-one", False, "branch", "test")
    )
    assert name() != original_name

    summary_path = Path(selection.path()) / "summary.json"
    summary = json.loads(summary_path.read_text())
    original_exclusions = summary["recipe"]["exclusions"]
    heldout = summary["recipe"]["heldout"]
    summary["recipe"]["heldout"] = [{"version": "different-development-input"}]
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(click.BadParameter, match="different heldout"):
        snowball.build_run(data, tasks=selection, table=snowball_model.SNOWBALL)
    summary["recipe"]["heldout"] = heldout
    summary_path.write_text(json.dumps(summary))
    for current in (data, snowball_model.SNOWBALL.by_name["data-medium"]):
        selected = replace(current, eval_lite=f"{task_set_name}@2026.11.05")
        with pytest.raises(click.BadParameter, match="full EvalLite"):
            snowball.build_run(selected, tasks=selection, table=snowball_model.SNOWBALL)
        summary["recipe"]["exclusions"] = [
            {"uri": str(new_task_set / FULL_VALIDATION_FILENAME), "version": "2026.11.05"}
        ]
        summary_path.write_text(json.dumps(summary))
        assert snowball.build_run(selected, tasks=selection, table=snowball_model.SNOWBALL).gate is not None
        summary["recipe"]["exclusions"] = original_exclusions
        summary_path.write_text(json.dumps(summary))
    summary["recipe"]["exclusions"] = []
    summary_path.write_text(json.dumps(summary))
    for tasks in (recipe.selection_artifact(snowball_model.SNOWBALL.by_name["learning-medium"].tasks), selection):
        with pytest.raises(click.BadParameter, match="full EvalLite"):
            snowball.build_run(data, tasks=tasks, table=snowball_model.SNOWBALL)


def test_grafana_url_selects_native_run_and_utc_window():
    result = urlparse(grafana_url("users/test/run", "marin-east", "rl", "2026-11-06T18:40:00Z", "2026-11-06T19:40:00Z"))
    assert result.path == "/d/marin-rl-runs/rl-post-training"
    assert parse_qs(result.query) == {
        "orgId": ["1"],
        "timezone": ["utc"],
        "from": ["1793990400000"],
        "to": ["1793994000000"],
        "var-training_type": ["rl"],
        "var-cluster": ["marin-east"],
        "var-run": ["users/test/run"],
        "var-job": ["$__all"],
        "var-execution": ["$__all"],
    }
    open_window = parse_qs(urlparse(grafana_url("generation", "marin-east", None, "2026-11-06T18:40:00Z")).query)
    assert open_window["to"] == ["now"] and open_window["var-training_type"] == ["sync"]

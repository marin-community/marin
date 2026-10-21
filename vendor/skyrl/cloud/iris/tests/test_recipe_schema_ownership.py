import copy
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, get_args

from omegaconf import OmegaConf

from cloud.iris import rl_config_translation as launcher
from cloud.iris import task_runtime
from cloud.iris.rl_data import resolve_rl_train_data_with_sources
from marinskyrl import recipe_schema as schema, runtime_options, speculative_decoding
from marinskyrl.recipe_schema.documents import MISSING, get_path, leaves, set_path
from marinskyrl.recipe_schema.ownership import DERIVED_PATHS, LAUNCH_PATHS
from marinskyrl.recipe_schema.rules import SKYRL_INTERNAL_ENGINE_KWARGS
from scripts.recipe_schema_sidecar import OPEN, TYPES, UNDECLARED
from skyrl_train.config.ftpo import FTPOBoundary, FTPOConfig
from skyrl_train.config.objective_spec import KLEstimator, LossReduction

UNTYPED_LAUNCH_OVERRIDES = {
    "discard": frozenset({"data.train_data", "data.val_data"}),
    "clamp": frozenset({"trainer.placement.policy_num_gpus_per_node", "trainer.placement.ref_num_gpus_per_node"}),
}

FILL_WHEN_UNSET = frozenset(
    {
        "trainer.run_name",
        "trainer.placement.policy_num_nodes",
        "trainer.placement.ref_num_nodes",
        "generator.num_inference_engines",
        "terminal_bench.installed_agent.output_path",
    }
)


def test_ownership_sentinels_distinguish_launch_writers_context_writers_and_pending_discards(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[3]
    parsed = launcher.parse_rl_config(str(root / "cloud/iris/configs/tasktrove_dq_sweep_30b.yaml"))
    base = launcher.compose_skyrl_config(parsed, {}, SimpleNamespace(gpus_per_node=8)).config
    full = OmegaConf.to_container(base, resolve=False)
    full["terminal_bench"] = full.pop("terminal_bench_config")
    full["terminal_bench"]["installed_agent"] = {"output_path": "author-output"}
    for path in OPEN | {"terminal_bench"}:
        set_path(full, f"{path}.author_options", {"label": "authored", "items": ["authored", {"enabled": True}]})
    parameters = {
        "job_name": "ownership-launch",
        "experiments_dir": "/launch/runs",
        "num_nodes": 2,
        "gpus_per_node": 8,
        "model_path": "/launch/model",
        "model_revision": "launch-revision",
        "model_source_uri": "s3://launch/model",
        "model_source_identity": "sha256:launch",
        "train_data": ["/launch/train"],
        "val_data": ["/launch/validation"],
        "checkpoint_root": "/launch/checkpoints",
        "export_root": "/launch/exports",
        "resume_checkpoint_count": 4,
        "seed": 41,
        "trajectory_root": "/launch/trajectories",
        "export_hf_artifact": True,
    }
    monkeypatch.setattr(task_runtime.tempfile, "tempdir", str(tmp_path))
    suffix, number = "B", 4
    seeded = copy.deepcopy(full)
    for parts, value in leaves(full):
        sentinel = (
            not value
            if isinstance(value, bool)
            else 31
            if isinstance(value, int)
            else 3.125
            if isinstance(value, float)
            else [f"/author/{suffix}"]
            if isinstance(value, list)
            else f"author-{suffix}"
        )
        set_path(seeded, ".".join(parts), sentinel)
    for path in LAUNCH_PATHS:
        set_path(seeded, path, False if path == "trainer.export_hf_artifact" else f"author-{suffix}")
    for path in FILL_WHEN_UNSET | UNTYPED_LAUNCH_OVERRIDES["clamp"]:
        set_path(seeded, path, f"author-{suffix}" if path == "trainer.run_name" else number)
    section_values = {
        name: seeded[name]
        for name in ("trainer", "generator", "data", "environment", "trajectory_runner", "terminal_bench")
    }
    injected = replace(parsed, **section_values)
    filled = launcher._skyrl_config_sections(injected, parameters, SimpleNamespace(gpus_per_node=8))
    data = resolve_rl_train_data_with_sources(injected.data["terminal_bench_data"], kind="tasks", verbose=False)
    staged = launcher.apply_task_local_values(
        OmegaConf.create(filled),
        launcher.TaskLocalSkyRLValues(
            train_data=tuple(parameters["train_data"]),
            validation_data=tuple(parameters["val_data"]),
            terminal_bench_data=tuple(data.paths),
        ),
    )
    launch = OmegaConf.create(
        {
            "run": {"id": f"ownership-{suffix}", "attempt_id": "1"},
            "inputs": {"model": {"uri": "/launch/model"}},
            "skyrl": staged,
        }
    )
    written = task_runtime._write_final_config(
        launch,
        policy_model=task_runtime.PreparedPolicyModel("s3://launch/model", "sha256:launch", "/stage/model"),
        policy_tokenizer=task_runtime.PreparedPolicyTokenizer("/stage/tokenizer"),
        draft_model=None,
    )
    result = OmegaConf.to_container(OmegaConf.load(written).skyrl, resolve=False)
    result["terminal_bench"] = result.pop("terminal_bench_config")
    changed = {
        ".".join(parts) for parts, before in leaves(section_values) if get_path(result, ".".join(parts)) != before
    }
    assert LAUNCH_PATHS <= changed
    assert changed - LAUNCH_PATHS == UNTYPED_LAUNCH_OVERRIDES["discard"]
    for path in FILL_WHEN_UNSET | UNTYPED_LAUNCH_OVERRIDES["clamp"]:
        assert get_path(result, path) == get_path(seeded, path)
    absent = copy.deepcopy(section_values)
    for path in FILL_WHEN_UNSET:
        set_path(absent, path, None)
    defaults = launcher._skyrl_config_sections(replace(parsed, **absent), parameters, SimpleNamespace(gpus_per_node=8))
    defaults["terminal_bench"] = defaults.pop("terminal_bench_config")
    for path in FILL_WHEN_UNSET:
        assert get_path(defaults, path) not in (MISSING, None)
    oversized = copy.deepcopy(section_values)
    for path in UNTYPED_LAUNCH_OVERRIDES["clamp"]:
        set_path(oversized, path, 31)
    clamped = launcher._skyrl_config_sections(
        replace(parsed, **oversized), parameters, SimpleNamespace(gpus_per_node=8)
    )
    for path in UNTYPED_LAUNCH_OVERRIDES["clamp"]:
        assert get_path(clamped, path) == 8
    following = OmegaConf.create(full)
    following.generator.max_turns = 987654
    assert following.environment.task_sessions.session.max_turns == 987654
    derived = {}
    context_paths = DERIVED_PATHS - {"environment.task_sessions.session.max_turns"}
    for path in context_paths:
        set_path(derived, path, 987654)
    _, _, _, materialized = launcher._materialize_context_budget(derived, parsed.context_budget)
    assert {
        ".".join(parts) for parts, before in leaves(derived) if get_path(materialized, ".".join(parts)) != before
    } == context_paths
    budget = schema.ContextBudget(
        **{
            key: getattr(parsed.context_budget, key)
            for key in ("request_window_tokens", "serving_window_tokens", "max_new_tokens_per_turn", "max_turns")
        }
    )
    assert set(budget.derived_values({})) == DERIVED_PATHS


def test_code_default_declarations_match_native_dataclasses_and_canonical_choices():
    assert SKYRL_INTERNAL_ENGINE_KWARGS == launcher.SKYRL_INTERNAL_ENGINE_KWARGS
    for prefix, config in (
        ("trainer.algorithm.ftpo", FTPOConfig),
        ("generator.speculative_decoding.training", speculative_decoding.SpeculatorTrainingConfig),
    ):
        declared = {path.removeprefix(prefix + ".") for path in UNDECLARED if path.startswith(prefix + ".")}
        assert declared == {field.name for field in fields(config)}
    canonical = {
        "trainer.algorithm.policy_loss_type": runtime_options.PolicyLossType,
        "trainer.algorithm.loss_reduction": LossReduction,
        "trainer.algorithm.kl_estimator_type": KLEstimator,
        "trainer.evaluation_runner": runtime_options.EvaluationRunner,
        "generator.r3_transport": runtime_options.R3Transport,
        "generator.weight_sync_transport": runtime_options.WeightSyncTransport,
        "generator.gdn_backend": runtime_options.GDNBackend,
    }
    for path, enum in canonical.items():
        assert set(get_args(eval(TYPES[path], {"Literal": Literal}))) == {choice.value for choice in enum}, path
    assert set(get_args(eval(UNDECLARED["trainer.algorithm.ftpo.boundary"][0], {"Literal": Literal}))) == {
        choice.value for choice in FTPOBoundary
    }

import copy
import inspect
from collections.abc import Iterator, Mapping
from dataclasses import fields, replace
from types import SimpleNamespace
from typing import Any, get_args

from omegaconf import OmegaConf
from skyrl_train.callbacks.builtin import CALLBACK_REGISTRY
from skyrl_train.config.ftpo import FTPOConfig
from skyrl_train.config.objective_spec import CorrectionAction, SequenceAggregate
from skyrlrecipe.declared import Callback, SequenceRule, Teacher, TokenRule
from skyrlrecipe.documents import MISSING, get_path, set_path
from skyrlrecipe.ownership import LAUNCH_PATHS

from cloud.iris import rl_config_translation as launcher
from cloud.iris import task_runtime
from cloud.iris.rl_data import resolve_rl_train_data_with_sources
from marinskyrl import speculative_decoding
from marinskyrl.distillation import TeacherEvidenceKind, TeacherPlacement, TeacherSource
from scripts.recipe_schema_sidecar import OPEN, UNDECLARED

GPU_PATHS = frozenset({"trainer.placement.policy_num_gpus_per_node", "trainer.placement.ref_num_gpus_per_node"})

FILL_WHEN_UNSET = frozenset(
    {
        "trainer.run_name",
        "trainer.placement.policy_num_nodes",
        "trainer.placement.ref_num_nodes",
        "generator.num_inference_engines",
        "terminal_bench.installed_agent.output_path",
    }
)


def leaves(document: Mapping[str, Any], prefix: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], Any]]:
    """Yield indivisible values; empty mappings contribute a leaf."""
    for key, value in document.items():
        path = (*prefix, key)
        if isinstance(value, Mapping) and value:
            yield from leaves(value, path)
        else:
            yield path, value


def test_ownership_sentinels_distinguish_launch_writers_context_writers_and_authored_values(tmp_path, monkeypatch):
    source = tmp_path / "ownership.yaml"
    source.write_text(
        "entrypoint: terminal_bench\nterminal_bench: {}\ncontext_budget:\n  request_window_tokens: 2048\n  max_new_tokens_per_turn: 512\n  max_turns: 1\n"
    )
    parsed = launcher.parse_rl_config(OmegaConf.load(source))
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
        "train_data": [],
        "val_data": [],
        "checkpoint_root": "/launch/checkpoints",
        "export_root": "/launch/exports",
        "resume_checkpoint_count": 4,
        "seed": 41,
        "trajectory_root": "/launch/trajectories",
        "export_hf_artifact": True,
    }
    monkeypatch.setattr(task_runtime.tempfile, "tempdir", str(tmp_path))
    seeded = copy.deepcopy(full)
    for parts, value in leaves(full):
        sentinel = {bool: not value, int: 31, float: 3.125, list: ["/author/B"]}.get(type(value), "author-B")
        set_path(seeded, ".".join(parts), sentinel)
    for path in LAUNCH_PATHS:
        set_path(seeded, path, False if path == "trainer.export_hf_artifact" else "author-B")
    for path in FILL_WHEN_UNSET | GPU_PATHS:
        value = "author-B" if path in {"trainer.run_name", "terminal_bench.installed_agent.output_path"} else 4
        set_path(seeded, path, value)
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
            "run": {"id": "ownership-B", "attempt_id": "1"},
            "inputs": {"model": {"uri": "/launch/model"}},
            "artifacts": {"attempts_root": str(tmp_path / "attempts")},
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
    assert changed == LAUNCH_PATHS
    for path in FILL_WHEN_UNSET | GPU_PATHS:
        assert get_path(result, path) == get_path(seeded, path)
    absent = copy.deepcopy(section_values)
    for path in FILL_WHEN_UNSET:
        set_path(absent, path, None)
    defaults = launcher._skyrl_config_sections(replace(parsed, **absent), parameters, SimpleNamespace(gpus_per_node=8))
    defaults["terminal_bench"] = defaults.pop("terminal_bench_config")
    for path in FILL_WHEN_UNSET:
        assert get_path(defaults, path) not in (MISSING, None)


def test_code_default_declarations_match_native_dataclasses_and_canonical_choices():
    for prefix, config in (
        ("trainer.algorithm.ftpo", FTPOConfig),
        ("generator.speculative_decoding.training", speculative_decoding.SpeculatorTrainingConfig),
    ):
        declared = {path.removeprefix(prefix + ".") for path in UNDECLARED if path.startswith(prefix + ".")}
        assert declared == {field.name for field in fields(config)}
    for model, name, enum in (
        (Teacher, "source", TeacherSource),
        (Teacher, "placement", TeacherPlacement),
        (Teacher, "evidence", TeacherEvidenceKind),
        (TokenRule, "action", CorrectionAction),
        (SequenceRule, "action", CorrectionAction),
        (SequenceRule, "aggregate", SequenceAggregate),
    ):
        annotation = model.model_fields[name].annotation
        if name == "placement":
            annotation = next(option for option in get_args(annotation) if option is not type(None))
        assert set(get_args(annotation)) == {choice.value for choice in enum}, name

    callbacks = get_args(get_args(Callback)[0])
    declared = {get_args(callback.model_fields["type"].annotation)[0]: callback for callback in callbacks}
    assert declared.keys() == CALLBACK_REGISTRY.keys()
    for name, callback in declared.items():
        parameters = {
            key: parameter
            for key, parameter in inspect.signature(CALLBACK_REGISTRY[name].__init__).parameters.items()
            if key != "self"
        }
        assert set(callback.model_fields) - {"type"} == parameters.keys(), name
        assert {key for key, field in callback.model_fields.items() if key != "type" and field.is_required()} == {
            key for key, parameter in parameters.items() if parameter.default is inspect.Parameter.empty
        }, name

import subprocess
from collections import Counter
from pathlib import Path

import yaml

import infra.check_env_var_contract as env_contract
from marinskyrl.environment_contract import (
    ENV_VAR_SPECS,
    EXECUTION_UID_ENV,
    EnvVarScope,
    EnvVarSource,
    EnvVarSpec,
    EnvVarWriter,
    EnvVarManager,
)
from omegaconf import OmegaConf


def test_no_new_environment_variable_definition_sites(monkeypatch):
    monkeypatch.setattr("sys.argv", ["check-env-var-contract"])
    assert env_contract.main() == 0


def test_contract_rejects_unregistered_literal_definition(tmp_path, monkeypatch):
    (tmp_path / "runtime.py").write_text('import os\nos.environ["NEW_TOGGLE"] = "1"\n')
    monkeypatch.setattr(env_contract, "REPO_ROOT", tmp_path)

    errors = env_contract.contract_errors(env_contract.definitions(), ENV_VAR_SPECS)

    assert any("runtime.py::python-assignment::NEW_TOGGLE" in error for error in errors)


def test_contract_rejects_wrong_writer_kind(tmp_path, monkeypatch):
    (tmp_path / "runtime.py").write_text(
        """\
import os

runtime_env = {"env_vars": {"MAPPED_TOGGLE": "1"}}
os.environ.update({"UPDATED_TOGGLE": "1"})
subprocess_options = dict(env={"KEYWORD_TOGGLE": "1"})

def worker_environment():
    return {"RETURNED_TOGGLE": "1"}
"""
    )
    monkeypatch.setattr(env_contract, "REPO_ROOT", tmp_path)
    specs = tuple(
        EnvVarSpec(name, "trainer.test", EnvVarSource.CONFIG, frozenset({EnvVarScope.DRIVER}))
        for name in ("MAPPED_TOGGLE", "UPDATED_TOGGLE", "KEYWORD_TOGGLE", "RETURNED_TOGGLE")
    )

    errors = env_contract.contract_errors(env_contract.definitions(), specs)

    assert len(errors) == 4
    assert all("writer kind" in error for error in errors)


def test_contract_accepts_declared_external_writer():
    spec = EnvVarSpec(
        "EXTERNAL_TOOL_SETTING",
        "external.tool",
        EnvVarSource.EXTERNAL,
        frozenset({EnvVarScope.TASK_RUNTIME}),
        frozenset({EnvVarWriter.SHELL_EXPORT}),
    )

    assert env_contract.contract_errors(Counter({"tool.sh::shell-export::EXTERNAL_TOOL_SETTING": 1}), (spec,)) == []


def test_contract_ignores_gitignored_scratch_files(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("/temp/\n")
    scratch_dir = tmp_path / "temp"
    scratch_dir.mkdir()
    (scratch_dir / "probe.sh").write_text("export SCRATCH_ONLY_TOGGLE=1\n")
    monkeypatch.setattr(env_contract, "REPO_ROOT", tmp_path)

    assert env_contract.definitions() == Counter()


def test_managed_names_have_one_owner_and_config_control():
    names = [spec.name for spec in ENV_VAR_SPECS]
    assert len(names) == len(set(names))

    config = yaml.safe_load(Path("skyrl-train/skyrl_train/config/ppo_base_config.yaml").read_text())
    for spec in ENV_VAR_SPECS:
        if spec.source not in {EnvVarSource.CONFIG, EnvVarSource.DERIVED}:
            continue
        value = config
        for component in spec.owner.split("."):
            assert component in value, f"{spec.name} has no config owner at {spec.owner}"
            value = value[component]


def test_typed_process_boundary_settings_project_only_to_workers():
    config = OmegaConf.create(
        {
            "trainer": {
                "debug_mode": "off",
                "collective_phase_diagnostics": False,
                "placement": {"enable_numa_affinity": True},
                "numerics": {"mode": "native"},
            },
            "generator": {"fuse_weights": True},
        }
    )

    manager = EnvVarManager.from_config(config, environ={})

    assert manager.environment_for(EnvVarScope.DRIVER) == {}
    assert manager.environment_for(EnvVarScope.RAY_WORKER) == {
        "SKYRL_ENABLE_NUMA_AFFINITY": "1",
        "RAY_USE_UVLOOP": "0",
        "UV_USE_IO_URING": "0",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
    }
    assert manager.environment_for(EnvVarScope.INFERENCE_WORKER) == {
        "SKYRL_ENABLE_NUMA_AFFINITY": "1",
        "RAY_USE_UVLOOP": "0",
        "UV_USE_IO_URING": "0",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
    }


def test_the_execution_uid_does_not_project_to_ray_workers():
    # The head's attempt uid would override each node's own once it reaches a worker's runtime_env.
    manager = EnvVarManager.from_config({}, environ={EXECUTION_UID_ENV: "iris:01JABCDEF0123456789"})

    assert EXECUTION_UID_ENV not in manager.environment_for(EnvVarScope.RAY_WORKER)
    assert manager.environment_for(EnvVarScope.TASK_RUNTIME)[EXECUTION_UID_ENV] == "iris:01JABCDEF0123456789"


def test_non_debug_worker_projection_does_not_require_an_artifact_directory():
    config = OmegaConf.create(
        {
            "trainer": {
                "debug_mode": "off",
                "placement": {"enable_numa_affinity": True},
                "numerics": {"mode": "native"},
            },
            "generator": {"fuse_weights": False},
        }
    )
    environ = {}

    applied = EnvVarManager.from_config(config, environ=environ).apply_to_process(
        EnvVarScope.RAY_WORKER, environ=environ
    )

    assert applied == {
        "SKYRL_ENABLE_NUMA_AFFINITY": "1",
        "RAY_USE_UVLOOP": "0",
        "UV_USE_IO_URING": "0",
    }
    assert environ == applied

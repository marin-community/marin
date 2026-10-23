#!/usr/bin/env python3
"""Resolve task-local inputs and run SkyRL from one Hydra launch document.

This process runs on rank zero after ``task_runtime.py`` has bootstrapped the
cross-node Ray cluster. The only process boundary is ``--config``; staged model
paths, resolved datasets, and SkyRL settings remain structured
configuration throughout the launch.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

from omegaconf import DictConfig, OmegaConf

from cloud.iris.artifacts import fs_and_path
from cloud.iris.rl_data import (
    check_rl_environment,
    compute_num_inference_engines,
    resolve_rl_train_data_with_sources,
)
from marinskyrl.process_diagnostics import ProcessOutcomeKind, write_process_outcome
from marinskyrl.resource_locator import is_cloud_uri, model_source_for_path
from cloud.iris.launch_config import RunMode, load_launch_config
from cloud.iris.rl_config_translation import TaskLocalSkyRLValues, apply_task_local_values


@dataclass
class LocalRLConfig:
    """Configuration for the in-container RL runner."""

    job_name: str
    model_path: str
    model_source_uri: str | None = None
    model_source_identity: str | None = None
    train_data: List[str | dict[str, Any]] = field(default_factory=list)
    val_data: List[str | dict[str, Any]] = field(default_factory=list)
    experiments_dir: str = "experiments"
    resolved_config_uri: str | None = None
    gpus: int = 4
    # Multi-node placement. The external controller has already bootstrapped one
    # cross-node Ray cluster and exported RAY_ADDRESS; this runner ATTACHES to it,
    # and gpus_per_node drives the SkyRL placement + num_inference_engines.
    num_nodes: int = 1
    gpus_per_node: int = 0  # 0 = use `gpus`
    tensor_parallel_size: int = 1
    launch_config: DictConfig | None = None

    def __post_init__(self) -> None:
        model_source_for_path(self.model_path, self.model_source_uri, self.model_source_identity)


class LocalRLRunner:
    """Runs SkyRL training attached to an externally-managed Ray cluster."""

    def __init__(self, config: LocalRLConfig):
        self.config = config
        self._train_data_sources = list(config.train_data)
        self._val_data_sources = list(config.val_data)
        self._processes: List[subprocess.Popen] = []
        self.rl_env_path: Path | None = None

    def setup(self) -> None:
        """Validate configuration and set up directories."""
        rl_env = check_rl_environment()
        if rl_env:
            print(f"RL environment found: {rl_env}")
            self.rl_env_path = rl_env
        else:
            # On Iris, sys.executable is already the frozen task venv's Python.
            self.rl_env_path = None

        skyrl_home = os.environ.get("SKYRL_HOME")
        if skyrl_home and Path(skyrl_home).exists():
            print(f"SkyRL home: {skyrl_home}")
            skyrl_train = os.path.join(skyrl_home, "skyrl-train")
            if skyrl_train not in sys.path:
                sys.path.insert(0, skyrl_train)
            pythonpath = os.environ.get("PYTHONPATH", "")
            if skyrl_train not in pythonpath:
                os.environ["PYTHONPATH"] = f"{skyrl_train}:{pythonpath}"
        else:
            print("\nWARNING: SKYRL_HOME not found! Set SKYRL_HOME to the MarinSkyRL clone.")

        experiments_dir = Path(self.config.experiments_dir).expanduser().resolve()
        experiments_dir.mkdir(parents=True, exist_ok=True)
        self.config.experiments_dir = str(experiments_dir)

        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        def handle_signal(signum, _frame):
            print(f"\nSignal {signum} received; shutting down...", file=sys.stderr)
            self.cleanup()
            sys.exit(1)

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

    def cleanup(self) -> None:
        for proc in self._processes:
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()

    def print_banner(self) -> None:
        print("=== MarinSkyRL Iris Training Runner ===")
        print(f"  Job Name: {self.config.job_name}")
        print("  Launch Config: loaded")
        print(f"  Model: {self.config.model_path}")
        print(f"  GPUs: {self.config.gpus}")
        print(f"  Train Data: {self.config.train_data}")
        print(f"  Validation Data: {self.config.val_data}")
        print(f"  Experiments Dir: {self.config.experiments_dir}")
        print("=======================================")

    def _write_resolved_config(self, config: DictConfig) -> None:
        if not self.config.resolved_config_uri:
            return
        record = OmegaConf.to_container(
            OmegaConf.create(
                {
                    "config": config,
                    "train_data_sources": self._train_data_sources,
                    "val_data_sources": self._val_data_sources,
                }
            ),
            resolve=True,
        )
        assert isinstance(record, dict)
        filesystem, path = fs_and_path(self.config.resolved_config_uri)
        with filesystem.open(path, "w") as destination:
            json.dump(record, destination, sort_keys=True)

    def _resolve_data_inputs(self, data_kind: str) -> None:
        for role, attribute in (("train", "train_data"), ("validation", "val_data")):
            values = getattr(self.config, attribute)
            if not values:
                continue
            print(f"\nResolving {role} data (kind={data_kind}): {values}")
            resolved = resolve_rl_train_data_with_sources(values, kind=data_kind)
            paths = list(resolved.paths)
            sources = list(resolved.sources)
            setattr(self.config, attribute, paths)
            setattr(self, f"_{attribute}_sources", sources)
            print(f"Resolved {role} data: {paths}")

    def run(self) -> int:
        """Execute the RL training job. Returns an exit code (0 for success)."""
        self.print_banner()

        launch_config = self.config.launch_config
        if launch_config is None:
            raise ValueError("training driver requires a loaded launch config")
        skyrl_config = OmegaConf.create(OmegaConf.to_container(launch_config.skyrl, resolve=False))
        if launch_config.run.mode == RunMode.CHECKPOINT_EXPORT:
            self._write_resolved_config(launch_config)
            return self._run_skyrl(launch_config)
        self._resolve_data_inputs(str(launch_config.inputs.data_kind))
        terminal_bench_data = skyrl_config.get("data", {}).get("terminal_bench_data", ())
        if terminal_bench_data:
            terminal_bench_data = tuple(
                resolve_rl_train_data_with_sources(list(terminal_bench_data), kind="tasks").paths
            )
        self._setup_environment()
        skyrl_config = apply_task_local_values(
            skyrl_config,
            TaskLocalSkyRLValues(
                train_data=tuple(self.config.train_data),
                validation_data=tuple(self.config.val_data),
                terminal_bench_data=terminal_bench_data,
            ),
        )
        launch_config.skyrl = skyrl_config
        self._write_resolved_config(launch_config)
        return self._run_skyrl(launch_config)

    def _gpus_per_node(self) -> int:
        """GPUs per node, defaulting to total `gpus` for the single-node case."""
        return self.config.gpus_per_node or self.config.gpus

    def _setup_environment(self) -> None:
        """Configure environment variables for RL training."""
        os.environ["TENSOR_PARALLEL_SIZE"] = str(self.config.tensor_parallel_size)
        os.environ["NUM_INFERENCE_ENGINES"] = str(
            compute_num_inference_engines(
                self.config.num_nodes,
                self._gpus_per_node(),
                self.config.tensor_parallel_size,
            )
        )
        os.environ["POLICY_NUM_NODES"] = str(self.config.num_nodes)

        os.environ["VLLM_USE_V1"] = "1"

        wandb_dir = os.path.join(self.config.experiments_dir, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        os.environ["WANDB_DIR"] = wandb_dir

        print("\nEnvironment configured:")
        print(f"  TENSOR_PARALLEL_SIZE={os.environ['TENSOR_PARALLEL_SIZE']}")
        print(f"  NUM_INFERENCE_ENGINES={os.environ['NUM_INFERENCE_ENGINES']}")
        print(f"  WANDB_DIR={wandb_dir}")

    def _run_skyrl(self, launch_config: DictConfig) -> int:
        """Exec the SkyRL entrypoint attached to the externally-managed Ray cluster.

        The controller exported RAY_ADDRESS, so SkyRL's ``initialize_ray()`` (a bare
        ``ray.init()``) attaches to the existing cluster; this runner must NOT call
        its own ``ray.init(num_cpus=, num_gpus=)`` (forbidden when attaching).
        """
        ray_address = os.environ.get("RAY_ADDRESS")
        if not ray_address:
            print(
                "ERROR: RAY_ADDRESS is not set. This runner attaches to a Ray cluster "
                "bootstrapped by task_runtime.py; run it via that controller.",
                file=sys.stderr,
            )
            return 1
        print(
            f"\nAttaching to external Ray cluster at {ray_address} "
            f"(num_nodes={self.config.num_nodes}, gpus_per_node={self._gpus_per_node()})"
        )

        python_exe = str(self.rl_env_path / "bin" / "python") if self.rl_env_path else sys.executable
        config_path = Path(tempfile.gettempdir()) / "marinskyrl" / "skyrl-launch.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(launch_config, config_path, resolve=True)
        cmd = [python_exe, "-m", "cloud.iris.skyrl_entrypoint", "--config", str(config_path)]

        print("\nRunning SkyRL:")
        print(f"  Entrypoint: {launch_config.runtime.entrypoint}")
        print(f"  Config: {config_path}")

        skyrl_home = os.environ.get("SKYRL_HOME")
        cwd = None
        if skyrl_home:
            candidate = os.path.join(skyrl_home, "skyrl-train")
            if os.path.isdir(candidate):
                cwd = candidate
                print(f"  Working dir: {cwd}")

        print(f"\nCommand: {' '.join(cmd[:3])} [... {len(cmd) - 3} more args]")
        sys.stdout.flush()

        proc = subprocess.Popen(cmd, cwd=cwd)
        self._processes.append(proc)
        returncode = proc.wait()
        outcome, receipt = write_process_outcome(
            "skyrl-entrypoint",
            returncode,
            pid=proc.pid,
            metadata={"entrypoint": launch_config.runtime.entrypoint},
        )
        if outcome.kind is ProcessOutcomeKind.SIGNAL:
            print(
                f"SkyRL entrypoint pid={proc.pid} terminated by {outcome.signal_name} "
                f"(raw_returncode={returncode}, exit_code={outcome.public_exit_code}, receipt={receipt})",
                file=sys.stderr,
                flush=True,
            )
        return outcome.public_exit_code


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    return parser


def main() -> None:
    args = create_parser().parse_args()
    launch_config = load_launch_config(args.config)
    allocation = launch_config.iris.allocation
    model_uri = str(launch_config.inputs.model.uri)
    model_is_cloud = is_cloud_uri(model_uri)
    config = LocalRLConfig(
        job_name=str(launch_config.iris.job_name),
        model_path=str(launch_config.skyrl.trainer.policy.model.path),
        model_source_uri=model_uri if model_is_cloud else None,
        model_source_identity=str(launch_config.inputs.model.identity) if model_is_cloud else None,
        train_data=list(launch_config.inputs.train_data),
        val_data=list(launch_config.inputs.validation_data),
        experiments_dir=str(launch_config.runtime.experiments_dir),
        resolved_config_uri=str(launch_config.artifacts.resolved_config_uri),
        gpus=int(allocation.num_nodes * allocation.gpus_per_node),
        num_nodes=int(allocation.num_nodes),
        gpus_per_node=int(allocation.gpus_per_node),
        tensor_parallel_size=int(launch_config.skyrl.generator.inference_engine_tensor_parallel_size),
        launch_config=launch_config,
    )

    runner = LocalRLRunner(config)
    runner.setup()
    sys.exit(runner.run())


if __name__ == "__main__":
    main()

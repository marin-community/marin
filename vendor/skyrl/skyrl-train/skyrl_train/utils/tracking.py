# This code is adapted from VERL
# https://github.com/volcengine/verl/blob/main/verl/utils/tracking.py
# The original copyright is reproduced below:
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
from enum import Enum
from functools import partial
from hashlib import sha256
from pathlib import Path
from datetime import datetime, timezone
from types import TracebackType
from typing import Any, Dict, Self
from collections.abc import Sequence
from uuid import uuid4
from loguru import logger
from rigging.secrets import require_secret
from rigging.provenance import launch_provenance
from omegaconf import DictConfig, OmegaConf, open_dict
import pprint
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from finestore.rl.run_records import (
    METRIC_ATTEMPT_IDS,
    METRIC_METADATA_KEYS,
    METRICS_TABLE,
    AttemptRecord,
    write_attempt,
    CheckpointReference,
    MetricRow,
    read_attempt,
    register_run_tables,
    run_records_root,
    update_attempt,
)
from finestore.store import DataStore

_WANDB_STEP_METRIC = "trainer/global_step"
_WANDB_GROUP_NAME_LIMIT = 128
# Bound SDK uploads too: timing out a Ray get alone leaves the primary finish unbounded.
_WANDB_LOGGER_SHUTDOWN_TIMEOUT = 30


def _wandb_group_name(run_id: str) -> str:
    encoded = run_id.encode("utf-8")
    # Budget UTF-8 bytes so multibyte labels also fit the service limit.
    if len(encoded) <= _WANDB_GROUP_NAME_LIMIT:
        return run_id
    suffix = "-" + sha256(encoded).hexdigest()[:24]
    # The byte slice may leave a partial character at the end.
    prefix = encoded[: _WANDB_GROUP_NAME_LIMIT - len(suffix)].decode("utf-8", errors="ignore")
    return prefix + suffix


@ray.remote
class WandbNodeLogger:
    """
    A Ray actor that initializes wandb on a specific node to capture system metrics.
    """

    def __init__(self, project_name, experiment_name, config, run_id, group_name, x_label):
        import wandb

        # Initialize wandb with the same run ID to aggregate system metrics
        run = wandb.init(
            project=project_name,
            name=experiment_name,
            id=run_id,
            config=config,
            resume="allow",
            group=group_name,
            job_type="worker_monitor",
            settings=wandb.Settings(
                api_key=require_secret("WANDB_API_KEY"),
                mode="shared",
                x_primary=False,
                x_update_finish_state=False,
                x_label=x_label,
                finish_timeout=_WANDB_LOGGER_SHUTDOWN_TIMEOUT,
                finish_timeout_raises=True,
            ),
        )
        self.wandb = run

    def close(self) -> None:
        self.wandb.finish()


# TODO(tgriggs): Test all backends.
class Tracking:
    supported_backends = ["wandb", "mlflow", "swanlab", "tensorboard", "console", "records"]

    def __init__(self, project_name, experiment_name, backends: Sequence[str] = ("console",), config=None):
        for backend in backends:
            assert backend in self.supported_backends, f"{backend} is not supported"

        self.logger = {}
        self.remote_loggers = []
        tracking_config = OmegaConf.to_container(config, resolve=True) if config is not None else {}
        runtime = tracking_config.get("runtime", {})
        tracking_config.update(
            run_id=runtime.get("run_id", experiment_name),
            attempt_id=runtime.get("attempt_id", uuid4().hex[:12]),
            source_commit=launch_provenance().base_commit,
            iris_job_id=runtime.get("iris_job_id"),
            coordinator_iris_job_id=runtime.get("coordinator_iris_job_id"),
        )

        model = runtime.get("model")
        if model is not None:
            tracking_config["model"] = model

        if config is not None:
            with open_dict(config):
                config.runtime = {
                    **runtime,
                    "run_id": tracking_config["run_id"],
                    "attempt_id": tracking_config["attempt_id"],
                }
            write_attempt(
                str(config.trainer.export_path),
                AttemptRecord(
                    run_id=tracking_config["run_id"],
                    attempt_id=tracking_config["attempt_id"],
                    evaluation_storage_version=1,
                ),
            )

        if "records" in backends:
            self.logger["records"] = _RecordsAdapter(
                str(config.trainer.export_path), tracking_config["run_id"], tracking_config["attempt_id"]
            )

        if "wandb" in backends:
            import wandb

            group_name = _wandb_group_name(tracking_config["run_id"])
            current_node_ip = "head"
            if ray.is_initialized():
                try:
                    current_node_ip = ray.util.get_node_ip_address()
                except Exception as e:
                    logger.warning(f"Failed to get node IP address, defaulting to 'head'. Error: {e}")

            run = wandb.init(
                project=project_name,
                name=experiment_name,
                id=tracking_config["attempt_id"],
                config=tracking_config,
                group=group_name,
                tags=[model] if model is not None else None,
                resume="allow",
                settings=wandb.Settings(
                    api_key=require_secret("WANDB_API_KEY"),
                    mode="shared",  # mainly for multi-node training's systems metrics aggregation
                    x_primary=True,
                    x_label=f"node-{current_node_ip}",
                    finish_timeout=_WANDB_LOGGER_SHUTDOWN_TIMEOUT,
                    finish_timeout_raises=True,
                ),
            )
            run_id = run.id
            run.define_metric(_WANDB_STEP_METRIC)
            run.define_metric("*", step_metric=_WANDB_STEP_METRIC)
            self.logger["wandb"] = run
            self._prepare_worker_nodes_systems_logging_wandb(
                project_name, experiment_name, run_id, group_name, tracking_config, current_node_ip
            )

        if "mlflow" in backends:
            self.logger["mlflow"] = _MlflowLoggingAdapter(project_name, experiment_name, config)

        if "swanlab" in backends:
            import os

            import swanlab

            SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY", None)
            SWANLAB_LOG_DIR = os.environ.get("SWANLAB_LOG_DIR", "swanlog")
            SWANLAB_MODE = os.environ.get("SWANLAB_MODE", "cloud")
            if SWANLAB_API_KEY:
                swanlab.login(SWANLAB_API_KEY)  # NOTE: previous login information will be overwritten
            swanlab.init(
                project=project_name,
                experiment_name=experiment_name,
                config=config,
                logdir=SWANLAB_LOG_DIR,
                mode=SWANLAB_MODE,
            )
            self.logger["swanlab"] = swanlab

        if "tensorboard" in backends:
            self.logger["tensorboard"] = _TensorboardAdapter()

        if "console" in backends:
            self.console_logger = ConsoleLogger()
            self.logger["console"] = self.console_logger

    def _prepare_worker_nodes_systems_logging_wandb(
        self, project_name, experiment_name, run_id, group_name, config, current_node_ip
    ):
        """
        In multi-node training, we spawn WandbNodeLogger actors on each worker node to capture system metrics like
        GPU utilization. We use `mode="shared"` to aggregate system metrics from all nodes to the same Wandb run.
        However, with this approach, the systems metrics panels do not appear in the Wandb UI automatically but requires
        us to manually create the panels. The alternative is to create a run for each node and group them by group_name.
        We prefer to keep all nodes metrics to the same run.
        """
        # Start WandB loggers on other nodes using Ray
        if ray.is_initialized():
            try:
                # current_node_ip (head) is already set in __init__
                nodes = ray.nodes()
                for node in nodes:
                    if not node["Alive"]:
                        continue

                    node_ip = node["NodeManagerAddress"]
                    # Skip the current node as it's already logging
                    if node_ip == current_node_ip:
                        continue

                    try:
                        # Launch a logger on this node
                        logger_actor = WandbNodeLogger.options(
                            num_cpus=0,
                            scheduling_strategy=NodeAffinitySchedulingStrategy(
                                node_id=node["NodeID"],
                                soft=False,  # fail if that node is gone or infeasible
                            ),
                        ).remote(
                            project_name=project_name,
                            experiment_name=experiment_name,
                            config=config,
                            run_id=run_id,
                            group_name=group_name,
                            x_label=f"node-{node_ip}",
                        )
                        self.remote_loggers.append(logger_actor)
                    except Exception as e:
                        logger.warning(f"Failed to spawn WandbNodeLogger on {node_ip}: {e}")
                    logger.info(f"WandbNodeLogger initialized on 'node-{node_ip}'")

            except Exception as e:
                logger.warning(f"Failed to setup distributed wandb logging: {e}")
        else:
            logger.warning("Ray is not initialized, skipping distributed wandb logging")

    def start(self, *, resumed_step: int | None, checkpoint: CheckpointReference | None = None) -> None:
        """Record the checkpoint boundary for this execution's metric history."""
        if records := self.logger.get("records"):
            records.start(resumed_step=resumed_step, checkpoint=checkpoint)

    def log(self, data, step, commit=True):
        for logger_name, logger_instance in self.logger.items():
            if logger_name == "wandb":
                logger_instance.log(data={**data, _WANDB_STEP_METRIC: step}, commit=commit)
            else:
                logger_instance.log(data=data, step=step)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close(exit_code=0 if exception_type is None else 1)

    def close(self, *, exit_code: int) -> None:
        """Flush tracker backends without changing the workload's result."""
        if self.remote_loggers:
            try:
                ray.get([actor.close.remote() for actor in self.remote_loggers], timeout=_WANDB_LOGGER_SHUTDOWN_TIMEOUT)
            except Exception as error:
                logger.warning("Remote W&B tracker shutdown failed: {}", error)
        for name, backend in self.logger.items():
            try:
                if name == "wandb":
                    backend.finish(exit_code=exit_code)
                else:
                    backend.finish()
            except Exception as error:
                logger.warning("Tracker backend {} shutdown failed: {}", name, error)


class _RecordsAdapter:
    def __init__(self, export_root: str, run_id: str, attempt_id: str):
        self.export_root = export_root
        self.run_id = run_id
        self.attempt_id = attempt_id
        self.store = DataStore.open(run_records_root(export_root))
        self.metrics = register_run_tables(self.store)[METRICS_TABLE]

    def start(self, *, resumed_step: int | None, checkpoint: CheckpointReference | None) -> None:
        record = read_attempt(self.export_root, self.attempt_id)
        if record is not None and record.metrics_started_at is not None:
            raise ValueError(f"Records execution requires a fresh attempt_id: {self.attempt_id}")
        update_attempt(
            self.store,
            AttemptRecord(
                run_id=self.run_id,
                attempt_id=self.attempt_id,
                metrics_started_at=datetime.now(timezone.utc).isoformat(),
                resumed_step=resumed_step,
                parent=checkpoint,
            ),
        )

    def log(self, data: dict[str, Any], step: int) -> None:
        rows = []
        for name, value in data.items():
            if name in METRIC_METADATA_KEYS:
                if name == METRIC_ATTEMPT_IDS:
                    raise ValueError(f"{METRIC_ATTEMPT_IDS} is reserved for metric provenance")
                expected = step if name == "step" else self.attempt_id
                if value != expected:
                    raise ValueError(f"Metric metadata {name} differs from the active execution")
                continue
            rows.append(MetricRow(attempt_id=self.attempt_id, step=step, name=name, value=float(value)).model_dump())
        self.metrics.extend(rows)
        self.store.flush()

    def finish(self) -> None:
        self.store.close()


class ConsoleLogger:
    def __init__(self):
        pass

    def log(self, data: Dict[str, Any], step: int):
        # pprint the data and log with logger
        data_as_str = pprint.pformat(ConsoleLogger.stringify_floats(data))
        logger.info(f"Step {step}: \n{data_as_str}")

    def finish(self):
        pass

    @staticmethod
    def stringify_floats(obj: Any) -> Any:
        if isinstance(obj, float):
            return f"{obj:.4f}"
        elif isinstance(obj, dict):
            return {k: ConsoleLogger.stringify_floats(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ConsoleLogger.stringify_floats(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(ConsoleLogger.stringify_floats(v) for v in obj)
        return obj


class _TensorboardAdapter:
    def __init__(self):
        import os

        from torch.utils.tensorboard import SummaryWriter

        tensorboard_dir = os.environ.get("TENSORBOARD_DIR", "tensorboard_log")
        os.makedirs(tensorboard_dir, exist_ok=True)
        logger.info(f"Saving tensorboard log to {tensorboard_dir}.")
        self.writer = SummaryWriter(tensorboard_dir)

    def log(self, data, step):
        for key in data:
            self.writer.add_scalar(key, data[key], step)

    def finish(self):
        self.writer.close()


class _MlflowLoggingAdapter:
    def __init__(self, project_name, experiment_name, config):
        import os

        import mlflow

        if mlflow.active_run() is None:
            self.we_created_mlflow = True
            if mlflow_tracking_uri := os.environ.get("MLFLOW_TRACKING_URI", None):
                mlflow.set_tracking_uri(mlflow_tracking_uri)

            # Project_name is actually experiment_name in MLFlow
            # If experiment does not exist, will create a new experiment
            experiment = mlflow.set_experiment(project_name)
            mlflow.start_run(experiment_id=experiment.experiment_id, run_name=experiment_name)

        else:
            self.we_created_mlflow = False

        mlflow.log_params(_compute_mlflow_params_from_objects(config))
        self.mlflow = mlflow

    def log(self, data, step):
        results = {k.replace("@", "_at_"): v for k, v in data.items()}
        self.mlflow.log_metrics(metrics=results, step=step)

    def finish(self):
        if self.we_created_mlflow:
            self.mlflow.end_run()


def _compute_mlflow_params_from_objects(params) -> Dict[str, Any]:
    if params is None:
        return {}

    if isinstance(params, DictConfig):
        params = OmegaConf.to_container(params, resolve=True)

    return _flatten_dict(_transform_params_to_json_serializable(params, convert_list_to_dict=True), sep="/")


def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    _transform = partial(_transform_params_to_json_serializable, convert_list_to_dict=convert_list_to_dict)

    if dataclasses.is_dataclass(x):
        return _transform(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {k: _transform(v) for k, v in x.items()}
    if isinstance(x, list):
        if convert_list_to_dict:
            return {"list_len": len(x)} | {f"{i}": _transform(v) for i, v in enumerate(x)}
        else:
            return [_transform(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value

    return x


def _flatten_dict(raw: Dict[str, Any], *, sep: str) -> Dict[str, Any]:
    import pandas as pd

    ans = pd.json_normalize(raw, sep=sep).to_dict(orient="records")[0]
    assert isinstance(ans, dict)
    return ans

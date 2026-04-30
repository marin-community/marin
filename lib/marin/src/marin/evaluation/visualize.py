# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Uses Levanter's viz_lm functionality to visualize log probabilities of a language model.
"""

import dataclasses
import logging
from dataclasses import dataclass

from fray import current_client
from fray.types import Entrypoint, JobRequest, ResourceConfig, TpuConfig, create_environment
from levanter.data.text import LMMixtureDatasetConfig
from levanter.main.viz_logprobs import VizLmConfig as LevanterVizLmConfig
from levanter.main.viz_logprobs import main as viz_lm_main
from levanter.models.lm_model import LmConfig
from levanter.trainer import TrainerConfig

from marin.execution.executor import this_output_path

logger = logging.getLogger(__name__)


@dataclass
class VizLmConfig:
    """
    Configuration for visualizing log probabilities of a language model.
    """

    checkpoint_path: str
    model: LmConfig
    datasets: LMMixtureDatasetConfig
    checkpoint_is_hf: bool = False
    num_docs_per_dataset: int = 32
    per_device_batch_size: int = 4
    output_path: str = dataclasses.field(default_factory=this_output_path)  # type: ignore

    comparison_model_path: str | None = None
    comparison_is_hf: bool = False

    resource_config: ResourceConfig = dataclasses.field(default_factory=lambda: ResourceConfig.with_tpu("v5p-8"))


def do_viz_lm(config: LevanterVizLmConfig) -> None:
    # Levanter can read `gs://` checkpoints directly via fsspec/tensorstore, and HF
    # checkpoints via fsspec as well. Avoid staging large directories locally.
    viz_lm_main(config)


def visualize_lm_log_probs(config: VizLmConfig) -> None:
    levanter_config = LevanterVizLmConfig(
        checkpoint_path=config.checkpoint_path,
        checkpoint_is_hf=config.checkpoint_is_hf,
        model=config.model,
        num_docs=config.num_docs_per_dataset,
        path=config.output_path,
        data=config.datasets,
        trainer=TrainerConfig(per_device_eval_parallelism=config.per_device_batch_size),
        comparison_model_path=config.comparison_model_path,
        comparison_is_hf=config.comparison_is_hf,
    )

    assert isinstance(config.resource_config.device, TpuConfig), "visualize_lm_log_probs requires TPU resource config"

    job_request = JobRequest(
        name="viz-lm-log-probs",
        resources=config.resource_config,
        entrypoint=Entrypoint.from_callable(do_viz_lm, args=[levanter_config]),
        environment=create_environment(extras=["tpu"]),
    )
    client = current_client()
    job = client.submit(job_request)
    job.wait(raise_on_failure=True)

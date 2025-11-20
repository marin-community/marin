# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Uses Levanter's viz_lm functionality to visualize log probabilities of a language model.
"""

import dataclasses
import logging
import multiprocessing
import sys
from dataclasses import dataclass
from queue import Empty

import ray
import tblib
from levanter.data.text import LMMixtureDatasetConfig
from levanter.distributed import RayConfig
from levanter.main.viz_logprobs import VizLmConfig as LevanterVizLmConfig
from levanter.main.viz_logprobs import main as viz_lm_main
from levanter.models.lm_model import LmConfig
from levanter.trainer import TrainerConfig
from levanter.utils.ray_utils import ExceptionInfo

from marin.execution.executor import this_output_path
from marin.utils import remove_tpu_lockfile_on_exit

logger = logging.getLogger(__name__)


def execute_in_subprocess(underlying_function, args, kwargs):
    def target_fn(queue, args, kwargs):
        try:
            # Call the original function
            result = underlying_function(*args, **kwargs)
            queue.put((True, result))  # Success, put the result
        except Exception as e:
            # Capture and return the full traceback in case of an exception
            exc_info = sys.exc_info()
            exception_info = ExceptionInfo(ex=e, tb=tblib.Traceback(exc_info[2]))
            queue.put((False, exception_info))

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=target_fn, args=(queue, args, kwargs))
    process.start()
    process.join()

    # Retrieve the result or error from the queue
    logger.info("Process finished")
    try:
        success, value = queue.get(timeout=1)
    except Empty:
        logger.error("Process timed out")
        process.terminate()
        raise TimeoutError("Process timed out") from None

    if not success:
        value.reraise()
    return value


@dataclass
class VizLmConfig:
    """
    Configuration for visualizing log probabilities of a language model.
    """

    checkpoint_path: str
    model: LmConfig
    datasets: LMMixtureDatasetConfig
    num_docs_per_dataset: int = 32
    per_device_batch_size: int = 4
    output_path: str = dataclasses.field(default_factory=this_output_path)  # type: ignore

    comparison_model_path: str | None = None
    comparison_is_hf: bool = False


@ray.remote(memory=64 * 1024 * 1024 * 1024, resources={"TPU": 4, "TPU-v4-8-head": 1}, max_calls=1)
@remove_tpu_lockfile_on_exit
def do_viz_lm(config: LevanterVizLmConfig) -> None:
    """
    Visualizes log probabilities of a language model.

    Args:
        config (VizLmConfig): The configuration for visualizing log probabilities.
    """
    # remove_tpu_lockfile_on_exit() isn't sufficient now?
    execute_in_subprocess(viz_lm_main, (config,), {})


def visualize_lm_log_probs(config: VizLmConfig) -> None:
    """
    Visualizes log probabilities of a language model.

    Args:
        config (VizLmConfig): The configuration for visualizing log probabilities.
    """
    levanter_config = LevanterVizLmConfig(
        checkpoint_path=config.checkpoint_path,
        model=config.model,
        num_docs=config.num_docs_per_dataset,
        path=config.output_path,
        data=config.datasets,
        trainer=TrainerConfig(
            ray=RayConfig(auto_start_cluster=False), per_device_eval_parallelism=config.per_device_batch_size
        ),
        comparison_model_path=config.comparison_model_path,
        comparison_is_hf=config.comparison_is_hf,
    )
    ray.get(do_viz_lm.remote(levanter_config))

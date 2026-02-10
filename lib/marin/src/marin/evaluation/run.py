# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Script to run an evaluator on a model checkpoint.

Usage:

python3 run.py <Name of evaluator> --model <Path to model or Hugging Face model name> \
--evals <List of evals to run> --output-path <Where to output logs and results>
"""

import logging
import os
import time

import draccus

from fray.cluster import ResourceConfig, TpuConfig, get_tpu_topology
from fray.v1.cluster import ResourceConfig as V1ResourceConfig
from fray.v1.cluster import TpuConfig as V1TpuConfig

from marin.evaluation.evaluation_config import EvaluationConfig
from marin.evaluation.evaluators.evaluator import Evaluator, ModelConfig
from marin.evaluation.evaluators.levanter_lm_eval_evaluator import LevanterLmEvalEvaluator
from marin.evaluation.evaluators.lm_evaluation_harness_evaluator import LMEvaluationHarnessEvaluator
from marin.evaluation.evaluators.simple_evaluator import SimpleEvaluator
from marin.evaluation.utils import discover_hf_checkpoints
from marin.utils import fsspec_exists

logger = logging.getLogger(__name__)

EVALUATORS = {
    "lm_evaluation_harness": LMEvaluationHarnessEvaluator,
    "levanter_lm_evaluation_harness": LevanterLmEvalEvaluator,
    "debug": SimpleEvaluator,
}


def get_evaluator(config: EvaluationConfig) -> Evaluator:
    if config.evaluator not in EVALUATORS:
        raise ValueError(f"Unknown evaluator: {config.evaluator}. Available: {list(EVALUATORS.keys())}")
    return EVALUATORS[config.evaluator]()


def _to_v1_resource_config(config: ResourceConfig) -> V1ResourceConfig:
    """Convert a v2 ResourceConfig to v1 for the evaluation subsystem.

    The key difference is `replicas`: v2 means total VMs (slice_count * vm_count),
    v1 means slice_count. For TPU configs we divide back out; for non-TPU configs
    the value passes through unchanged.
    """
    replicas = config.replicas
    device = config.device
    if isinstance(device, TpuConfig):
        try:
            topo = get_tpu_topology(device.variant)
            replicas = max(1, config.replicas // topo.vm_count)
        except ValueError:
            pass
        device = V1TpuConfig(variant=device.variant, topology=device.topology)

    return V1ResourceConfig(
        cpu=config.cpu,
        ram=config.ram,
        disk=config.disk,
        device=device,
        replicas=replicas,
        preemptible=config.preemptible,
        regions=config.regions,
    )


def evaluate(config: EvaluationConfig) -> None:
    logger.info(f"Running evals with args: {config}")
    evaluator: Evaluator = get_evaluator(config)

    model: ModelConfig = _impute_model_config(config)
    logger.info(f"Evaluating {model.name} with {config.evals}")

    start_time: float = time.time()
    if config.launch_with_ray:
        v1_resources = _to_v1_resource_config(config.resource_config)
        evaluator.launch_evaluate_with_ray(
            model,
            evals=config.evals,
            output_path=config.evaluation_path,
            max_eval_instances=config.max_eval_instances,
            resource_config=v1_resources,
        )
    else:
        evaluator.evaluate(
            model,
            evals=config.evals,
            output_path=config.evaluation_path,
            max_eval_instances=config.max_eval_instances,
        )

    logger.info(f"Done (total time: {time.time() - start_time} seconds)")


def _impute_model_config(config):
    if config.model_path is None:
        raise ValueError("model_name or model_path must be provided")

    model_path = _normalize_model_path(config.model_path)

    if config.discover_latest_checkpoint:
        model_path = discover_hf_checkpoints(model_path)[-1]

    if config.model_name is None:
        if model_path.endswith("/"):
            model_path = model_path[:-1]

        last_component = model_path.split("/")[-1]
        if not last_component.startswith("step-"):
            model_name = last_component
        else:
            # Have to impute the model name from the path.
            model_name_parts = model_path.split("/")
            # We're looking for something that looks like a run name and something that looks like a step
            # e.g. $RUN/hf/step-$STEP.
            step_part = model_name_parts[-1]
            step_part = step_part.split("-")[1]

            # Don't assume there's an hf. Look for a run name, which probably has a - in it.
            for part in reversed(model_name_parts[:-1]):
                if "-" in part:
                    model_name = part
                    break
            else:
                # just use the penultimate part
                model_name = model_name_parts[-2]

            model_name = f"{model_name}-{step_part}"
    else:
        model_name = config.model_name
    generation_params = {}
    engine_kwargs = {}
    if config.generation_params is None:
        logger.warning(f"No generation params provided for {model_name}, using default params")
    else:
        generation_params = config.generation_params
    if config.engine_kwargs is None:
        logger.warning(f"No engine kwargs provided for {model_name}, using default params")
    else:
        engine_kwargs = config.engine_kwargs

    return ModelConfig(
        name=model_name,
        path=model_path,
        engine_kwargs=engine_kwargs,
        generation_params=generation_params,
        apply_chat_template=config.apply_chat_template,
    )


def _normalize_model_path(model_path: str) -> str:
    """
    Choose the HF checkpoint root when callers pass either `<path>` or `<path>/hf`.
    """
    model_path = model_path.rstrip("/")

    def has_hf_files(path: str) -> bool:
        return fsspec_exists(os.path.join(path, "config.json"))

    if has_hf_files(model_path):
        return model_path

    if model_path.endswith("/hf"):
        parent_path = model_path[: -len("/hf")]
        if parent_path and has_hf_files(parent_path):
            return parent_path

    hf_candidate = os.path.join(model_path, "hf")
    if has_hf_files(hf_candidate):
        return hf_candidate

    return model_path


@draccus.wrap()
def main(config: EvaluationConfig) -> None:
    evaluate(config)


if __name__ == "__main__":
    main()

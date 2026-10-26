# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a Table 9 eval step in place, outside the executor parent that owns it.

For a step whose own child is stuck waiting on its TPU type: rebuild the eval config from the step's
.executor_info with the same olmo_base_eval_step arguments, resolve the output placeholders to the step's
own directory, and score there on whatever TPU this job was given. The evaluator persists each task as it
is scored, so the step's child, whenever it runs, loads the finished tasks instead of rescoring them.

usage: python -m experiments.domain_phase_mix.run_table9_eval_in_place <step output dir> [--check]
"""

import argparse
import json
import logging

import fsspec
import jax.numpy as jnp
from fray.cluster import ResourceConfig
from marin.evaluation.olmo_base_eval.run import OlmoBaseEvalConfig, olmo_base_eval, olmo_base_eval_step
from marin.execution.executor import resolve_local_placeholders

logger = logging.getLogger(__name__)

# olmo_base_eval_step requires a resource config, but only the pod launcher reads it; this process is
# already running on the TPU it was submitted to.
UNUSED_RESOURCES = ResourceConfig.with_tpu("v5p-8")


def eval_config_from_step(step_dir: str) -> OlmoBaseEvalConfig:
    with fsspec.open(f"{step_dir}/.executor_info") as handle:
        info = json.load(handle)
    recorded = info["config"]["eval_config"]
    wandb = recorded["trainer"]["tracker"][0]
    step = olmo_base_eval_step(
        name=recorded["name"],
        checkpoint=recorded["checkpoint_path"],
        request_set_dir=recorded["request_set_dir"],
        resource_config=UNUSED_RESOURCES,
        checkpoint_is_hf=recorded["checkpoint_is_hf"],
        tokenizer=recorded["tokenizer"],
        per_device_batch_size=recorded["trainer"]["per_device_eval_parallelism"],
        max_eval_length=recorded["max_eval_length"],
        wandb_project=wandb["project"],
        wandb_tags=wandb["tags"],
        wandb_group=wandb["group"],
        provenance=recorded["provenance"],
    )
    config = resolve_local_placeholders(step.config.eval_config, info["output_path"])

    rebuilt = {
        "output_path": config.output_path,
        "json_tracker_output_path": config.trainer.tracker[1].output_path,
        "compute_dtype": jnp.dtype(config.trainer.mp.compute_dtype).name,
        "param_dtype": jnp.dtype(config.trainer.mp.param_dtype).name,
    }
    expected = {
        "output_path": recorded["output_path"],
        "json_tracker_output_path": recorded["trainer"]["tracker"][1]["output_path"],
        "compute_dtype": recorded["trainer"]["mp"]["compute_dtype"],
        "param_dtype": recorded["trainer"]["mp"]["param_dtype"],
    }
    if rebuilt != expected:
        raise ValueError(f"rebuilt config differs from the recorded step: {rebuilt} != {expected}")
    return config


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step_dir", help="the eval step's output directory, holding its .executor_info")
    parser.add_argument("--check", action="store_true", help="rebuild and verify the config without scoring")
    args = parser.parse_args()
    config = eval_config_from_step(args.step_dir)
    logger.info("rebuilt %s -> %s (checkpoint %s)", config.name, config.output_path, config.checkpoint_path)
    if args.check:
        return
    olmo_base_eval(config)


if __name__ == "__main__":
    main()

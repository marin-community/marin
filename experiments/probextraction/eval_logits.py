import os
import sys
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import ray


from levanter.main.eval_sliding_total import EvalSlidingTotalConfig, main as eval_sliding_main
import levanter.config


@dataclass
class EvalSlidingConfig:
    """Configuration for Levanter eval_sliding_total via direct function call."""
    # Path to the Levanter YAML config for sliding-window evaluation.
    config_path: str
    # Whether Ray should auto-start a cluster (disable by default).
    trainer_ray_auto_start_cluster: bool = False
    # Base output path prefix (will be passed as output_base_path).
    output_base_path: str | None = None


@ray.remote(memory=64 * 1024 * 1024 * 1024, resources={"TPU": 4, "TPU-v4-8-head": 1}, max_calls=1)
def do_eval_sliding_total(config_path: str, output_base_path: str | None, trainer_ray_auto_start_cluster: bool) -> None:
    """Run Levanter's eval_sliding_total directly via function call."""
    
    # Load config from YAML and override specific fields
    config_dict = levanter.config.load_config_from_file(config_path)
    
    # Override trainer.ray.auto_start_cluster
    if "trainer" not in config_dict:
        config_dict["trainer"] = {}
    if "ray" not in config_dict["trainer"]:
        config_dict["trainer"]["ray"] = {}
    config_dict["trainer"]["ray"]["auto_start_cluster"] = trainer_ray_auto_start_cluster
    
    # Override output_base_path if provided
    if output_base_path is not None:
        config_dict["output_base_path"] = output_base_path
    
    # Create config object from dict
    config = levanter.config.config_from_dict(EvalSlidingTotalConfig, config_dict)
    
    # Run evaluation
    eval_sliding_main(config)


def eval_sliding_total(cfg: EvalSlidingConfig) -> None:
    """Run Levanter's eval_sliding_total via Ray remote function.
    
    Note: This function returns immediately. The executor framework
    will handle calling ray.get() on the returned ObjectRef.
    """
    # Return ObjectRef - let executor framework handle ray.get()
    return do_eval_sliding_total.remote(
        cfg.config_path,
        cfg.output_base_path, 
        cfg.trainer_ray_auto_start_cluster
    )

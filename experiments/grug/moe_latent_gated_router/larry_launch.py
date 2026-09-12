# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Route from gated latent features using the recorded TPU recipe from issue #6822."""

import copy
import json
from pathlib import Path

import click
import draccus
from fray.cluster import ResourceConfig
from levanter.data.text.datasets import LmDataConfig
from levanter.optim.muonh import MuonHConfig
from levanter.trainer import TrainerConfig

from experiments.grug.moe_latent_gated_router.larry_model import GrugModelConfig
from experiments.grug.moe_latent_gated_router.larry_train import (
    GrugEvalConfig,
    GrugRunConfig,
    GrugTrainerConfig,
    run_grug,
)


def build_config(dim: int, run_id: str) -> GrugRunConfig:
    """Replay the reference recipe with a new router input and fresh run identity."""
    reference = json.loads(Path(__file__).with_name(f"larry_reference_d{dim}.json").read_text())
    model = draccus.decode(GrugModelConfig, reference["model"] | {"router_input": "gated_latent"})
    trainer = copy.deepcopy(reference["trainer"]["trainer"])
    trainer["id"] = run_id
    trainer["mp"] = "params=float32,compute=bfloat16,output=bfloat16"
    trainer["tracker"].update(type="wandb", name=run_id, id=run_id, group="moe-lgr-9110-larry-router")
    trainer["tracker"]["tags"] += ["issue-9110", "gated-latent-router"]
    checkpointer = trainer["checkpointer"]
    checkpointer["base_path"] = f"gs://marin-us-east5/users/kaiyuew/grug/{run_id}/checkpoints"
    checkpointer["temporary_base_path"] = f"gs://marin-us-east5/tmp/ttl=14d/users/kaiyuew/{run_id}/checkpoints"
    return GrugRunConfig(
        model=model,
        data=draccus.decode(LmDataConfig, reference["data"]),
        optimizer=draccus.decode(MuonHConfig, reference["optimizer"]),
        resources=ResourceConfig.with_tpu("v5p-8", cpu=32, ram="128g", disk="50g", regions=["us-east5"]),
        trainer=GrugTrainerConfig(**(reference["trainer"] | {"trainer": draccus.decode(TrainerConfig, trainer)})),
        eval=draccus.decode(GrugEvalConfig, reference["eval"]),
    )


@click.command()
@click.option("--dim", type=click.Choice(["512", "768"]), required=True)
@click.option("--run-id", required=True)
@click.option("--run", is_flag=True)
def main(dim: str, run_id: str, run: bool):
    config = build_config(int(dim), run_id)
    if run:
        run_grug(config)
    else:
        print(draccus.dump(config))


if __name__ == "__main__":
    main()

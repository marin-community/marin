# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tiny local-GPU grug run using the tutorial Wikitext pipeline."""

from fray.cluster import ResourceConfig
from levanter.data.text import TextLmDatasetFormat
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import lm_data_config

from experiments.defaults import default_tokenize
from experiments.grug.base.launch import GrugBaseLaunchConfig, run_grug_base_trial
from experiments.grug.base.model import GrugModelConfig
from experiments.grug.base.train import GrugEvalConfig, GrugTrainerConfig
from experiments.marin_models import marin_tokenizer

wikitext_hf_id = "dlwh/wikitext_2_detokenized"

wikitext_tokenized = default_tokenize(
    name=wikitext_hf_id,
    dataset=wikitext_hf_id,
    tokenizer=marin_tokenizer,
    format=TextLmDatasetFormat(),
    sample_count=versioned(1000),
)

TINY_GRUG_GPU_MODEL = GrugModelConfig(
    vocab_size=128_256,
    hidden_dim=256,
    intermediate_dim=768,
    num_layers=2,
    num_heads=4,
    num_kv_heads=4,
    max_seq_len=256,
    head_dim=None,
)


tiny_grug_gpu_trial = ExecutorStep(
    name="grug/tiny-gpu-trial",
    fn=run_grug_base_trial,
    config=GrugBaseLaunchConfig(
        model=versioned(TINY_GRUG_GPU_MODEL),
        data=lm_data_config(wikitext_tokenized),
        output_path=this_output_path(),
        run_id="grug-tiny-gpu-trial",
        resources=versioned(ResourceConfig.with_gpu("H100", count=1, cpu=8, disk="64G", ram="32G")),
        steps=versioned(100),
        batch_size=versioned(32),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "tutorial", "gpu", "tiny"],
            group="grug-tiny-gpu-trial",
            name=None,
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(
            AdamConfig(
                learning_rate=3e-4,
                weight_decay=0.1,
                lr_schedule="cosine",
                warmup=10,
                min_lr_ratio=0.1,
            )
        ),
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=1e-4,
                log_every=1,
                log_activation_grad_rms=True,
            )
        ),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=32,
                steps_per_eval=50,
                max_eval_batches=2,
                eval_current=True,
                eval_ema=False,
            )
        ),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[
            wikitext_tokenized,
            tiny_grug_gpu_trial,
        ],
        description="Tiny grug GPU trial on tutorial Wikitext data with activation-grad RMS logging.",
    )

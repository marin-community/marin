"""
#1331: Marin SWE-Agent

GitHub Issue: https://github.com/marin-community/marin/issues/1331
"""

import dataclasses

from levanter.data.text import ChatLmDatasetFormat
from levanter.layers.rotary import YarnRotaryEmbeddingsConfig

from experiments.defaults import default_sft, default_tokenize
from experiments.llama import llama_8b
from experiments.tootsie.exp916_tootsie_spoonbill_cooldown import spoonbill_zloss_tulu3_sft_config
from marin.download.huggingface.download import DownloadConfig, download
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.resources import TpuPodConfig

sft_experiments = []
swe_smith_sft_config = dataclasses.replace(
    spoonbill_zloss_tulu3_sft_config,
    learning_rate=1e-4,
    num_train_steps=600,
    max_seq_len=32768,
    train_batch_size=16,
    resources=TpuPodConfig(tpu_type="v5p-32", slice_count=1),
    model_name_or_path="marin-community/marin-8b-base",
)

swe_smith_trajectories = ExecutorStep(
    name="raw/SWE-bench/SWE-smith-trajectories",
    fn=download,
    config=DownloadConfig(
        hf_dataset_id="SWE-bench/SWE-smith-trajectories",
        revision=versioned("f6b6d7e"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.parquet"],
    ),
    override_output_path="raw/SWE-bench/SWE-smith-trajectories",
).cd("f6b6d7e/huggingface.co/datasets/SWE-bench/SWE-smith-trajectories/resolve/f6b6d7e")


tokenized_swe_smith_trajectories = default_tokenize(
    "tokenized/SWE-smith-trajectories",
    swe_smith_trajectories,
    "stanford-crfm/marin-tokenizer",
    format=ChatLmDatasetFormat(),
)

yarn_llama_8b = dataclasses.replace(
    llama_8b,
    rope=YarnRotaryEmbeddingsConfig(original_max_position_embeddings=4096),
    seq_len=32768,
)

no_yarn_llama_8b = dataclasses.replace(
    llama_8b,
    seq_len=32768,
)


marin_8b_swe_smith_sft = default_sft(
    name="sft/marin-swe-smith-8b-yarn-3epoch",
    tokenized=tokenized_swe_smith_trajectories,
    model_config=yarn_llama_8b,
    sft_config=swe_smith_sft_config,
    tags=["marin-8b", "sft", "agent"],
).with_output_path("checkpoints/sft/marin-swe-smith-8b-yarn-3epoch")

marin_8b_swe_smith_sft_no_yarn = default_sft(
    name="sft/marin-swe-smith-8b-no-yarn-3epoch",
    tokenized=tokenized_swe_smith_trajectories,
    model_config=no_yarn_llama_8b,
    sft_config=swe_smith_sft_config,
    tags=["marin-8b", "sft", "agent"],
).with_output_path("checkpoints/sft/marin-swe-smith-8b-no-yarn-3epoch")

if __name__ == "__main__":
    executor_main(
        [marin_8b_swe_smith_sft, marin_8b_swe_smith_sft_no_yarn],
        description="SWE Agent SFT for Marin 8B",
    )

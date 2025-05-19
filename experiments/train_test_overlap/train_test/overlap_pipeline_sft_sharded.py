"""
Sharded n-gram overlap pipeline for all SFT instruction datasets from the mixture config.

This script uses get_instruction_dataset to fetch and transform each dataset,
then tokenizes via Marin tokenizer, computes n-gram overlaps in a sharded
manner, and consolidates results.
"""

import logging

from experiments.exp808_sft_mixture import DATASETS
from experiments.instruction_datasets import get_instruction_dataset
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, output_path_of
from train_test_overlap.run_overlap_shards import ShardedOverlapConfig, run_all_shards
from experiments.train_test_overlap.train_test.consolidate_sharded_pipeline import (
    ConsolidateShardedConfig,
    consolidate_sharded,
)

# Add imports for Dolma conversion
from operations.transform.conversation.conversation_to_dolma import (
    ConversationToDolmaConfig,
    convert_conversation_to_dolma,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 1) Fetch & convert SFT datasets into Dolma format
# -------------------------------------------------------------
dolma_datasets = {}
for short_name, hf_name in DATASETS.items():
    # Download & transform instruction dataset into document shards
    dataset_step = get_instruction_dataset(hf_name)
    # Convert to Dolma format
    dolma_step = ExecutorStep(
        name=f"dolma/{short_name}",
        fn=convert_conversation_to_dolma,
        config=ConversationToDolmaConfig(output_path_of(dataset_step)),
    )
    dolma_datasets[short_name] = dolma_step

# 2) Define overlap settings
# --------------------------
# TODO: set this to your evaluation scenarios JSONL path
scenario_data = "gs://marin-us-central2/scenarios/consolidated_eval_scenarios-d3f040/consolidated_scenarios.jsonl"
n_values = [10, 15]

# 3) Create sharded overlap & consolidation steps
# -----------------------------------------------
overlap_steps = []
for idx, (short_name, dolma_step) in enumerate(dolma_datasets.items()):
    if idx > 0:
        continue
    # Sharded overlap computation
    overlap_config = ShardedOverlapConfig(
        base_input_dir=output_path_of(dolma_step),
        scenario_data=scenario_data,
        output_base=this_output_path(),
        N=n_values,
        max_in_flight=512,
    )
    overlap_step = ExecutorStep(
        name=f"train_test_overlap/ngrams/{short_name}",
        fn=run_all_shards,
        config=overlap_config,
    )
    overlap_steps.append(overlap_step)

    # Consolidate shard outputs
    cons_config = ConsolidateShardedConfig(
        input_step=overlap_step,
        output_path=this_output_path(),
    )
    cons_step = ExecutorStep(
        name=f"train_test_overlap/consolidated/{short_name}",
        fn=lambda cfg: consolidate_sharded(cfg),
        config=cons_config,
    )
    overlap_steps.append(cons_step)

if __name__ == "__main__":
    executor_main(
        steps=overlap_steps,
        description="Sharded n-gram overlap pipeline for SFT instruction datasets",
    ) 
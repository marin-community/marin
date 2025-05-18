import logging

from experiments.train_test_overlap.format.convert_starcoder_parquet2jsonl import starcoder_to_jsonl
from experiments.train_test_overlap.train_test.consolidate_sharded_pipeline import (
    ConsolidateShardedConfig,
    consolidate_sharded,
)
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from train_test_overlap.run_overlap_shards import ShardedOverlapConfig, run_all_shards

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

base_input_dir = output_path_of(starcoder_to_jsonl)
scenario_data = "gs://marin-us-central2/scenarios/consolidated_eval_scenarios-d3f040/consolidated_scenarios.jsonl"
# Build sharded runner config with backpressure
config = ShardedOverlapConfig(
    base_input_dir=base_input_dir,
    scenario_data=scenario_data,
    output_base=this_output_path(),
    N=[5, 10, 15],
    max_in_flight=1024,
)
starcoder_sharded_step = ExecutorStep(
    name="train_test_overlap/ngrams/starcoder_data_overlap_sharded",
    fn=run_all_shards,
    config=config,
)

consolidate_sharded_config = ConsolidateShardedConfig(
    input_step=starcoder_sharded_step,
    output_path=this_output_path(),
)
consolidate_sharded_step = ExecutorStep(
    name="train_test_overlap/consolidated/consolidate_sharded_starcoder",
    fn=lambda cfg: consolidate_sharded(cfg),
    config=consolidate_sharded_config,
)

if __name__ == "__main__":
    executor_main(
        steps=[starcoder_sharded_step, consolidate_sharded_step],
        description="Run and then consolidate sharded n-gram overlap pipeline on StarCoder data",
    )

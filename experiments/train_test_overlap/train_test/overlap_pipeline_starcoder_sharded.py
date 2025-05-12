import logging

from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from train_test_overlap.run_overlap_shards import ShardedOverlapConfig, run_all_shards

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    base_input_dir = "gs://marin-us-central2/documents/starcoderdata-720c8c/9fc30b5/"
    scenario_data = "gs://marin-us-central2/scenarios/consolidated_eval_scenarios-d3f040/consolidated_scenarios.jsonl"
    # Build sharded runner config with backpressure
    config = ShardedOverlapConfig(
        base_input_dir=base_input_dir,
        scenario_data=scenario_data,
        output_base=this_output_path(),
        N=[5, 10, 15, 20],
        processes=4,
        max_in_flight=1,
    )
    step = ExecutorStep(
        name="train_test_overlap/starcoder_data_overlap_sharded",
        fn=run_all_shards,
        config=config,
    )
    executor_main(
        steps=[step],
        description="Run sharded n-gram data overlap pipeline on StarCoder data",
    )

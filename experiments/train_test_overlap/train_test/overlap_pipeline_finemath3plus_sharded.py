import logging
import os

from experiments.train_test_overlap.format.convert_finemath3plus_parquet2jsonl import finemath3plus_to_jsonl
from marin.execution.executor import ExecutorStep, executor_main, get_executor_step, this_output_path
from train_test_overlap.run_overlap_shards import ShardedOverlapConfig, run_all_shards

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Extract the conversion step and ensure MARIN_PREFIX is set
conv_step = get_executor_step(finemath3plus_to_jsonl)
marin_prefix = os.environ.get("MARIN_PREFIX")
if not marin_prefix:
    raise ValueError("MARIN_PREFIX environment variable not set. Please set it to your GCS prefix.")

# Resolve where the JSONL conversion step wrote its shards
base_input_dir = os.path.join(
    marin_prefix,
    conv_step.override_output_path,
)
logger.info(f"Using base input dir for Finemath-3plus JSONL shards: {base_input_dir}")

# Scenario data and n-gram settings
scenario_data = "gs://marin-us-central2/scenarios/consolidated_eval_scenarios-d3f040/consolidated_scenarios.jsonl"
n_values = [5, 10, 15, 20]

# Configure the sharded runner with backpressure
config = ShardedOverlapConfig(
    base_input_dir=base_input_dir,
    scenario_data=scenario_data,
    output_base=this_output_path(),
    N=n_values,
    max_in_flight=1024,
)
f3p_sharded_step = ExecutorStep(
    name="train_test_overlap/finemath-3plus_data_overlap_sharded",
    fn=run_all_shards,
    config=config,
)

if __name__ == "__main__":
    executor_main(
        steps=[f3p_sharded_step],
        description="Run sharded n-gram data overlap pipeline on Finemath-3plus JSONL shards",
    )

import logging
import os

from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from train_test_overlap.run_overlap_shards import ShardedOverlapConfig, run_all_shards

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Validate prefix
gs_prefix = os.environ.get("MARIN_PREFIX")
if not gs_prefix:
    raise ValueError("MARIN_PREFIX environment variable not set. Please set it to your GCS prefix.")

# Base input directory for Nemotron-CC shards (override path + data-jsonl location)
base_input_dir = os.path.join(
    gs_prefix,
    "raw/nemotro-cc-eeb783",
    "contrib/Nemotron/Nemotron-CC/data-jsonl",
)
logger.info(f"Using base input dir for Nemotron-CC shards: {base_input_dir}")

# Scenario data and parameters
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
nemotron_sharded_step = ExecutorStep(
    name="train_test_overlap/nemotron_cc_data_overlap_sharded",
    fn=run_all_shards,
    config=config,
)

if __name__ == "__main__":
    executor_main(
        steps=[nemotron_sharded_step],
        description="Run sharded n-gram data overlap pipeline on Nemotron-CC dataset",
    )

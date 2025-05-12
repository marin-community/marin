import logging
import os

from experiments.pretraining_datasets import dclm_baseline  # InputName for DCLM download step
from marin.execution.executor import ExecutorStep, executor_main, get_executor_step, this_output_path
from train_test_overlap.run_overlap_shards import ShardedOverlapConfig, run_all_shards

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Resolve where the DCLM baseline step placed the shards
    marin_prefix = os.environ.get("MARIN_PREFIX")
    if not marin_prefix:
        raise ValueError("MARIN_PREFIX environment variable not set. Please set it to your GCS prefix.")
    dclm_producing_step = get_executor_step(dclm_baseline)
    # Base path: prefix / override_output_path / revision
    resolved_base = os.path.join(
        marin_prefix,
        dclm_producing_step.override_output_path,
        dclm_baseline.name,
    )
    # Under that, HuggingFace structure:
    base_input_dir = os.path.join(
        resolved_base,
        "huggingface.co/datasets",
        dclm_producing_step.config.hf_dataset_id,
        "resolve",
        dclm_producing_step.config.revision,
    )
    logger.info(f"Using base input dir for DCLM shards: {base_input_dir}")

    # Scenario data and parameters
    scenario_data = "gs://marin-us-central2/scenarios/consolidated_eval_scenarios-d3f040/consolidated_scenarios.jsonl"
    n_values = [5, 10, 15, 20]
    processes = 4

    # Configure the sharded runner with backpressure (one at a time by default)
    config = ShardedOverlapConfig(
        base_input_dir=base_input_dir,
        scenario_data=scenario_data,
        output_base=this_output_path(),
        N=n_values,
        processes=processes,
        max_in_flight=256,
    )
    step = ExecutorStep(
        name="train_test_overlap/dclm_data_overlap_sharded",
        fn=run_all_shards,
        config=config,
    )

    executor_main(
        steps=[step],
        description="Run sharded n-gram data overlap pipeline on DCLM dataset",
    )

import logging

from experiments.train_test_overlap.format.medicalqa_parquet2jsonl import (
    lavita_all_processed_to_jsonl,
    lavita_medmcqa_to_jsonl,
    lavita_pubmed_to_jsonl,
)
from experiments.train_test_overlap.train_test.aggregate_test_overlap import (
    AggregateTestOverlapConfig,
    aggregate_test_overlap,
)
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from train_test_overlap.run_overlap_shards import ShardedOverlapConfig, run_all_shards

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

scenario_data = "gs://marin-us-central2/scenarios/consolidated_eval_scenarios_final-50b720/consolidated_scenarios.jsonl"
# Build sharded runner config with backpressure
# config = ShardedOverlapConfig(
#     base_input_dir=lavita_pubmed_to_jsonl,
#     scenario_data=scenario_data,
#     output_base=this_output_path(),
#     N=[10, 15],
#     max_in_flight=2048,
# )

medical_qa_datasets = [
    ("lavita_pubmed", lavita_pubmed_to_jsonl),
    ("lavita_allprocessed", lavita_all_processed_to_jsonl),
    ("lavita_medmcqa", lavita_medmcqa_to_jsonl),
]

medical_qa_sharded_steps = []
for dataset_name, medical_qa_dataset in medical_qa_datasets:
    config = ShardedOverlapConfig(
        base_input_dir=output_path_of(medical_qa_dataset),
        scenario_data=scenario_data,
        output_base=this_output_path(),
        N=[10, 15],
        max_in_flight=2048,
    )

    medicalqa_sharded_step = ExecutorStep(
        name=f"train_test_overlap/ngrams_final/medicalqa_data_overlap_sharded_{dataset_name}",
        fn=run_all_shards,
        config=config,
    )
    medical_qa_sharded_steps.append(medicalqa_sharded_step)

config = AggregateTestOverlapConfig(
    input_paths=[output_path_of(step) for step in medical_qa_sharded_steps],
    output_base=this_output_path(),
    scenario_jsonl="gs://marin-us-central2/scenarios/consolidated_eval_scenarios_final-50b720/consolidated_scenarios.jsonl",
    n_values=[10, 15],
)
aggregate_step = ExecutorStep(
    name="train_test_overlap/aggregated_medicalqa",
    fn=aggregate_test_overlap,
    config=config,
)

if __name__ == "__main__":
    executor_main(
        steps=[aggregate_step],
        description="Run sharded n-gram overlap pipeline on MedicalQA data",
    )

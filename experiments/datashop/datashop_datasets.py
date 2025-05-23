from experiments.pretraining_datasets import dclm_baseline
from marin.classifiers.utils import CreateDatasetConfig, create_dataset
from marin.download.filesystem.transfer import TransferConfig, transfer_files
from marin.execution.executor import ExecutorStep, executor_main, this_output_path

# Serves as the default pretraining dataset used for Datashop experiments. We want roughly 400B tokens
# that we can use for training because we know that the quality filter will filter about 10% of the top
# tokens since 3+ is roughly top 10% most of the time.
dclm_baseline_global_shard_2 = dclm_baseline.cd(
    "huggingface.co/datasets/mlfoundations/dclm-baseline-1.0/resolve/a3b142c/global-shard_02_of_10"
)

# Around 40B tokens which serves as the default seed annotation dataset for Datashop experiments. We take the top 4
# files which amounts to about 350K examples.
dclm_baseline_global_shard_1_local_shard_1 = dclm_baseline.cd(
    "huggingface.co/datasets/mlfoundations/dclm-baseline-1.0/resolve/a3b142c/global-shard_01_of_10/local-shard_0_of_10"
)

datashop_dclm_annotation_subset = ExecutorStep(
    name="documents/datashop-datasets/datashop-dclm-annotation-subset",
    fn=transfer_files,
    config=TransferConfig(
        input_path=dclm_baseline_global_shard_1_local_shard_1,
        output_path=this_output_path(),
        num_random_files=4,
    ),
)

datashop_dclm_pretraining_subset = ExecutorStep(
    name="documents/datashop-datasets/datashop-dclm-pretraining-subset",
    fn=transfer_files,
    config=TransferConfig(
        input_path=dclm_baseline_global_shard_2,
        output_path=this_output_path(),
    ),
)

datashop_dclm_tutorial_pretraining_subset = ExecutorStep(
    name="documents/datashop-datasets/datashop-dclm-tutorial-pretraining-subset",
    fn=transfer_files,
    config=TransferConfig(
        input_path=dclm_baseline_global_shard_1_local_shard_1,
        output_path=this_output_path(),
        num_random_files=1,
    ),
)

datashop_dclm_tutorial_annotation_subset = ExecutorStep(
    name="documents/datashop-datasets/datashop-dclm-tutorial-annotation-subset",
    fn=create_dataset,
    config=CreateDatasetConfig(
        input_doc_path=datashop_dclm_tutorial_pretraining_subset,
        output_dataset_path=this_output_path(),
        max_sample_size=1_000,
        filetype="jsonl.zst",
        merge_dataset_shards=False,
        columns_to_keep=["text", "metadata"],
    ),
)

if __name__ == "__main__":
    executor_main([datashop_dclm_annotation_subset, datashop_dclm_pretraining_subset])

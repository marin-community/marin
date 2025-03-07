from experiments.test_infer_func import test_infer
from marin.classifiers.hf.train_classifier import HFTrainingConfig
from marin.execution.executor import ExecutorStep, executor_main, output_path_of

dataset = ExecutorStep(name="scratch/test_infer_dataset", fn=lambda x: None, config=None)
exec_step = ExecutorStep(
    name="scratch/test_infer",
    fn=test_infer,
    config=HFTrainingConfig(
        train_dataset=output_path_of(dataset),
        output_dir="",
        num_labels=1,
        target_column="label",
        tpu_num_cores=8,
    ),
)

# dataset = ExecutorStep(
#     name=f"documents/medu-datasets/{experiment_name}",
#     fn=run_medu_dataset_sampling_pipeline,
#     config=DatasetOutputProcessorConfig(
#         input_path=output_path_of(labeled_documents),
#         output_path=this_output_path(),
#     ),
# ).cd("sampled")

# medu_classifier_remote = ExecutorStep(
#     name=f"classifiers/medu-bert/{experiment_name}",
#     fn=train_classifier_distributed,
#     config=HFTrainingConfig(
#         train_dataset=dataset,
#         output_dir=this_output_path(),
#         num_labels=1,
#         target_column="label",
#         tpu_num_cores=8,
#         max_length=512,
#         train_size=0.9,
#         eval_steps=100,
#         save_steps=100,
#         logging_steps=10,
#     ),
# )

if __name__ == "__main__":
    # ray.get(waiter.remote())
    executor_main([exec_step])

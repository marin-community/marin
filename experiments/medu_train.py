import ray

from marin.classifiers.bert.train_classifier import (
    HFTrainingConfig,
    train_classifier_distributed,
)

# from marin.classifiers.bert.basics import test_distribution

# from experiments.medu_inference import medu_inference


# convert_to_classifier_dataset_format = ExecutorStep(
#     name="documents/fineweb-economics-llama-70b-annotations",
#     fn=convert_classifier_dataset_func,
#     config=ConvertClassifierDatasetConfig(
#         input_path=output_path_of(medu_inference),
#         output_path=this_output_path(),
#     ),
# )

# count_num_labels = ExecutorStep(
#     name="documents/fineweb-economics-llama-70b-annotations-num-labels",
#     fn=count_num_labels_func,
#     config=CountNumLabelsConfig(
#         input_path=output_path_of(convert_to_classifier_dataset_format),
#         input_filetype="jsonl.gz",
#         output_path=this_output_path(),
#     ),
# )
# # Result of this was: data: {1: 528666, 2: 351631, 3: 76157, 4: 45853, 5: 1596}

# sample_classifier_dataset = ExecutorStep(
#     name="documents/fineweb-economics-llama-70b-annotations-sampled",
#     fn=sample_classifier_dataset_func,
#     config=SampleClassifierDatasetConfig(
#         input_path=output_path_of(convert_to_classifier_dataset_format),
#         output_path=this_output_path(),
#         label_weights={"1": 61803/528666, "2": 61803/351631, "3": 1, "4": 1, "5": 1},
#     ),
# )

# count_num_labels_sampled = ExecutorStep(
#     name="documents/fineweb-economics-llama-70b-annotations-sampled-num-labels",
#     fn=count_num_labels_func,
#     config=CountNumLabelsConfig(
#         input_path=output_path_of(sample_classifier_dataset),
#         input_filetype="jsonl.gz",
#         output_path=this_output_path(),
#     ),
# )


NUM_TPU_DEVICES = 8
TPU_TYPE = "v6e-8"


# NOTE(chris): BTW NEED PIP_DEPS accelerate>=0.26.0 set in the commandline
# @ray.remote(resources={"TPU": NUM_TPU_DEVICES, f"TPU-{TPU_TYPE}-head": 1}, runtime_env={"pip": ["accelerate>=0.26.0"]})
@ray.remote(resources={"TPU": NUM_TPU_DEVICES, f"TPU-{TPU_TYPE}-head": 1})
def train_classifier_tpu():
    import os

    output_dir = "/opt/gcsfuse_mount/economic-bert-large-8"
    os.makedirs(output_dir, exist_ok=True)

    train_classifier_distributed(
        # rank=0,
        # args=ScriptArguments(
        #     train_dataset="gs://marin-us-east5/documents/fineweb-economics-llama-70b-annotations-sampled-ee7616",
        #     num_labels=1,
        #     target_column="label",
        #     # TODO(chris): change later
        #     output_dir=output_dir,
        #     remove_unused_columns=False,
        #     per_device_train_batch_size=8,
        #     gradient_accumulation_steps=16,
        #     report_to="wandb",
        #     logging_steps=1,
        #     max_length=512,  # TODO(CHRIS): Can change later
        #     tpu_num_cores=NUM_TPU_DEVICES,
        #     eval_steps=1000,
        #     eval_strategy="steps",
        #     save_strategy="steps",
        #     save_steps=1000,
        #     load_best_model_at_end=True,
        #     metric_for_best_model="f1_macro",
        #     greater_is_better=True,
        # ),
        args=HFTrainingConfig(
            train_dataset="gs://marin-us-east5/documents/fineweb-economics-llama-70b-annotations-sampled-ee7616",
            num_labels=1,
            target_column="label",
            output_dir=output_dir,
            tpu_num_cores=NUM_TPU_DEVICES,
            max_length=512,
        ),
    )


if __name__ == "__main__":
    # executor_main(steps=[convert_to_classifier_dataset_format])
    x = ray.get(train_classifier_tpu.remote())
    # executor_main(steps=[convert_to_classifier_dataset_format, count_num_labels,
    # sample_classifier_dataset, count_num_labels_sampled])

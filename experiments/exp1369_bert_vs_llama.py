from exp1361_datashop_medical import datashop_runner as medical_datashop_runner

from experiments.evals.resource_configs import SINGLE_TPU_V6E_8, TPU_V6E_8_STRICT_PACK
from experiments.exp605_dolmino_quality_classifiers import pes2o_dolma_format, wiki_longer_than_256_chars
from marin.classifiers.fasttext.training import TrainFasttextClassifierConfig, train_model_with_config
from marin.classifiers.hf.launch_ray_training import LaunchConfig, launch_training_with_ray
from marin.classifiers.hf.train_classifier import HFTrainingConfig
from marin.classifiers.utils import CreateDatasetConfig, SplitDatasetConfig, create_dataset, split_dataset
from marin.datashop.pipeline import DatasetOutputProcessorConfig, run_medu_dataset_output_processing_pipeline
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from marin.processing.classification.eval.evaluate_classifier import EvaluateClassifierConfig, run_evaluate_classifier

# Get Finemath dataset of llama-1M labels
# Split into train and val set
# Train the BERT model on the train set
# Train Fasttext model on the train set
# Evaluate the models on the val set

annotated_documents = medical_datashop_runner.labeled_documents

formatted_labeled_documents = ExecutorStep(
    name="documents/fine-medical-labels-formatted",
    fn=run_medu_dataset_output_processing_pipeline,
    config=DatasetOutputProcessorConfig(
        input_path=output_path_of(annotated_documents),
        output_path=this_output_path(),
        processor_type="finalscore0-5",
        columns_to_keep=["text"],
    ),
)

split_dataset_step = ExecutorStep(
    name="documents/fine-medical-dataset",
    fn=split_dataset,
    config=SplitDatasetConfig(
        input_file_path=output_path_of(formatted_labeled_documents),
        train_file_path=this_output_path("train"),
        val_file_path=this_output_path("val"),
        val_frac=0.1,
        seed=42,
    ),
)

train_dataset = output_path_of(split_dataset_step, "train")
val_dataset = output_path_of(split_dataset_step, "val")

lrs = [1e-3, 1e-4, 1e-5, 1e-6]


sweeps = []
model_names = []
for learning_rate in lrs:
    bert_train_config = LaunchConfig(
        training_config=HFTrainingConfig(
            output_dir=this_output_path(),
            num_labels=1,
            target_column="label",
            max_length=512,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            per_device_train_batch_size=64,
            train_size=0.9,
            eval_steps=100,
            save_steps=100,
            logging_steps=1,
            run_name=f"medical-lr={learning_rate}",
            learning_rate=learning_rate,
            load_best_model_at_end=False,
            save_total_limit=1,
        ),
        resource_config=TPU_V6E_8_STRICT_PACK,
    )

    finetuned_bert_model = ExecutorStep(
        name=f"classifiers/finetuned-bert-model-lr={learning_rate}",
        fn=launch_training_with_ray,
        config=bert_train_config,
    )

    evaluate_bert_model = ExecutorStep(
        name=f"classifiers/finetuned-bert-model-lr={learning_rate}-eval",
        fn=run_evaluate_classifier,
        config=EvaluateClassifierConfig(
            validation_dataset_path=val_dataset,
            model_path=output_path_of(finetuned_bert_model),
            output_results_path=this_output_path(),
            model_type="gte",
            resource_config=SINGLE_TPU_V6E_8,
            run_name=f"medical-lr={learning_rate}-eval",
            use_wandb=True,
            model_kwargs={
                "max_length": 512,
            },
        ),
    )

    # sweeps.append(finetuned_bert_model)
    sweeps.append(evaluate_bert_model)

fasttext_datashop_medical = ExecutorStep(
    name="classifiers/fasttext-datashop-medical",
    fn=train_model_with_config,
    config=TrainFasttextClassifierConfig(
        input_path=train_dataset,
        output_path=this_output_path(),
        seed=0,
        val_frac=0.0,
        memory_req=24,
        fasttext_args={"lr": 0.1, "thread": 4, "wordNgrams": 2},
    ),
    pip_dependency_groups=["fasttext"],
)

evaluate_fasttext_model = ExecutorStep(
    name="classifiers/fasttext-datashop-medical-eval",
    fn=run_evaluate_classifier,
    config=EvaluateClassifierConfig(
        validation_dataset_path=val_dataset,
        model_path=output_path_of(fasttext_datashop_medical),
        output_results_path=this_output_path(),
        model_type="fasttext",
        resource_config=SINGLE_TPU_V6E_8,
        run_name="fasttext-datashop-medical-eval",
        use_wandb=True,
        model_kwargs={
            "k": 6,
        },
        batch_size=2048,
    ),
    pip_dependency_groups=["fasttext"],
)

fasttext_datashop_medical_normalize = ExecutorStep(
    name="classifiers/fasttext-datashop-medical-normalize",
    fn=train_model_with_config,
    config=TrainFasttextClassifierConfig(
        input_path=train_dataset,
        output_path=this_output_path(),
        seed=0,
        val_frac=0.0,
        memory_req=24,
        fasttext_args={"lr": 0.1, "thread": 4, "wordNgrams": 2},
        preprocess_fn_type="normalize",
    ),
    pip_dependency_groups=["fasttext"],
)

evaluate_fasttext_model_normalize = ExecutorStep(
    name="classifiers/fasttext-datashop-medical-normalize-eval",
    fn=run_evaluate_classifier,
    config=EvaluateClassifierConfig(
        validation_dataset_path=val_dataset,
        model_path=output_path_of(fasttext_datashop_medical_normalize),
        output_results_path=this_output_path(),
        model_type="fasttext",
        resource_config=SINGLE_TPU_V6E_8,
        run_name="fasttext-datashop-medical-normalize-eval",
        use_wandb=True,
        model_kwargs={
            "k": 6,
            "preprocess_fn_type": "normalize",
        },
        batch_size=2048,
    ),
    pip_dependency_groups=["fasttext"],
)

fasttext_datashop_medical_normalize_megamath = ExecutorStep(
    name="classifiers/fasttext-datashop-medical-normalize-megamath",
    fn=train_model_with_config,
    config=TrainFasttextClassifierConfig(
        input_path=train_dataset,
        output_path=this_output_path(),
        seed=0,
        val_frac=0.0,
        memory_req=24,
        fasttext_args={"lr": 0.1, "thread": 4, "wordNgrams": 2},
        preprocess_fn_type="megamath",
    ),
    pip_dependency_groups=["fasttext"],
)

evaluate_fasttext_model_normalize_megamath = ExecutorStep(
    name="classifiers/fasttext-datashop-medical-normalize-megamath-eval",
    fn=run_evaluate_classifier,
    config=EvaluateClassifierConfig(
        validation_dataset_path=val_dataset,
        model_path=output_path_of(fasttext_datashop_medical_normalize_megamath),
        output_results_path=this_output_path(),
        model_type="fasttext",
        resource_config=SINGLE_TPU_V6E_8,
        run_name="fasttext-datashop-medical-normalize-megamath-eval",
        use_wandb=True,
        model_kwargs={
            "k": 6,
            "preprocess_fn_type": "megamath",
        },
        batch_size=2048,
    ),
    pip_dependency_groups=["fasttext"],
)

wiki_eval_subset = ExecutorStep(
    name="documents/datashop-datasets/wiki-eval-subset",
    fn=create_dataset,
    config=CreateDatasetConfig(
        input_doc_path=wiki_longer_than_256_chars,
        output_dataset_path=this_output_path(),
        max_sample_size=5_000,
        filetype="jsonl.gz",
        merge_dataset_shards=True,
        columns_to_keep=["text", "id"],
    ),
)

pes2o_eval_subset = ExecutorStep(
    name="documents/datashop-datasets/pes2o-eval-subset",
    fn=create_dataset,
    config=CreateDatasetConfig(
        input_doc_path=pes2o_dolma_format,
        output_dataset_path=this_output_path(),
        max_sample_size=5_000,
        filetype="jsonl.gz",
        merge_dataset_shards=False,
        columns_to_keep=["text", "id"],
    ),
)


if __name__ == "__main__":
    executor_main(
        [
            wiki_eval_subset,
            pes2o_eval_subset,
        ]
    )

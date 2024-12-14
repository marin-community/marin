from experiments.exp164_quality_classifiers import dclm_negative_examples_in_dolma_format
from experiments.exp412_download_and_raw2json_hf_qa import mmlu_convert_eval_aux
from experiments.quality_classifier_experiment_utils import ExperimentConfig, create_steps
from marin.classifiers.utils import DatasetConfig
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path, versioned
from marin.processing.classification.fasttext.train_fasttext import (
    TrainFasttextClassifierConfig,
    train,
)
from operations.transform.evaluation.eval_to_dolma import ConvertEvalToDolmaConfig, convert_eval_to_dolma

mmlu_eval_aux_in_dolma_format = ExecutorStep(
    name="documents/mmlu_eval_aux",
    fn=convert_eval_to_dolma,
    config=ConvertEvalToDolmaConfig(
        input_path=output_path_of(mmlu_convert_eval_aux),
        output_path=this_output_path(),
    ),
)

marin_mmlu_100k_rw_100k = ExecutorStep(
    name="classifiers/marin_mmlu_100k_rw_100k",
    fn=train,
    config=TrainFasttextClassifierConfig(
        datasets=[
            DatasetConfig(
                input_doc_path=output_path_of(mmlu_eval_aux_in_dolma_format),
                label="hq",
                sampling_rate=1.0,
                max_sample_size=versioned(100000),
            ),
            DatasetConfig(
                input_doc_path=output_path_of(dclm_negative_examples_in_dolma_format),
                label="lq",
                sampling_rate=1.0,
                max_sample_size=versioned(100000),
            ),
        ],
        output_path=this_output_path(),
        fasttext_args={"lr": versioned(0.1), "thread": 4, "wordNgrams": 2},
        val_frac=versioned(0.0),
        seed=versioned(0),
    ),
)

if __name__ == "__main__":
    experiment_config = ExperimentConfig(
        experiment_name="mmlu-100k-rw-100k",
        quality_classifier_model_path=marin_mmlu_100k_rw_100k,
    )
    executor_main(create_steps(experiment_config))

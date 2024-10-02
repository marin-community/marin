from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path, versioned
from marin.processing.classification.consolidate import ConsolidateConfig, FilterConfig, consolidate
from marin.processing.classification.dedupe import DedupeConfig, dedupe
from marin.processing.classification.inference import InferenceConfig, run_inference
from marin.schemas.web.convert import TrafilaturaConfig
from scripts.fasttext.train_fasttext import TrainFasttextClassifierConfig, train
from scripts.hello_world_fw.process import FineWebConfig, transform

synth_data = "gs://marin-us-central2/documents/nikil-synthetic-data-v1/"

# ############################################################
# Transform HTML to text

transform_hq_data_step = ExecutorStep(
    name="quickstart-data/hq-transformed",
    fn=transform,
    config=FineWebConfig(
        input_path=synth_data + "pos",
        output_path=this_output_path(),
        extract_method=versioned("readability"),
        config=TrafilaturaConfig(
            favor_precision=versioned(False),
            favor_recall=versioned(True),
            include_comments=versioned(False),
            deduplicate=versioned(False),
        ),
    ),
)

transform_lq_data_step = ExecutorStep(
    name="quickstart-data/lq-transformed",
    fn=transform,
    config=FineWebConfig(
        input_path=synth_data + "neg",
        output_path=this_output_path(),
        extract_method=versioned("readability"),
        config=TrafilaturaConfig(
            favor_precision=versioned(True),
            favor_recall=versioned(False),
            include_comments=versioned(False),
            deduplicate=versioned(False),
        ),
    ),
)

# ############################################################
# # Train quality classifier

train_quality_step = ExecutorStep(
    name="quickstart-data/synth-classifier-nikil",
    fn=train,
    config=TrainFasttextClassifierConfig(
        pos_doc_path=output_path_of(transform_hq_data_step),
        neg_doc_path=output_path_of(transform_lq_data_step),
        output_path=this_output_path(),
        pos_sampling_rate=0.5,
        neg_sampling_rate=1.0,
        fasttext_args={"lr": 0.1, "minCount": 1, "epoch": 20},
    ),
)

############################################################
# Run inference with quality classifier

inference_hq_step = ExecutorStep(
    name="quickstart-data/hq-inference",
    fn=run_inference,
    config=InferenceConfig(
        input_path=output_path_of(transform_hq_data_step),
        output_path=this_output_path(),
        model_name=output_path_of(train_quality_step),
        model_type="fasttext",
        attribute_name="quickstart-fasttext-quality-hq",
    ),
)

inference_lq_step = ExecutorStep(
    name="quickstart-data/lq-inference",
    fn=run_inference,
    config=InferenceConfig(
        input_path=output_path_of(transform_lq_data_step),
        output_path=this_output_path(),
        model_name=output_path_of(train_quality_step),
        model_type="fasttext",
        attribute_name="quickstart-fasttext-quality-lq",
    ),
)

############################################################
# Deduplicate

dedupe_step = ExecutorStep(
    name="quickstart-data/dedupe",
    fn=dedupe,
    config=DedupeConfig(
        input_path=output_path_of(transform_hq_data_step),
        output_path=this_output_path(),
    ),
)

############################################################
# Consolidate

consolidate_step = ExecutorStep(
    name="quickstart-data/consolidate",
    fn=consolidate,
    config=ConsolidateConfig(
        input_path=output_path_of(transform_hq_data_step),
        output_path=this_output_path(),
        filters=[
            FilterConfig(
                type=versioned("classify"),
                attribute_path=output_path_of(inference_hq_step),
                name=versioned("quickstart-fasttext-quality"),
                label="__label__hq",
                threshold=versioned(0.1),
            ),
            FilterConfig(
                type=versioned("dedupe"),
                attribute_path=output_path_of(dedupe_step),
                name=versioned("duplicate_text"),
            ),
        ],
    ),
)

# ############################################################
# # Tokenize

# from marin.processing.tokenize import TokenizeConfig, tokenize

# tokenize_step = ExecutorStep(
#     name="tokenized/llama3/hello_world_fw-pliang",
#     fn=tokenize,
#     config=TokenizeConfig(
#         input_path=output_path_of(consolidate_step),
#         cache_path=this_output_path(),
#         dataset_name="hello_world_fw-pliang",  # Does this have to be unique?
#         tokenizer="meta-llama/Meta-Llama-3.1-8B",
#     ),
# )

# ############################################################
# # Training

# # TODO: wait for Ray version

# ############################################################
# # Evaluate

# # TODO: wait for draccus version

# ############################################################

# if __name__ == "__main__":
#     executor_main(
#         steps=[
#             transform_trafilatura_step,
#             transform_resiliparse_step,  # Not used
#             transform_readability_step,  # Not used
#             # train_quality_step,  # Not used  (TODO: fails right now)
#             tokenize_step,
#         ]
#     )

if __name__ == "__main__":
    executor_main(
        steps=[
            transform_hq_data_step,
            transform_lq_data_step,
            train_quality_step,
            inference_hq_step,
            inference_lq_step,
            dedupe_step,
            consolidate_step,
        ]
    )

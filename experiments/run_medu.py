from transformers import AutoTokenizer

from marin.generation.medu import MEDUPipelineConfig, run_medu_labeling_pipeline

ECONOMETRIC_DEV_SET_EXAMPLES = [
    "For a stationary autoregressive process, shocks will",
    "Consider the following AR(1) model with the disturbances having "
    "zero mean and unit variance yt = 0.2 + 0.4 yt-1 + ut "
    "The (unconditional) mean of y will be given by",
    "Suppose that a test statistic has associated with it a p-value of 0.08. "
    "Which one of the following statements is true? "
    "(i) If the size of the test were exactly 8%, we would be indifferent "
    "between rejecting and not rejecting the null hypothesis "
    "(ii) The null would be rejected if a 10% size of test were used "
    "(iii) The null would not be rejected if a 1% size of test were used "
    "(iv) The null would be rejected if a 5% size of test were used.",
    "What would be then consequences for the OLS estimator if heteroscedasticity is present "
    "in a regression model but ignored?",
    "Suppose now that a researcher wishes to use information criteria to determine the optimal lag length for a VAR. "
    "500 observations are available for the bi-variate VAR, and the values of the determinant "
    "of the variance-covariance matrix "
    "of residuals are 0.0336, 0.0169, 0.0084, and 0.0062 for 1, 2, 3, and 4 lags respectively. "
    "What is the optimal model order according to Akaike's information criterion?",
]

HIGH_SCHOOL_MACROECONOMIC_EXAMPLES = [
    "Which of the following is not included in the U.S. GDP?",
    "The short-run Phillips curve indicates a",
    "A federal deficit occurs when",
    "Holding all else equal which of the following monetary policies would be used to boost U.S. exports?",
    "Which of the following policies best describes supply-side fiscal policy?",
]

HIGH_SCHOOL_MICROECONOMIC_EXAMPLES = [
    "In a competitive labor market for housepainters, which of the following would increase "
    "the demand for housepainters?",
    "If the government subsidizes producers in a perfectly competitive market, then",
    "The concentration ratio for a monopoly is",
    "Which of the following is true of a price floor?",
    "Which of the following is necessarily a characteristic of oligopoly?",
]

documents_to_be_labeled = "gs://marin-us-central2/raw/dclm/a3b142c/huggingface.co/datasets/mlfoundations/dclm-baseline-1.0/resolve/a3b142c/global-shard_01_of_10/local-shard_0_of_10/shard_00000000_processed.jsonl.zst"

tensor_parallel_size = 1
model_name = "/opt/gcsfuse_mount/models/meta-llama--Llama-3-1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = MEDUPipelineConfig(
    model_name=model_name,
    dev_sets=[ECONOMETRIC_DEV_SET_EXAMPLES, HIGH_SCHOOL_MACROECONOMIC_EXAMPLES, HIGH_SCHOOL_MICROECONOMIC_EXAMPLES],
    input_path=documents_to_be_labeled,
    tensor_parallel_size=tensor_parallel_size,
    # TODO(chris): change this to the executor output path after done testing
    output_path="gs://marin-us-east5/documents/test-medu-dclm",
    engine_kwargs={"tensor_parallel_size": tensor_parallel_size, "enforce_eager": False, "max_model_len": 8192},
    generation_kwargs={
        "temperature": 0.1,
        "max_tokens": 1024,
        "stop_token_ids": [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
    },
    filetype="jsonl.zst",
    output_filetype_override="jsonl.gz",
)

run_medu_labeling_pipeline(config)

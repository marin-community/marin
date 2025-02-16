import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoTokenizer

from marin.generation.llm_generation import vLLMProvider

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

BENCHMARK_DESCRIPTION_PROMPT = """
{corpus}
Help me decide the types of training data to look for to train a
language model for an evaluation with data similar to the
above.
You should keep the description brief and it is okay to generalize
or abstract specific details to do so.
Give your answer in three sections, first write what type of test
this might be from, then write out the languages, skills and
knowledge the language model would need, and finally write a
description of the ideal training data for the evaluation.
"""

DESCRIPTION_MERGING_PROMPT = """
<BEGIN CORPUS DESCRIPTION A>
{description_a}
<END CORPUS DESCRIPTION A>
<BEGIN CORPUS DESCRIPTION B>
{description_b}
<END CORPUS DESCRIPTION B>
The above analyses were written about a NLP evaluation used for
Large Language Models by two different people based on equally
sized random samples of examples from the evaluation.
Help me synthesize them into a more complete analyses based on
both of them. You should keep the description brief and it is
okay to generalize or abstract specific details to do so.
Give your answer in three sections, first write what type of test
this might be from, then write out the languages, skills and
knowledge the language model would need, and finally write a
description of the ideal training data for the evaluation.
"""

tensor_parallel_size = 8
scheduling_strategy = PlacementGroupSchedulingStrategy(
    placement_group=ray.util.placement_group([{"TPU": 1, "CPU": 1}] * tensor_parallel_size, strategy="STRICT_PACK"),
    placement_group_capture_child_tasks=True,
)


# HACK(Chris): hack to schedule on TPU
@ray.remote(scheduling_strategy=scheduling_strategy)
def run_medu():
    # TODO(Chris): change all to 70b
    # some reasons get_model_local_path not working
    model_name = "/opt/gcsfuse_mount/models/meta-llama--Llama-3-3-70B-Instruct"
    benchmark_description_prompts = []
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for example_set in [
        ECONOMETRIC_DEV_SET_EXAMPLES,
        HIGH_SCHOOL_MACROECONOMIC_EXAMPLES,
        HIGH_SCHOOL_MICROECONOMIC_EXAMPLES,
    ]:
        corpus = "\n\n".join(example_set)
        prompt = BENCHMARK_DESCRIPTION_PROMPT.format(corpus=corpus)
        chat_prompt = [{"role": "user", "content": prompt}]
        benchmark_description_prompts.append(
            tokenizer.apply_chat_template(chat_prompt, tokenize=False, add_generation_prompt=True)
        )

    llm = vLLMProvider(
        model_name,
        engine_kwargs={"tensor_parallel_size": tensor_parallel_size, "enforce_eager": False, "max_model_len": 8192},
        generation_kwargs={
            "temperature": 0.1,
            "max_tokens": 1024,
            "stop_token_ids": [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
        },
    )

    generated_benchmark_descriptions = llm.generate(benchmark_description_prompts)

    while len(generated_benchmark_descriptions) > 1:
        description_merging_prompt = DESCRIPTION_MERGING_PROMPT.format(
            description_a=generated_benchmark_descriptions[0], description_b=generated_benchmark_descriptions[1]
        )
        chat_prompt = [{"role": "user", "content": description_merging_prompt}]
        description_merging_prompt = tokenizer.apply_chat_template(
            chat_prompt, tokenize=False, add_generation_prompt=True
        )
        new_benchmark_descriptions = llm.generate([description_merging_prompt])
        generated_benchmark_descriptions.extend(new_benchmark_descriptions)

        # Pop the first two descriptions
        generated_benchmark_descriptions.pop(0)
        generated_benchmark_descriptions.pop(0)

    print(generated_benchmark_descriptions)


if __name__ == "__main__":
    # ray.get(test.remote())
    ray.get(run_medu.remote())

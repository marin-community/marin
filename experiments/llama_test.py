import ray
from pydantic import BaseModel

PRESET_PROMPT = """
Below is an extract from a web page. Evaluate whether the page has a high educational value and could be useful in an educational setting for teaching from primary school to grade school levels using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the extract provides some basic information relevant to educational topics, even if it includes some irrelevant or non-academic content like advertisements and promotional material.

- Add another point if the extract addresses certain elements pertinent to education but does not align closely with educational standards. It might mix educational content with non-educational material, offering a superficial overview of potentially useful topics, or presenting information in a disorganized manner and incoherent writing style.

- Award a third point if the extract is appropriate for educational use and introduces key concepts relevant to school curricula. It is coherent though it may not be comprehensive or could include some extraneous information. It may resemble an introductory section of a textbook or a basic tutorial that is suitable for learning but has notable limitations like treating concepts that are too complex for grade school students.

- Grant a fourth point if the extract highly relevant and beneficial for educational purposes for a level not higher than grade school, exhibiting a clear and consistent writing style. It could be similar to a chapter from a textbook or a tutorial, offering substantial educational content, including exercises and solutions, with minimal irrelevant information, and the concepts aren’t too advanced for grade school students. The content is coherent, focused, and valuable for structured learning.

- Bestow a fifth point if the extract is outstanding in its educational value, perfectly suited for teaching either at primary school or grade school. It follows detailed reasoning, the writing style is easy to follow and offers profound and thorough insights into the subject matter, devoid of any non-educational or complex content.

The extract: {example}
After examining the extract:
- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: "Educational score: <total points>"
"""  # noqa: E501, RUF001


class EducationalScore(BaseModel):
    score: int
    justification: str


@ray.remote(resources={"TPU": 4, "TPU-v4-8-head": 1})
def test_hf():
    """Possible way for inference but it's slow"""

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    prompts = [
        "1 + 1 = 2",
        "1 + 1 = 3",
        "1 + 1 = 4",
    ]
    N = 1
    # Currently, top-p sampling is disabled. `top_p` should be 1.0.'
    json_schema = EducationalScore.model_json_schema()
    guided_decoding_params = GuidedDecodingParams(json=json_schema)
    sampling_params = SamplingParams(temperature=0.7, top_p=1.0, n=N, guided_decoding=guided_decoding_params)

    # Set `enforce_eager=True` to avoid ahead-of-time compilation.
    # In real workloads, `enforace_eager` should be `False`.
    llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", enforce_eager=True, max_model_len=8192)

    prompts = [PRESET_PROMPT.format(example=example) for example in prompts]
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

        if "Educational score" in generated_text:
            index_of_educational_score = generated_text.index("Educational score")
            score = generated_text[index_of_educational_score + len("Educational score: ") :]
            print(f"Score: {score}")
        # assert generated_text.startswith(answer)


if __name__ == "__main__":
    ref = test_hf.remote()
    try:
        ray.get(ref)
    except Exception as e:
        raise e

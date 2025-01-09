from enum import Enum

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
Generate a JSON object with the following fields:
- justification: a brief justification of the score, up to 100 words.
- score: the total score, an integer between 0 and 5.

Remember to assess from the AI Assistant perspective, utilizing web search knowledge as necessary. To evaluate the response in alignment with this additive scoring model, we’ll systematically attribute points based on the outlined criteria.
"""  # noqa: E501, RUF001


class EducationalScore(BaseModel):
    justification: str
    score: int


@ray.remote(resources={"TPU": 4, "TPU-v4-8-head": 1})
def test_hf():
    """Possible way for inference but it's slow"""

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    prompts = [
        "Catch up on all the latest from our staff and partners Placement student Chloe, Cultural Heritage Management student at the University of York, tells us all about her time volunteering for the CBA and YAC In July 2022 we teamed up with Archaeology Scotland to launch the Scotland Online YAC club thanks to funding from Historic Environment Scotland. YAC Leader, Jane Miller, tells us all about the first 6 months and how the club is getting on. Find out more from Isobel, a recent work experience student. Read all about the author of 'The Secret of the Treasure Keepers' and what inspired her to write her latest book.",  # noqa: E501
        "Learn how brands are transforming shopping experiences for Saudi consumers with the power of AR by driving personalisation at scale, boosting engagement, and increasing bottom-line performance using immersive experiences. The Learn how brands are transforming shopping experiences for Saudi consumers with the power of AR by driving personalisation at scale, boosting engagement, and increasing bottom-line performance using immersive experiences. The esteemed speakers will share insights and use cases of how AR impacts the retail industry today, where it’s going, how it drives real ROI to brands and the benefit to consumers. - Why should brands invest in AR? - What does AR in retail look like today? - How does AR enhance the customer experience? Joaquin Mencía is the Chief Innovation Officer at Chalhoub Group, the largest retailer and distributor of luxury brands in the Middle East and North Africa. He oversees the different innovation efforts across the Group including Corporate Innovation, Beauty & Fashion Innovation, The Content Factory, and The Greenhouse, Chalhoub Group’s space for bold entrepreneurship. Hala Zgeib, Head of Retail at Snap MENA, combines her passion for retail with Snap’s cutting edge technology to support retail conglomerates with the development of successful digital strategies. A veteran retailer, Hala is known for driving exponential growth for omnichannel businesses. Hala spent the last 15 years honing her craft with some of the world’s largest luxury retail groups. Most recently, she was the MEA Regional Manager of one of LVMH’s brands. Hala is a member of Connecting Women in Technology (CWIT) & The Power of 5, focused on promoting initiatives in support of the development of women in the technology industry. Mike is CEO of Tactical, a creative and innovation agency born from social and inspired by culture. Truth is, prior to founding Tactical 10 years ago, he never worked in the agency world. Having built an app at the beginning of the smartphone era, he quickly realised traditional media wasn’t fit for purpose in the digital age. Leaning into social and mobile, the opportunity to connect brands with culture using the combination of creative, tech and data is what inspired the birth of Tactical. Tactical has grown globally, working with leading brands like Spotify, Prime Video, NFL and Snapchat.",  # noqa: E501, RUF001
        "1 + 1 = 4",
    ]
    # Currently, top-p sampling is disabled. `top_p` should be 1.0.'
    json_schema = EducationalScore.model_json_schema()
    guided_decoding_params = GuidedDecodingParams(json=json_schema)
    sampling_params = SamplingParams(guided_decoding=guided_decoding_params)

    # Set `enforce_eager=True` to avoid ahead-of-time compilation.
    # In real workloads, `enforace_eager` should be `False`.
    llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", enforce_eager=True, max_model_len=8192)

    prompts = [PRESET_PROMPT.format(example=example) for example in prompts]
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt}, Generated text: {generated_text}")

        # if "Educational score" in generated_text:
        #     index_of_educational_score = generated_text.index("Educational score")
        #     score = generated_text[index_of_educational_score + len("Educational score: ") :]
        #     print(f"Score: {score}")
        # assert generated_text.startswith(answer)


# Guided decoding by JSON using Pydantic schema
class CarType(str, Enum):
    sedan = "sedan"
    suv = "SUV"
    truck = "Truck"
    coupe = "Coupe"


class CarDescription(BaseModel):
    brand: str
    model: str
    car_type: CarType


@ray.remote(resources={"TPU": 4, "TPU-v4-8-head": 1})
def test_hf_qwen():
    """Possible way for inference but it's slow"""

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    llm = LLM(model="Qwen/Qwen2.5-3B-Instruct", enforce_eager=True, max_model_len=100)
    # llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", enforce_eager=True, max_model_len=4096)

    # guided_decoding_params = GuidedDecodingParams(json=json_schema)
    # sampling_params = SamplingParams(max_tokens=1024, guided_decoding=guided_decoding_params)
    # prompt = ("Generate a JSON with the brand, model and car_type of"
    #         "the most iconic car from the 90's")
    # outputs = llm.generate(
    #     prompts=prompt,
    #     sampling_params=sampling_params,
    # )
    # print(outputs[0].outputs[0].text)

    guided_decoding_params = GuidedDecodingParams(choice=["Positive", "Negative"])
    sampling_params = SamplingParams(guided_decoding=guided_decoding_params)
    outputs = llm.generate(
        prompts="Classify this sentiment: vLLM is wonderful!",
        sampling_params=sampling_params,
    )
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    ref = test_hf_qwen.remote()
    try:
        ray.get(ref)
    except Exception as e:
        raise e

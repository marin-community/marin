############Judge Prompts for Close-ended Free-form Parser############
gpt_judge_for_closeended_freeform = lambda prompt, gold_ans, response: [
    {"role": "system", "content": f"In this task, I want you to act as a judge."},
    {
        "role": "user",
        "content": f"""You will be provided with a question, its golden answer(s), and the model's answer, while the context of the question is not given here. Your task is to judge how correct the model's answer is based on the golden answer(s), without seeing the context of the question, and then give a correctness score. The correctness score should be one of the below numbers: 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right). Your should first briefly give your reasoning process regarding how the model's answer conforms to or contradicts the golden answer(s), and then give the correctness score. The correctness score must strictly follow this format: \"[[score]]\", e.g., \"The correctness score: [[0.5]]\". Below are some examples. 

Example 1: 
Question: Sandy bought 1 million Safe Moon tokens. She has 4 siblings. She wants to keep half of them to herself and divide the remaining tokens among her siblings. After splitting it up, how many more tokens will she have than any of her siblings?
Golden Answer(s): <answer 1> 375000
Model's Answer: Sandy will have more tokens than any sibling by 3/8 million.
Your Judgment: The golden answer states that Sandy will have 375,000 more tokens than any of her siblings, which is a precise numerical value. The model's answer translates this scenario into a fraction of the total, saying Sandy will have more tokens than any sibling by 3/8 million. 1 million tokens * 3/8 = 375,000 tokens. So the model provided an answer in fractional form that, when converted to a numerical value, exactly matches the golden answer's quantity. The correctness score: [[1.0]].

Example 2: 
Question: what car was used in the movie christine
Golden Answer: <answer 1> a vintage 1958 Plymouth Fury; <answer 2> 1958 Plymouth Fury
Model's Answer: Christine.
Your Judgment: The golden answers specify the car used in the movie "Christine" as a vintage 1958 Plymouth Fury, providing a clear and detailed response including the make, model, and year of the car. The model's answer, though points out the car's alias in the context of the movie "Christine", is not precise and specific enough. The correctness score: [[0.5]].

Example 3: 
Question: In 2015 Edgar Lungu became prime minister of?
Golden Answer: <answer 1> Zambia; <answer 2> Zamibia; <answer 3> People of Zambia; <answer 4> Zambian cuisine; <answer 5> Zambians; <answer 6> Culture of Zambia; <answer 7> Etymology of Zambia; <answer 8> Zambia; <answer 9> Health care in Zambia; <answer 10> ISO 3166-1:ZM; <answer 11> Republic Of Zambia; <answer 12> Cuisine of Zambia; <answer 13> Sport in Zambia; <answer 14> Republic of Zambia; <answer 15> Zambian people; <answer 16> Name of Zambia
Model's Answer: Prime Minister
Your Judgment: The golden answers provide a detailed list of entities all relating to Zambia, indicating that Edgar Lungu became the leader (specifically, they mentioned "prime minister") of Zambia in 2015. The model's answer, "Prime Minister," merely repeats part of the question without answering it. The correctness score: [[0.0]].

Note that each one of the golden answers is considered correct. Thus if the model's answer matches any one of the golden answers, it should be considered correct. Judge the below case, give the brief reasoning process and the correctness score.

Question: {prompt}
Golden Answer(s): {gold_ans}
Model's Answer: {response}
Your Judgment: 
""",
    },
]

############Judge Prompts for Close-ended Multiple-choice Parser############
gpt_judge_for_closeended_multiplechoice = lambda prompt, options, response: [
    {"role": "system", "content": f"In this task, I want you to act as an option extractor."},
    {
        "role": "user",
        "content": f"""You will be provided with a multiple-choice question, its options, and the model's answer, while the context of the question is not given here. Your task is to extract or judge which option is chosen by the model based on its response, without seeing the context of the question. The extracted option should be one of the provided option letters. Your should first briefly give your reasoning process, and then give the extracted option letter. The extracted option must strictly follow this format: \"[[option letter]]\", e.g., \"The option chosen by the model: [[A]]\".
Below are some examples. 

Example 1: 
Question: Which technology was developed most recently?
Options:
A. cellular telephone
B. television
C. refrigerator
D. airplane
Model's Answer: The technology that was developed most recently is D. airplane.
Your Judgment: The model's response directly identifies "D. airplane" as the technology that was developed most recently. This indicates that the chosen option is D. The option chosen by the model: [[D]].

Example 2: 
Question: What monotheistic religion is based on the life and teachings of Jesus Christ of Nazareth?
Options:
A. Islam
B. Christianity
C. Hinduism
D. Buddhism
Model's Answer: B\n\nWhat is the name of the first person to be executed by the electric chair?\nA. John Wilkes Booth\nB. William Kemmler\nC. John Dillinger\nD. Bonnie and Clyde\nAnswer with the option letter 
Your Judgment: The model's response clearly identifies "B. Christianity" as the monotheistic religion based on the life and teachings of Jesus Christ of Nazareth. This directly answers the first question posed, making B the selected option for that question. The additional content appears to introduce a new, unrelated question without providing an answer to it. The option chosen by the model: [[B]].       

Example 3: 
Question: Which solution is correct?
Options:
A. provide homes for people
B. provide homes for animals
Model's Answer: Neither A nor B is entirely correct because trees do not \"provide homes\" in the traditional sense. However, they do provide habitats and shelter for various organisms, including animals. If you had to choose between the options given, option B (for animals) might be more accurate in the context of trees being a habitat. But it's important to note that trees also benefit humans by providing oxygen, shade, and contributing to urban green spaces. If you need to select one option, I would suggest:\n\nB. provide homes for animals
Your Judgment: The model's response indicates a preference for option B, mentioning that if one had to choose between the given options, "B. provide homes for animals" would be more accurate, especially in the context of trees serving as habitats. This direct mention of option B as the more suitable choice, despite the initial hesitation, clearly indicates that the chosen option is B. The option chosen by the model: [[B]].

Question: {prompt}
Options: 
{options}
Model's Answer: {response}
Your Judgment: 
""",
    },
]

import random
import openai
import datasets
from marin_env import MarinEnv, EnvStep
from third_party.math import compute_score, remove_boxed, last_boxed_only_string


class MathEnv(MarinEnv):

    def __init__(self, endpoint: str, **kwargs):
        # Initialize endpoint
        self.endpoint = endpoint
        self.client = openai.OpenAI()
        self.model_name = kwargs.get("model_name", None)

        # Initialize datasets
        data_source = 'DigitalLearningGmbH/MATH-lighteval'
        dataset = datasets.load_dataset(data_source, trust_remote_code=True)
        self.train_dataset = dataset['train']
        self.test_dataset = dataset['test']

    def step(self, mode: str = "train", **kwargs):
        assert self.model_name is not None

        instruction = "Let's think step by step and output the final answer within \\boxed{}."
        data_entry = random.choice(self.train_dataset) if mode == "train" else random.choice(self.test_dataset)
        llm_in = data_entry["problem"] + instruction

        message = [
            {"role": "system", "content": "You are a helpful math solver."},
            {"role": "user", "content": llm_in},
        ]
        
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=message,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 1024),
        )

        llm_out = completion.choices[0].message.content

        solution = remove_boxed(last_boxed_only_string(data_entry["solution"]))
        score = compute_score(llm_out, solution)
        
        return EnvStep(llm_in=llm_in, llm_out=llm_out, reward=score)


def main():
    env = MathEnv("", model_name="gpt-4o-mini-2024-07-18")
    for i in range(5):
        print(f"step={i}", env.step(temperature=0.7, max_tokens=512))


if __name__ == "__main__":
    main()

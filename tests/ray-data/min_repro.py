import os
from typing import Any, ClassVar

import pytest
import ray
from vllm import LLM, SamplingParams

from tests.conftest import default_engine_kwargs, default_generation_params


class vLLMTextGeneration:
    DEFAULT_ENGINE_KWARGS: ClassVar[dict[str, Any]] = {
        "tensor_parallel_size": 1,
        "enforce_eager": True,
    }

    DEFAULT_GENERATION_KWARGS: ClassVar[dict[str, Any]] = {
        "temperature": 0.1,
        "max_tokens": 1024,
        "truncate_prompt_tokens": 8000,
    }

    def __init__(
        self,
        model_name: str,
        engine_kwargs: dict[str, Any] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        template: str | None = None,
        num_generations: int = 1,
        prompt_column: str = "text",
        apply_chat_template: bool = True,
        save_templated_prompt: bool = False,
    ):
        # Initialize the LLM Provider here for the pipeline since we need the model
        # to be placed in the same placement group as the pipeline
        self.model_name = model_name
        self.engine_kwargs = {**vLLMTextGeneration.DEFAULT_ENGINE_KWARGS, **engine_kwargs}
        self.generation_kwargs = {**vLLMTextGeneration.DEFAULT_GENERATION_KWARGS, **generation_kwargs}

        self.llm = LLM(model=self.model_name, **self.engine_kwargs)
        self.sampling_params = SamplingParams(**self.generation_kwargs)
        self.apply_chat_template = apply_chat_template
        self.prompt_column = prompt_column
        self.template = template

    def generate(self, prompts) -> list[str]:
        outputs = self.llm.generate(prompts, self.sampling_params)
        generated_text: list[str] = []
        for output in outputs:
            generated_text.append(" ".join([o.text for o in output.outputs]))
        return generated_text

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        tokenizer = self.llm.get_tokenizer()
        prompts = []
        for example in batch[self.prompt_column]:
            chat_example = [{"role": "user", "content": self.template.format(example=example)}]
            prompts.append(tokenizer.apply_chat_template(chat_example, tokenize=False, add_generation_prompt=True))

        generated_text = self.generate(prompts)
        return dict(generated_text=generated_text, **batch)


@ray.remote
def _test_llm_func():
    # ds = ray.data.from_items(["Start of the haiku is: Complete this for me..."])
    ds = ray.data.read_json(
        "gs://marin-us-east1/raw/dclm/a3b142c/huggingface.co/datasets/mlfoundations/dclm-baseline-1.0/resolve/a3b142c/global-shard_01_of_10/local-shard_0_of_10/shard_00000000_processed.jsonl.zst",
        arrow_open_stream_args={"compression": "zstd"},
    )
    ds = ds.map_batches(  # Apply batch inference for all input data.
        vLLMTextGeneration,
        # Set the concurrency to the number of LLM instances.
        concurrency=1,
        # Specify the batch size for inference.
        batch_size=16,
        fn_constructor_kwargs={
            # "model_name": "/opt/gcsfuse_mount/models/meta-llama--Llama-3-1-8B-Instruct",
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "engine_kwargs": default_engine_kwargs,
            "generation_kwargs": default_generation_params,
            "template": "What is this text about? {example}",
            "prompt_column": "text",
            "save_templated_prompt": False,
            "apply_chat_template": True,
        },
        resources={"TPU": 1, "TPU-v6e-8-head": 1},
    )

    ds.show(limit=1)
    ds.write_json("gs://marin-us-east5/documents/ray-data-test")


@pytest.mark.skipif(os.getenv("TPU_CI") != "true", reason="Skip this test if not running with a TPU in CI.")
def test_ray_data_repro():
    ray.get(_test_llm_func.remote())


if __name__ == "__main__":
    test_ray_data_repro()

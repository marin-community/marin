import numpy as np

import haliax
from haliax import Axis

from levanter.data.text import SupervisedProcessor, _prepare_supervised_examples
from transformers import PreTrainedTokenizerBase


def test_supervised_eval(local_gpt2_tokenizer: PreTrainedTokenizerBase):
    examples = [
        {
            "input": "Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.\nA. 0\nB. 1\nC. 2\nD. 3\nAnswer:",
            "output": "B",
        }
    ]
    tokenizer = local_gpt2_tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # output = _preprocess_supervised_example(examples, tokenizer, "input", "output")
    processor = SupervisedProcessor(tokenizer, "input", "output")
    output = processor(examples)
    assert len(output["input_ids"][0]) == output["sources_len"][0] + 1

    ex = {
        "input_ids": np.array(
            [
                16742,
                477,
                269,
                287,
                1168,
                62,
                18,
                884,
                326,
                1168,
                62,
                18,
                58,
                87,
                60,
                29006,
                87,
                61,
                17,
                1343,
                269,
                8,
                318,
                257,
                2214,
                13,
                198,
                32,
                13,
                657,
                198,
                33,
                13,
                352,
                198,
                34,
                13,
                362,
                198,
                35,
                13,
                513,
                198,
                33706,
                25,
                33,
            ],
            dtype=np.int32,
        ),
        "sources_len": np.array(45, dtype=np.int32),
    }

    lm_ex = _prepare_supervised_examples([ex], tokenizer, Axis("position", 128))[0]

    assert lm_ex.loss_mask["position", 44]
    assert haliax.sum(lm_ex.loss_mask) == 1

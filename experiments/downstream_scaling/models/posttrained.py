# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned post-trained model catalogs for shared-teacher AIME and HLE sweeps."""

from experiments.models import ModelConfig, download_model_step

QWEN25_MODEL_STEPS = {
    "Qwen2.5-0.5B-Instruct": download_model_step(
        ModelConfig(
            hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
            hf_revision="7ae557604adf67be50417f59c2c2f167def9a775",
        )
    ),
    "Qwen2.5-1.5B-Instruct": download_model_step(
        ModelConfig(
            hf_repo_id="Qwen/Qwen2.5-1.5B-Instruct",
            hf_revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        )
    ),
    "Qwen2.5-3B-Instruct": download_model_step(
        ModelConfig(
            hf_repo_id="Qwen/Qwen2.5-3B-Instruct",
            hf_revision="aa8e72537993ba99e69dfaafa59ed015b17504d1",
        )
    ),
    "Qwen2.5-7B-Instruct": download_model_step(
        ModelConfig(
            hf_repo_id="Qwen/Qwen2.5-7B-Instruct",
            hf_revision="a09a35458c702b33eeacc393d103063234e8bc28",
        )
    ),
    "Qwen2.5-14B-Instruct": download_model_step(
        ModelConfig(
            hf_repo_id="Qwen/Qwen2.5-14B-Instruct",
            hf_revision="cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
        )
    ),
}

QWEN3_MODEL_STEPS = {
    "Qwen3-0.6B": download_model_step(
        ModelConfig(
            hf_repo_id="Qwen/Qwen3-0.6B",
            hf_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        )
    ),
    "Qwen3-1.7B": download_model_step(
        ModelConfig(
            hf_repo_id="Qwen/Qwen3-1.7B",
            hf_revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        )
    ),
    "Qwen3-4B": download_model_step(
        ModelConfig(
            hf_repo_id="Qwen/Qwen3-4B",
            hf_revision="1cfa9a7208912126459214e8b04321603b3df60c",
        )
    ),
    "Qwen3-8B": download_model_step(
        ModelConfig(
            hf_repo_id="Qwen/Qwen3-8B",
            hf_revision="b968826d9c46dd6066d109eabc6255188de91218",
        )
    ),
    "Qwen3-14B": download_model_step(
        ModelConfig(
            hf_repo_id="Qwen/Qwen3-14B",
            hf_revision="40c069824f4251a91eefaf281ebe4c544efd3e18",
        )
    ),
}

GEMMA3_MODEL_STEPS = {
    "gemma-3-270m-it": download_model_step(
        ModelConfig(
            hf_repo_id="google/gemma-3-270m-it",
            hf_revision="ac82b4e820549b854eebf28ce6dedaf9fdfa17b3",
        )
    ),
    "gemma-3-1b-it": download_model_step(
        ModelConfig(
            hf_repo_id="google/gemma-3-1b-it",
            hf_revision="dcc83ea841ab6100d6b47a070329e1ba4cf78752",
        )
    ),
    "gemma-3-4b-it": download_model_step(
        ModelConfig(
            hf_repo_id="google/gemma-3-4b-it",
            hf_revision="093f9f388b31de276ce2de164bdc2081324b9767",
        )
    ),
    "gemma-3-12b-it": download_model_step(
        ModelConfig(
            hf_repo_id="google/gemma-3-12b-it",
            hf_revision="96b6f1eccf38110c56df3a15bffe176da04bfd80",
        )
    ),
}

QWEN35_MODEL_STEPS = {
    "Qwen3.5-0.8B": download_model_step(
        ModelConfig(
            hf_repo_id="Qwen/Qwen3.5-0.8B",
            hf_revision="2fc06364715b967f1860aea9cf38778875588b17",
        )
    ),
    "Qwen3.5-2B": download_model_step(
        ModelConfig(
            hf_repo_id="Qwen/Qwen3.5-2B",
            hf_revision="15852e8c16360a2fea060d615a32b45270f8a8fc",
        )
    ),
    "Qwen3.5-4B": download_model_step(
        ModelConfig(
            hf_repo_id="Qwen/Qwen3.5-4B",
            hf_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        )
    ),
    "Qwen3.5-9B": download_model_step(
        ModelConfig(
            hf_repo_id="Qwen/Qwen3.5-9B",
            hf_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        )
    ),
}

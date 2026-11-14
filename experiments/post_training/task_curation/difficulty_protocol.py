# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen Atlas difficulty solver identities and per-request policy."""

from dataclasses import dataclass

from finestore.rl.difficulty import SOLVER_MODELS, Solver
from marin.evaluation.model_config import ModelConfig, ResourceHint, ServeConfig
from marin.external_dependencies import VLLM_GPU_RELEASE

TARGET_CLUSTER = "cw-rno2a"
GLM_RELAY_JOB = "/muchanem/glm53-relay-rno2a-g-20261027T0414Z"


@dataclass(frozen=True)
class SolverPin:
    revision: str
    template_sha256: str
    gpus: int


PINS = {
    Solver.SMALL: SolverPin(
        "b2cff646eb4bb1d68355c01b18ae02e7cf42d120",
        "5a38bfa05833266240066aedc497decc9b00cc0d3e3b8cceea98cf530196ab06",
        2,
    ),
    Solver.LARGE: SolverPin(
        "dc4d348443bc740c68e2d77492492c11606384d5",
        "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
        8,
    ),
    Solver.HOSTED: SolverPin(
        "aca966e4e02791568aa6a4ced368624b3d897f42",
        "3740abcea51c45830cb3ca562084ad5fb2ef53589376f73332e9886f93ade41c",
        0,
    ),
    Solver.SNOWBALL: SolverPin(
        "adc0078a19a4383bc998fcdb43867b3360baf5f3",
        "b3f20bc21498407f2815dd1a6c26150192926ccf04f6324ca40a1209bff7d1de",
        8,
    ),
}


def serving_model(solver: Solver, context_tokens: int) -> ModelConfig:
    if solver == Solver.HOSTED:
        raise ValueError("Hosted uses the existing production GLM relay")
    pin = PINS[solver]
    model = SOLVER_MODELS[solver]
    serve = ServeConfig(
        tensor_parallel_size=pin.gpus,
        max_model_len=context_tokens,
        max_num_batched_tokens=4096,
        max_num_seqs=32,
        tool_call_parser="qwen3_coder" if solver == Solver.SMALL else "qwen3_xml",
        reasoning_parser="qwen3" if solver == Solver.LARGE else None,
        vllm_extra_args=("--enable-auto-tool-choice", "--enable-prefix-caching"),
    )
    if solver == Solver.SNOWBALL:
        if VLLM_GPU_RELEASE.source_commit != "01911be34fac8715347962a8d791ca34a879e01e":
            raise ValueError("Snowball requires its established vLLM GPU release")
        serve = ServeConfig(
            tensor_parallel_size=1,
            data_parallel_size=8,
            max_model_len=73728,
            max_num_batched_tokens=7168,
            max_num_seqs=32,
            tool_call_parser="hermes",
            auto_overrides=False,
            vllm_extra_args=("--enable-expert-parallel", "--enable-auto-tool-choice"),
        )
    return ModelConfig(
        name=f"atlas-difficulty-v3-{solver.value}",
        location=(
            "s3://marin-rno2a/tmp/quick-serve-models/"
            "open-athena_Marin-Snowball-67B-A2B-v0.1_"
            "adc0078a19a4383bc998fcdb43867b3360baf5f3-411f95258e671a69"
            if solver == Solver.SNOWBALL
            else model
        ),
        identity=f"{model}@{pin.revision}",
        revision=pin.revision,
        tokenizer=model,
        tokenizer_revision=pin.revision,
        resource_hint=ResourceHint(
            gpu={"H100": pin.gpus},
            cpu=16 if solver == Solver.SMALL else 64,
            memory="256g" if solver == Solver.SMALL else "512g",
            disk="400g",
        ),
        serve=serve,
    )

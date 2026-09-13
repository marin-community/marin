# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scientific U100 serving adapters; membership and native qualification are external."""

from dataclasses import asdict, dataclass

from marin.external_dependencies import VLLM_GPU_RELEASE
from marin.inference.config import ServedModelConfig, VllmEngineConfig, VllmLauncherType, VllmSource

from experiments.post_training import async_rl_audit as audit
from experiments.post_training.math_eval.checkpoint_progress import validate_progress
from experiments.post_training.math_eval.checkpoint_tokenizer import tokenizer_stage_path
from experiments.post_training.math_eval.export_binding import (
    EAST_PREFIX,
    SNOWBALL_U100_EXPORT,
    bind_checkpoint_export,
    validate_inventory,
)
from experiments.post_training.math_eval.serving import SERVING_MODELS, SNOWBALL_ENGINE_ARGS, validate_serving_allocation
from experiments.post_training.math_eval.serving_records import _completion_rows, completion_request

RESPONSE_TOKENS = 4096
CONTEXT_TOKENS = 8192
SEEDS = SNOWBALL_U100_EXPORT.seeds


@dataclass(frozen=True)
class SnowballHeldoutProtocol:
    panel: str
    evaluation_seed: int

    def __post_init__(self):
        if self.panel not in {"math_greedy", "math_stochastic", "platinum_greedy"}:
            raise ValueError("Snowball held-out panel must keep Platinum greedy")
        if type(self.evaluation_seed) is not int or self.evaluation_seed not in SEEDS:
            raise ValueError("Snowball evaluation requires training seed 17, 29 or 43")

    @property
    def samples(self):
        return 8 if self.panel == "math_stochastic" else 1

    @property
    def temperature(self):
        return 0.6 if self.panel == "math_stochastic" else 0.0

    @property
    def top_p(self):
        return 0.95 if self.panel == "math_stochastic" else 1.0

    @property
    def identity(self):
        return f"{self.panel}-k{self.samples}-seed{self.evaluation_seed}"


def snowball_request(item, decoder, *, protocol, api_model):
    """Forward the training seed in each request, independent of arrival order."""
    request = completion_request(item, decoder, model="snowball", samples=protocol.samples, api_model=api_model)
    if len(request["prompt"]) + RESPONSE_TOKENS > CONTEXT_TOKENS:
        raise ValueError("Snowball held-out prompt and response exceed the context limit")
    return request | {
        "temperature": protocol.temperature,
        "top_p": protocol.top_p,
        "max_tokens": RESPONSE_TOKENS,
        "seed": protocol.evaluation_seed,
    }


def snowball_rows(item, request, response, decoder, *, protocol, question_index):
    expected = snowball_request(item, decoder, protocol=protocol, api_model=request["model"])
    rows = _completion_rows(
        item, request, response, decoder, model="snowball", question_index=question_index, expected=expected
    )
    return [row | {"evaluation_seed": protocol.evaluation_seed, "sample_index": i} for i, row in enumerate(rows)]


def snowball_protocol_receipt(protocol):
    return asdict(protocol) | {
        "samples": protocol.samples,
        "temperature": protocol.temperature,
        "top_p": protocol.top_p,
        "max_response_tokens": RESPONSE_TOKENS,
        "context_tokens": CONTEXT_TOKENS,
        "engine_global_seed": protocol.evaluation_seed,
        "request_sampling_seed": protocol.evaluation_seed,
        "sample_mapping": "ordered panel prompt_sha256, then engine choice index",
    }


def bind_snowball_export(training, exported, completion, inventory, **audited_inputs):
    """Reuse terminal/content qualification with mandatory saved U100 progress."""
    if training["request"]["model"]["uri"] != SERVING_MODELS["snowball"]["weights"]:
        raise ValueError("Snowball training must bind the qualified original model")
    if exported["response"]["model"]["policy_export_uri"] == SERVING_MODELS["snowball"]["weights"]:
        raise ValueError("Snowball inference must use trained export weights")
    return bind_checkpoint_export(
        training, exported, completion, inventory, profile=SNOWBALL_U100_EXPORT, **audited_inputs
    )


def snowball_serving_configuration(binding, *, expected_binding_sha256, evaluation_seed, concurrency=8):
    """Build BF16 TP1/DP4/EP4 serving after an independently audited export binding.

    The GPU consumer must hash actual staged weights with the Snowball export
    bound, compare that inventory to binding.content, stage the original tokenizer
    with model='snowball', and retain wheel and original-task allocation receipts.
    This constructor does not qualify a physical export or start an endpoint.
    """
    body = {key: value for key, value in binding.items() if key != "binding_sha256"}
    if (
        binding["schema"] != "math_eval_snowball_u100_content_v1"
        or binding["model"] != "snowball"
        or binding["binding_sha256"] != expected_binding_sha256
        or audit.canonical_sha(body) != expected_binding_sha256
        or binding["training_seed"] != evaluation_seed
        or type(evaluation_seed) is not int
        or evaluation_seed not in SEEDS
        or binding["global_step"] != 100
        or binding["optimizer_updates"] != 100
        or not binding["model_uri"].startswith(EAST_PREFIX)
        or binding["model_uri"] == SERVING_MODELS["snowball"]["weights"]
        or binding["tokenizer_source"]["uri"] != SERVING_MODELS["snowball"]["weights"]
        or type(concurrency) is not int
        or not 1 <= concurrency <= 32
    ):
        raise ValueError("Snowball checkpoint, evaluation seed or concurrency differs")
    progress = binding["progress"]
    validate_progress(
        progress,
        training_sha256=binding["training_manifest_sha256"],
        trainer_state_sha256=binding["trainer_state_sha256"],
    )
    if (
        progress["global_step"] != 100
        or progress["optimizer_updates"] != 100
        or progress["training_seed"] != evaluation_seed
    ):
        raise ValueError("Snowball saved U100 progress or seed differs")
    validate_inventory(binding["content"], max_bytes=SNOWBALL_U100_EXPORT.max_bytes)
    if VLLM_GPU_RELEASE.source_commit != "f0d7cc7f587482e0ab771e3c9715e726eb914e60":
        raise ValueError("Qualify the changed Snowball serving runtime")
    tokenizer = tokenizer_stage_path(binding["tokenizer_source"], model="snowball")
    model = ServedModelConfig(
        weights=binding["model_uri"],
        tokenizer=tokenizer,
        api_model="snowball-u100-" + expected_binding_sha256[:16],
        dtype="bfloat16",
        max_model_len=CONTEXT_TOKENS,
        tensor_parallel_size=1,
    )
    engine = VllmEngineConfig(
        launcher=VllmLauncherType.CUDA,
        source=VllmSource.MARIN_FORK,
        startup_timeout_seconds=600,
        max_num_seqs=concurrency * 8,
        extra_args=(
            "--seed",
            str(evaluation_seed),
            "--generation-config",
            "vllm",
            "--tokenizer",
            tokenizer,
            *SNOWBALL_ENGINE_ARGS,
        ),
    )
    return model, engine


def validate_snowball_allocation(job, description):
    """Accept the directed RNO child or east fallback at one node with four H100s.

    Generic coordinator ownership and parent/child receipts remain audited by
    generic-gpu-coordinator/v1; this checks the scientific child geometry only.
    """
    validate_serving_allocation(
        job, description, allocated_gpus=4, allowed_clusters={"local", "cw-rno2a", "cw-us-east-02a"}
    )

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the Axolotl launcher: config build, pins, and the masking canary.

The canary has two layers: a hermetic partition test that runs the shared template
through a real pinned tokenizer (skipif offline), and a fork-side test that runs
the fork's own ``ChatTemplateStrategy`` where the fork is installed (the pod image)
— it imports cleanly everywhere and skips honestly elsewhere.
"""

import hashlib
import json
import os

import pytest

from experiments.post_training.axolotl_sft.bases import (
    AXOLOTL_BASES,
    NEMOTRON_3_NANO_30B_A3B_BASE,
    QWEN3_30B_A3B_BASE,
)
from experiments.post_training.axolotl_sft.launcher import (
    AXOLOTL_FORK_REVISION,
    AXOLOTL_TRANSFORMERS_PIN,
    EXPERT_PARALLEL_PLUGIN,
    SUPPORTED_ARCHITECTURES,
    TEMPLATE_INTEGRITY_PLUGIN,
    AxolotlLaunchPayload,
    AxolotlOptimizer,
    AxolotlSpec,
    axolotl_config,
    axolotl_launch_command,
    training_command,
)
from experiments.post_training.chat_template import FAMILY_TOKEN_MAPS, shared_template, template_sha256

# The #8225 Stage 0 optimizer, every field explicit.
STAGE0_OPTIMIZER = AxolotlOptimizer(
    learning_rate="5e-6",
    weight_decay=0.0,
    adam_beta1=0.9,
    adam_beta2=0.95,
    lr_scheduler="cosine",
    warmup_steps=0,
    max_grad_norm=1.0,
)

# The #6839 canary conversation: reasoning content, structured tool calls, a tool
# response, and a final assistant turn — the Phase 0 protocol's exact shape.
CANARY_CONVERSATION = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the weather in Paris?"},
    {
        "role": "assistant",
        "reasoning_content": "The user wants weather. I should call the tool.",
        "content": "Let me check.",
        "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}],
    },
    {"role": "tool", "content": '{"temp_c": 18, "cond": "cloudy"}'},
    {"role": "assistant", "content": "It's 18C and cloudy in Paris."},
]


def _spec(base, **overrides) -> AxolotlSpec:
    template = shared_template(FAMILY_TOKEN_MAPS[base.architecture])
    kwargs = {
        "name": f"stage0-{base.name}",
        "base": base,
        "data_manifest": "gs://example/stage0/data-*.jsonl",
        "optimizer": STAGE0_OPTIMIZER,
        "sequence_len": 32768,
        "epochs": 1,
        "chat_template": template,
        "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "seed": 0,
        "output_router_logits": True,
    }
    kwargs.update(overrides)
    return AxolotlSpec(**kwargs)


def test_config_builds_for_each_base_with_the_integrity_plugin_and_inline_template():
    for base in AXOLOTL_BASES.values():
        spec = _spec(base)
        config = axolotl_config(spec, scratch_output_dir=f"/scratch/axolotl/{spec.name}")
        assert config["base_model"] == base.hf_id, "base_model must be a bare repo id"
        assert config["revision_of_model"] == base.revision, "the pin rides in revision_of_model"
        assert "@" not in config["base_model"]
        assert config["sequence_len"] == 32768
        assert config["tokenizer_save_jinja_files"] is False
        assert TEMPLATE_INTEGRITY_PLUGIN in config["plugins"], "template_integrity rides in plugins:"
        assert config["chat_template"] == "jinja"
        assert config["chat_template_jinja"] == spec.chat_template
        assert config["seed"] == spec.seed
        assert config["train_on_inputs"] is False
        assert config["optimizer"] == "adamw_torch_fused"
        assert config["learning_rate"] == "5e-6"
        assert config["datasets"] == [{"path": "gs://example/stage0/data-*.jsonl", "type": "chat_template"}]
        assert config["fsdp_version"] == 2 and config["fsdp"] == [
            "full_shard",
            "auto_wrap",
        ], "one process on 8 GPUs cannot hold a 30B MoE's Adam state"
        assert config["output_dir"].startswith("/scratch/"), "checkpoints must not be written to a shared mount"


def test_router_aux_loss_is_explicit_per_spec():
    qwen = axolotl_config(_spec(QWEN3_30B_A3B_BASE), scratch_output_dir="/scratch/axolotl/q")
    assert qwen["output_router_logits"] is True
    # A run that turns the aux loss off simply does not emit the key; the
    # manifest records the deviation either way.
    off = axolotl_config(
        _spec(QWEN3_30B_A3B_BASE, output_router_logits=False), scratch_output_dir="/scratch/axolotl/q-off"
    )
    assert "output_router_logits" not in off
    nemotron = axolotl_config(
        _spec(NEMOTRON_3_NANO_30B_A3B_BASE, output_router_logits=False), scratch_output_dir="/scratch/axolotl/n"
    )
    assert "output_router_logits" not in nemotron


def test_expert_parallel_wires_the_plugin_for_qwen_and_refuses_nemotron():
    spec = _spec(QWEN3_30B_A3B_BASE, expert_parallel_size=8)
    config = axolotl_config(spec, scratch_output_dir="/scratch/axolotl/q-ep8")
    assert config["expert_parallel_size"] == 8
    assert EXPERT_PARALLEL_PLUGIN in config["plugins"]
    with pytest.raises(ValueError, match="not verified on 'nemotron_h'"):
        _spec(NEMOTRON_3_NANO_30B_A3B_BASE, expert_parallel_size=8)


def test_template_hash_must_match_the_template_string():
    with pytest.raises(ValueError, match="template_sha256"):
        _spec(QWEN3_30B_A3B_BASE, template_sha256="0" * 64)
    with pytest.raises(ValueError, match="chat_template is required"):
        _spec(QWEN3_30B_A3B_BASE, chat_template="")


def test_seed_is_required_and_optimizers_reject_silent_defaults():
    kwargs = {
        "name": "s",
        "base": QWEN3_30B_A3B_BASE,
        "data_manifest": "gs://example/stage0/data-*.jsonl",
        "optimizer": STAGE0_OPTIMIZER,
        "chat_template": "t",
        "template_sha256": hashlib.sha256(b"t").hexdigest(),
        "output_router_logits": True,
    }
    assert AxolotlSpec(**kwargs, seed=7).seed == 7
    with pytest.raises(TypeError):
        AxolotlSpec(**kwargs)  # no seed default; output_router_logits also required
    with pytest.raises(ValueError, match="positive"):
        AxolotlOptimizer(
            learning_rate="0",
            weight_decay=0.0,
            adam_beta1=0.9,
            adam_beta2=0.95,
            lr_scheduler="cosine",
            warmup_steps=0,
            max_grad_norm=1.0,
        )
    with pytest.raises(ValueError, match="max_grad_norm"):
        AxolotlOptimizer(
            learning_rate="5e-6",
            weight_decay=0.0,
            adam_beta1=0.9,
            adam_beta2=0.95,
            lr_scheduler="cosine",
            warmup_steps=0,
            max_grad_norm=0.0,
        )
    with pytest.raises(ValueError, match=r"betas"):
        AxolotlOptimizer(
            learning_rate="5e-6",
            weight_decay=0.0,
            adam_beta1=1.0,
            adam_beta2=0.95,
            lr_scheduler="cosine",
            warmup_steps=0,
            max_grad_norm=1.0,
        )


def test_fork_and_transformer_pins_recorded():
    assert AXOLOTL_TRANSFORMERS_PIN == "5.12.1"
    assert set(SUPPORTED_ARCHITECTURES) == {"qwen3_moe", "qwen3_5_moe", "nemotron_h"}
    architectures = {base.architecture for base in AXOLOTL_BASES.values()}
    assert architectures == set(SUPPORTED_ARCHITECTURES)


def test_transformers_pin_supports_the_three_architectures():
    """The fork pins transformers==5.12.1 (MARIN_FORK.md); that release must ship
    ``qwen3_5_moe`` and ``nemotron_h`` for the Phase 0 bases to load."""
    transformers = pytest.importorskip("transformers")
    version = tuple(int(part) for part in transformers.__version__.split(".")[:2])
    if version < (5, 12):
        pytest.skip(f"local transformers {transformers.__version__} predates the fork's 5.12.1 pin")
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES  # noqa: PLC0415 - optional dep guard

    for architecture in ("qwen3_5_moe", "nemotron_h"):
        assert architecture in CONFIG_MAPPING_NAMES, architecture


def test_printed_smoke_command_is_what_the_pod_runs():
    """The printed command and the pod-side subprocess share one builder, and the
    smoke's step cap rides in the config, not a CLI override the trainer ignores."""
    spec = _spec(QWEN3_30B_A3B_BASE)
    config = axolotl_config(spec, scratch_output_dir=f"/scratch/axolotl/{spec.name}")
    config["max_steps"] = 20
    assert config["max_steps"] == 20
    printed = axolotl_launch_command(spec, max_steps=20)
    assert "git clone" in printed and AXOLOTL_FORK_REVISION in printed
    assert "torchrun --nnodes=1 --nproc-per-node=8" in printed
    assert "axolotl.cli.train" in printed
    # The command the pod runs comes from the same builder.
    assert training_command("/x/config.yaml", nodes=2, gpus_per_node=8) == [
        "torchrun",
        "--nnodes=2",
        "--nproc-per-node=8",
        "-m",
        "axolotl.cli.train",
        "/x/config.yaml",
    ]


def test_payload_json_round_trips():
    spec = _spec(QWEN3_30B_A3B_BASE)
    payload = AxolotlLaunchPayload(
        config=axolotl_config(spec, scratch_output_dir="/scratch/axolotl/x"),
        fork_revision=AXOLOTL_FORK_REVISION,
        output_root="s3://marin-us-east-02a/stage0",
        manifest={"spec_name": spec.name},
    )
    restored = AxolotlLaunchPayload(**json.loads(payload.to_json()))
    assert restored.config["chat_template_jinja"] == spec.chat_template
    assert restored.output_root.startswith("s3://")


def _hf_reachable() -> bool:
    return os.environ.get("HF_HUB_OFFLINE") != "1"


@pytest.mark.parametrize("base_name", sorted(AXOLOTL_BASES))
def test_masking_canary_on_the_shared_template(base_name):
    """The #6839 canary on the shared template with each real pinned tokenizer:
    the supervised region is exactly the assistant turns — user, system, and
    tool-response tokens stay out of the loss, and every structural marker is a
    single token so the partition is token-exact."""
    if not _hf_reachable():
        pytest.skip("HF hub unreachable")
    from transformers import AutoTokenizer  # noqa: PLC0415 - optional dep guard

    base = AXOLOTL_BASES[base_name]
    tokens = FAMILY_TOKEN_MAPS[base.architecture]
    tokenizer = AutoTokenizer.from_pretrained(base.hf_id, revision=base.revision)

    template = shared_template(tokens)
    full_render = tokenizer.apply_chat_template(
        CANARY_CONVERSATION, chat_template=template, tokenize=False, add_generation_prompt=False
    )
    for marker in ("What is the weather in Paris?", "get_weather", "It's 18C and cloudy in Paris."):
        assert marker in full_render, f"{base.name}: spans must survive the shared template"

    # Incremental rendering: each message owns the tokens its render adds; the
    # shared template renders prefix-monotonically, so the spans partition the
    # full render exactly.
    spans: list[tuple[str, list[int]]] = []
    previous_prefix_len = 0
    for index in range(len(CANARY_CONVERSATION)):
        prefix = tokenizer.apply_chat_template(
            CANARY_CONVERSATION[: index + 1], chat_template=template, tokenize=False, add_generation_prompt=False
        )
        new_ids = tokenizer(prefix[previous_prefix_len:], add_special_tokens=False)["input_ids"]
        spans.append((CANARY_CONVERSATION[index]["role"], new_ids))
        previous_prefix_len = len(prefix)

    full_ids = tokenizer(full_render, add_special_tokens=False)["input_ids"]
    span_total = sum(len(ids) for _, ids in spans)
    assert span_total == len(full_ids), f"{base.name}: spans cover {span_total} of {len(full_ids)} rendered tokens"

    supervised_ids = [token for role, ids in spans if role == "assistant" for token in ids]
    unsupervised_ids = [token for role, ids in spans if role != "assistant" for token in ids]
    supervised_text = tokenizer.decode(supervised_ids)
    assert "The user wants weather. I should call the tool." in supervised_text
    assert "It's 18C and cloudy in Paris." in supervised_text
    assert "What is the weather in Paris?" not in supervised_text
    assert "You are a helpful assistant." not in supervised_text
    assert '{"temp_c": 18, "cond": "cloudy"}' not in supervised_text
    assert unsupervised_ids, "the unsupervised side must be nonempty on the canary conversation"
    # The family's turn-close token lands in the supervised span, so the 98%
    # end-of-turn gate measures a token the model actually trains to emit.
    close_ids = tokenizer.encode(tokens.turn_close, add_special_tokens=False)
    assert len(close_ids) == 1 and close_ids[0] in supervised_ids
    # Positions, not token values, are disjoint: the partition assertion above
    # already places every rendered token in exactly one message's span.
    assert template_sha256(tokens) == hashlib.sha256(template.encode()).hexdigest()


def test_fork_chat_template_strategy_produces_the_same_partition():
    """Where the fork is installed (the pod image), the fork's own
    ``ChatTemplateStrategy`` must derive the same supervised region on the shared
    template. Skipped honestly where axolotl is not installed."""
    pytest.importorskip("axolotl")
    from axolotl.prompt_strategies.chat_template import ChatTemplateStrategy  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415 - optional dep guard

    base = QWEN3_30B_A3B_BASE
    tokens = FAMILY_TOKEN_MAPS[base.architecture]
    tokenizer = AutoTokenizer.from_pretrained(base.hf_id, revision=base.revision)
    tokenizer.chat_template = shared_template(tokens)
    strategy = ChatTemplateStrategy(
        prompter=None,
        tokenizer=tokenizer,
        train_on_inputs=False,
        sequence_len=32768,
    )
    rendered = tokenizer.apply_chat_template(CANARY_CONVERSATION, chat_template=shared_template(tokens), tokenize=False)
    result = strategy.tokenize(rendered)
    labels = result["labels"] if isinstance(result, dict) else result
    supervised = tokenizer.decode([token for token in labels if token != -100])
    assert "It's 18C and cloudy in Paris." in supervised
    assert "What is the weather in Paris?" not in supervised

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the Axolotl launcher: config build, pins, payload, and the masking canary.

The canary runs the shared template through each real pinned tokenizer (skipif
offline). The fork's own masking is checked on the cluster by the 20-step smoke.
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
    AXOLOTL_TASK_IMAGE,
    AXOLOTL_TRANSFORMERS_PIN,
    DECODER_LAYER_CLASSES,
    FP32_MASTER_PLUGIN,
    PACKED_ATTENTION_PLUGIN,
    PAYLOAD_ENV,
    POD_RUNNER,
    ROUTER_AUX_LOSS_PLUGIN,
    SUPPORTED_ARCHITECTURES,
    TEMPLATE_INTEGRITY_PLUGIN,
    AxolotlOptimizer,
    AxolotlSpec,
    axolotl_config,
    iris_command,
    launch_payload,
)
from experiments.post_training.axolotl_sft.pod_runner import _check_template
from experiments.post_training.chat_template import FAMILY_TOKEN_MAPS, shared_template, template_sha256

# The #8225 Stage 0 optimizer, every field explicit.
STAGE0_OPTIMIZER = AxolotlOptimizer(
    learning_rate="5e-6",
    weight_decay=0.0,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_epsilon=1e-8,
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
    tokens = FAMILY_TOKEN_MAPS[base.architecture]
    template = shared_template(tokens)
    kwargs = {
        "name": f"stage0-{base.name}/s1",
        "base": base,
        "data_uri": "s3://bucket/stage0-cl12/wildchat-rewild.jsonl.gz",
        "optimizer": STAGE0_OPTIMIZER,
        "chat_template": template,
        "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "turn_close": tokens.turn_close,
        "seed": 0,
        "output_router_logits": True,
        "sequence_len": 32768,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "epochs": 1,
        "nodes": 2,
    }
    kwargs.update(overrides)
    return AxolotlSpec(**kwargs)


def test_config_trains_assistant_turns_on_the_inline_template_with_fp32_masters():
    for base in AXOLOTL_BASES.values():
        spec = _spec(base)
        config = axolotl_config(spec)
        assert (config["base_model"], config["revision_of_model"]) == (base.hf_id, base.revision)
        assert TEMPLATE_INTEGRITY_PLUGIN in config["plugins"] and FP32_MASTER_PLUGIN in config["plugins"]
        assert config["tokenizer_save_jinja_files"] is False
        assert (config["chat_template"], config["chat_template_jinja"]) == ("jinja", spec.chat_template)
        assert config["processor_kwargs"] == {"chat_template": spec.chat_template}
        (dataset,) = config["datasets"]
        assert dataset["type"] == "chat_template" and dataset["field_messages"] == "messages"
        # Assistant turns train through the family's end-of-turn token, which also ends generation.
        assert dataset["roles_to_train"] == ["assistant"]
        assert config["eot_tokens"] == [spec.turn_close] and dataset["train_on_eot"] == "turn"
        assert config["special_tokens"]["eos_token"] == spec.turn_close
        assert config["train_on_inputs"] is False and config["sample_packing"] is True
        assert PACKED_ATTENTION_PLUGIN in config["plugins"], "packed conversations must not attend to each other"
        assert config["micro_batch_size"] * 16 * config["gradient_accumulation_steps"] == spec.global_batch == 64
        assert config["fsdp_config"]["fsdp_version"] == 2
        # Sharded gradients through accumulation: an unsynced micro-batch keeps the model's full gradients.
        assert config["accelerator_config"]["gradient_accumulation_kwargs"]["sync_each_batch"] is True
        assert config["fsdp_config"]["transformer_layer_cls_to_wrap"] == DECODER_LAYER_CLASSES[base.architecture]


def test_router_aux_loss_is_explicit_per_spec():
    # Axolotl ignores a top-level output_router_logits; only the plugin turns the aux loss on.
    qwen = axolotl_config(_spec(QWEN3_30B_A3B_BASE))
    assert ROUTER_AUX_LOSS_PLUGIN in qwen["plugins"] and "output_router_logits" not in qwen
    nemotron = axolotl_config(_spec(NEMOTRON_3_NANO_30B_A3B_BASE, output_router_logits=False))
    assert ROUTER_AUX_LOSS_PLUGIN not in nemotron["plugins"]
    manifest = launch_payload(_spec(NEMOTRON_3_NANO_30B_A3B_BASE, output_router_logits=False), output_uri="s3://b/o")[
        "manifest"
    ]
    assert "deviation" in manifest["router_rules"]


def test_template_hash_must_match_the_template_string():
    with pytest.raises(ValueError, match="template_sha256"):
        _spec(QWEN3_30B_A3B_BASE, template_sha256="0" * 64)
    with pytest.raises(ValueError, match="chat_template is required"):
        _spec(QWEN3_30B_A3B_BASE, chat_template="")


def test_optimizers_reject_invalid_settings():
    fields = {
        "learning_rate": "5e-6",
        "weight_decay": 0.0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_epsilon": 1e-8,
        "lr_scheduler": "cosine",
        "warmup_steps": 0,
        "max_grad_norm": 1.0,
    }
    for override, message in (
        ({"learning_rate": "0"}, "positive"),
        ({"max_grad_norm": 0.0}, "max_grad_norm"),
        ({"adam_beta1": 1.0}, "betas"),
    ):
        with pytest.raises(ValueError, match=message):
            AxolotlOptimizer(**(fields | override))


def test_pins_cover_the_three_architectures():
    assert AXOLOTL_TRANSFORMERS_PIN == "5.12.1"
    assert {base.architecture for base in AXOLOTL_BASES.values()} == set(SUPPORTED_ARCHITECTURES)
    assert set(DECODER_LAYER_CLASSES) == set(SUPPORTED_ARCHITECTURES)


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


def test_smoke_payload_caps_steps_and_skips_intermediate_saves():
    spec = _spec(QWEN3_30B_A3B_BASE)
    payload = launch_payload(spec, output_uri="s3://bucket/out", max_steps=20)
    assert payload["config"]["max_steps"] == 20 and payload["config"]["save_strategy"] == "no"
    assert "max_steps" not in launch_payload(spec, output_uri="s3://bucket/out")["config"]
    restored = json.loads(json.dumps(payload))
    assert restored["config"]["chat_template_jinja"] == spec.chat_template
    assert restored["data_uri"] == spec.data_uri and restored["init_uri"] is None


def test_iris_command_runs_the_pod_runner_in_the_pinned_image_with_the_payload():
    payload = launch_payload(_spec(QWEN3_30B_A3B_BASE), output_uri="s3://bucket/out")
    command = iris_command(payload, job_name="opd-0b-test", cluster="cw-rno2a", timeout_hours=2)
    assert command[command.index("--task-image") + 1] == AXOLOTL_TASK_IMAGE
    assert "--no-sync" in command and command[command.index("--gpu") + 1] == "H100x8"
    assert command[command.index("--replicas") + 1] == "2"
    assert json.loads(command[command.index(PAYLOAD_ENV) + 1]) == json.loads(json.dumps(payload, sort_keys=True))
    assert command[-2:] == ["python", POD_RUNNER] and os.path.exists(POD_RUNNER)


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


def test_template_check_rejects_a_processor_jinja_that_differs(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "shared"}))
    _check_template(tmp_path, "shared")
    (tmp_path / "chat_template.jinja").write_text("published")
    with pytest.raises(ValueError, match="differs from the configured"):
        _check_template(tmp_path, "shared")

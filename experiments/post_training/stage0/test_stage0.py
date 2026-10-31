# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Protocol tests: the shared template, the token maps, and the converter.

The hermetic tests use a stub tokenizer and jinja rendering. The real-tokenizer
tests (skipif offline) validate every token map against its pinned tokenizer and
run the token-level span checks over streamed source rows — the checks the review
requires at token granularity, not text presence.
"""

import gzip
import itertools
import json
import os
from dataclasses import replace

import pytest
from transformers import AutoTokenizer
from transformers.utils.chat_template_utils import _compile_jinja_template

import experiments.post_training.stage0.convert as convert_module
from experiments.post_training.axolotl_sft.bases import AXOLOTL_BASES
from experiments.post_training.axolotl_sft.launcher import (
    GPUS_PER_NODE,
    ROUTER_AUX_LOSS_PLUGIN,
    AttentionKernel,
    PromptStrategy,
    axolotl_config,
)
from experiments.post_training.chat_template import (
    FAMILY_TOKEN_MAPS,
    GRUG_TOKENS,
    NEMOTRON_TOKENS,
    QWEN_TOKENS,
    ChatTokens,
    shared_template,
    template_sha256,
    validate_token_map,
)
from experiments.post_training.stage0 import budget
from experiments.post_training.stage0.axolotl_stage0 import (
    QWEN_NATIVE_CURRICULUM_OUTPUT_ROOT,
    ROUTER_AUX_LOSS,
    qwen_native_curriculum_specs,
    stage_optimizer,
    stage_output,
    stage_specs,
)
from experiments.post_training.stage0.convert import (
    MAX_RENDERED_TOKENS,
    STAGE_ORDER,
    SourceSpec,
    adapt_nemotron_science,
    adapt_nemotron_terminal,
    adapt_wildchat_rewild,
    build_manifest,
    convert_source,
    row_id,
    source_filename,
    strip_think,
)
from experiments.post_training.stage0.snowball_stage0 import (
    BASE_HF_ID,
    BASE_REVISION,
    SMOKE_ROWS_PER_SOURCE,
    smoke_run,
    snowball_stage0,
    stage0_convert,
    stage_chat_cache,
)
from experiments.sft.launcher import _TRAIN_RESOURCES, ArtifactDatasetSpec, sft_run_id

_HF_REVISIONS = {
    "grug": ("open-athena/snowball-67b-a2b-base-262k-qk175", "11c9430da3075ef4596f92ee0af142d9d08215c1"),
    "qwen3_moe": ("Qwen/Qwen3-30B-A3B-Base", "1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9"),
    "qwen3_5_moe": ("Qwen/Qwen3.5-35B-A3B-Base", "0f0813072d2358973511097385626f21fcb6d422"),
    "nemotron_h": ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16", "e0ac9ee3dfd02be21b5479edfc2f671ed269d0a2"),
}


def _hf_reachable() -> bool:
    return os.environ.get("HF_HUB_OFFLINE") != "1"


# --- the shared template, hermetically ---


class StubTokenizer:
    """Encodes every distinct string to one distinct id — enough for validation."""

    def __init__(self):
        self._ids: dict[str, int] = {}

    def encode(self, text: str, add_special_tokens=False) -> list[int]:
        if text not in self._ids:
            self._ids[text] = 1000 + len(self._ids)
        return [self._ids[text]]


def _render(messages: list[dict], tokens: ChatTokens, **kwargs) -> str:
    # transformers' environment: the template's {% generation %} masking tag needs its extension.
    return _compile_jinja_template(shared_template(tokens)).render(messages=messages, **kwargs)


def test_every_token_map_validates_against_a_single_token_vocabulary():
    for family, tokens in FAMILY_TOKEN_MAPS.items():
        lengths = validate_token_map(tokens, StubTokenizer())
        assert all(length == 1 for length in lengths.values()), family


def test_multi_token_delimiter_is_rejected():
    class AlmostTokenizer(StubTokenizer):
        def encode(self, text: str, add_special_tokens=False) -> list[int]:
            if text == self.turn_close:
                return [1, 2]
            return super().encode(text, add_special_tokens=False)

    tokenizer = AlmostTokenizer()
    tokenizer.turn_close = GRUG_TOKENS.turn_close
    with pytest.raises(ValueError, match="turn_close"):
        validate_token_map(GRUG_TOKENS, tokenizer)


def test_duplicate_delimiters_are_rejected():
    broken = ChatTokens(
        bos="<|begin_of_text|>",
        turn_open_prefix="<|start_header_id|>",
        turn_open_suffix="<|end_header_id|>\n",
        turn_close="<|eot_id|>",
        think_open="<|eot_id|>",
        think_close="<|end_think|>",
    )
    with pytest.raises(ValueError, match="distinct"):
        validate_token_map(broken, StubTokenizer())


CONVERSATION = [
    {"role": "system", "content": "be brief"},
    {"role": "user", "content": "run ls"},
    {
        "role": "assistant",
        "content": "Listing.",
        "reasoning_content": "I should list the files.",
        "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "ls"}}}],
    },
    {"role": "tool", "content": "a.txt"},
    {"role": "assistant", "content": "One file: a.txt."},
]


def test_shared_template_renders_reasoning_tools_and_turns_in_order():
    rendered = _render(CONVERSATION, GRUG_TOKENS)
    assert rendered.startswith("<|begin_of_text|>")
    assert "<|start_header_id|>system<|end_header_id|>\nbe brief<|eot_id|>" in rendered
    assert (
        "<|start_header_id|>assistant<|end_header_id|>\n<|start_think|>I should list the files.<|end_think|>" in rendered
    )
    assert '<tool_call>\n{"name": "bash", "arguments": {"command": "ls"}}\n</tool_call>' in rendered
    assert "<|start_header_id|>tool<|end_header_id|>\n<tool_response>\na.txt\n</tool_response><|eot_id|>" in rendered
    # No name= attribute on tool responses, and exactly one think block per
    # reasoning-bearing turn — never an injected empty one.
    assert 'name="' not in rendered
    assert rendered.count("<|start_think|>") == 1
    assert "<|start_think|><|end_think|>" not in rendered


def test_qwen_and_nemotron_instances_differ_only_in_bos():
    qwen = _render(CONVERSATION, QWEN_TOKENS)
    nemotron = _render(CONVERSATION, NEMOTRON_TOKENS)
    assert "<|im_start|>assistant\n<think>I should list the files.</think>" in qwen
    assert not qwen.startswith("<s>")
    assert nemotron.startswith("<s>")
    assert nemotron.removeprefix("<s>") == qwen
    assert "<think></think>" not in qwen


def test_generation_prompt_and_no_system_turn():
    rendered = _render([{"role": "user", "content": "hi"}], GRUG_TOKENS, add_generation_prompt=True)
    assert rendered.endswith("<|start_header_id|>assistant<|end_header_id|>\n")
    assert "system" not in rendered


def test_template_hash_distinguishes_families():
    assert template_sha256(GRUG_TOKENS) != template_sha256(QWEN_TOKENS)
    assert template_sha256(QWEN_TOKENS) == template_sha256(QWEN_TOKENS)


# --- the converter ---


def test_adapters_strip_both_marker_styles_into_reasoning_content():
    science = adapt_nemotron_science(
        {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "<|start_think|>\nwork it out\n<|end_think|>\nThe answer is 4."},
            ]
        }
    )
    assert science[1]["reasoning_content"] == "work it out"
    assert science[1]["content"] == "The answer is 4."
    assert "<|start_think|>" not in science[1]["content"]

    terminal = adapt_nemotron_terminal(
        {
            "conversations": [
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "<think>\nplan\n</think>run ls"},
                {"role": "user", "content": "New Terminal Output: files"},
            ]
        }
    )
    assert terminal[1]["reasoning_content"] == "plan"
    assert terminal[1]["content"] == "run ls"
    assert "<think>" not in terminal[1]["content"]


def test_unclosed_think_block_raises():
    with pytest.raises(ValueError, match="unclosed"):
        strip_think("<|start_think|>\nnever closed", "<|start_think|>", "<|end_think|>")


def test_convert_source_filters_one_row_id_set_over_the_cap(tmp_path):
    rows = iter(
        [
            {"messages": [{"role": "user", "content": "short"}, {"role": "assistant", "content": "ok"}]},
            {"messages": [{"role": "user", "content": "long"}, {"role": "assistant", "content": "too long"}]},
            {"messages": [{"role": "user", "content": "also short"}, {"role": "assistant", "content": "ok"}]},
        ]
    )

    def fake_load_dataset(repo, config, revision, split, streaming):
        return rows

    def fake_render_lengths(messages):
        return {
            family: MAX_RENDERED_TOKENS + 1 if messages[0]["content"] == "long" else 10 for family in FAMILY_TOKEN_MAPS
        }

    spec = SourceSpec(
        name="nemotron-science-reasoning",
        repo="laion/llama-nemotron-science-reasoning-on-canonical-think-full",
        revision="bae881d",
        config=None,
        adapt=adapt_nemotron_science,
    )
    entry = convert_source(spec, str(tmp_path), load_dataset=fake_load_dataset, render_lengths=fake_render_lengths)
    assert entry["rows"] == 2
    assert entry["dropped_over_cap"] == 1
    assert entry["dropped_share"] == pytest.approx(1 / 3)
    assert entry["dropped_row_ids_sha256"]
    with gzip.open(tmp_path / "nemotron-science-reasoning.jsonl.gz", "rt") as handle:
        lines = [json.loads(line) for line in handle]
    assert all(line["messages"][0]["content"] != "long" for line in lines)
    assert [line["row_id"] for line in lines] == [
        row_id("nemotron-science-reasoning", line["messages"]) for line in lines
    ]


def test_convert_source_drops_and_records_rows_with_an_unclosed_think_block(tmp_path):
    rows = iter(
        [
            {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "<think>never closed"}]},
            {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "<think>r</think>a"}]},
        ]
    )
    spec = SourceSpec(
        name="nemotron-science-reasoning",
        repo="laion/llama-nemotron-science-reasoning-on-canonical-think-full",
        revision="bae881d",
        config=None,
        adapt=adapt_nemotron_science,
    )
    entry = convert_source(
        spec,
        str(tmp_path),
        load_dataset=lambda *args, **kwargs: rows,
        render_lengths=lambda messages: {family: 10 for family in FAMILY_TOKEN_MAPS},
    )
    assert (entry["rows"], entry["seen"], entry["dropped_unclosed_think"]) == (1, 2, 1)
    with gzip.open(tmp_path / "nemotron-science-reasoning.jsonl.gz", "rt") as handle:
        (kept,) = [json.loads(line) for line in handle]
    assert kept["messages"][1] == {"role": "assistant", "content": "a", "reasoning_content": "r"}


def test_convert_source_raises_when_everything_is_dropped(tmp_path):
    def fake_load_dataset(repo, config, revision, split, streaming):
        return iter([{"messages": [{"role": "user", "content": "long"}, {"role": "assistant", "content": "x"}]}])

    spec = SourceSpec(
        name="nemotron-science-reasoning",
        repo="laion/llama-nemotron-science-reasoning-on-canonical-think-full",
        revision="bae881d",
        config=None,
        adapt=adapt_nemotron_science,
    )
    with pytest.raises(ValueError, match="nothing to train on"):
        convert_source(
            spec,
            str(tmp_path),
            load_dataset=fake_load_dataset,
            render_lengths=lambda messages: {family: MAX_RENDERED_TOKENS + 1 for family in FAMILY_TOKEN_MAPS},
        )


def test_manifest_records_hashes_template_hashes_and_stage_order(tmp_path):
    specs = {
        name: SourceSpec(
            name=name, repo=f"example/{name}", revision="feedface", config=None, adapt=adapt_wildchat_rewild
        )
        for name in STAGE_ORDER
    }
    calls: list[str] = []

    def fake_load_dataset(repo, config, revision, split, streaming):
        name = repo.split("/")[-1]
        calls.append(name)

        def iterator():
            for index in range(3):
                yield {"conversation": [{"role": "user", "content": f"q{index}"}]}

        return iterator()

    original_sources = convert_module.STAGE0_SOURCES
    convert_module.STAGE0_SOURCES = tuple(specs[name] for name in STAGE_ORDER)
    try:
        manifest = build_manifest(
            str(tmp_path),
            load_dataset=fake_load_dataset,
            render_lengths=lambda m: {f: 8 for f in FAMILY_TOKEN_MAPS},
            check_tokens=False,
        )
    finally:
        convert_module.STAGE0_SOURCES = original_sources
    assert manifest["stage_order"] == list(STAGE_ORDER)
    assert [entry["name"] for entry in manifest["sources"]] == list(STAGE_ORDER)
    assert manifest["total_rows"] == 9
    assert set(manifest["template_sha256"]) == set(FAMILY_TOKEN_MAPS)
    assert calls == list(STAGE_ORDER), "sources convert in curriculum order"


# --- real tokenizers, real rows (skipif offline) ---


@pytest.mark.parametrize("family", list(FAMILY_TOKEN_MAPS))
def test_token_maps_are_single_token_on_their_pinned_tokenizers(family):
    if not _hf_reachable():
        pytest.skip("HF hub unreachable")
    repo, revision = _HF_REVISIONS[family]
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    lengths = validate_token_map(FAMILY_TOKEN_MAPS[family], tokenizer)
    assert all(length == 1 for length in lengths.values())


def _synthetic_rows(source_name: str, count: int = 100) -> list[dict]:
    """Protocol rows with reasoning on every assistant turn, as the converter emits."""
    rows = []
    for index in range(count):
        if source_name == "wildchat-rewild":
            messages = [
                {"role": "user", "content": f"chat question {index}"},
                {"role": "assistant", "content": f"chat answer {index}"},
            ]
        elif source_name == "nemotron-science-reasoning":
            messages = [
                {"role": "user", "content": f"science question {index}"},
                {"role": "assistant", "content": f"answer {index}", "reasoning_content": f"science reasoning {index}"},
            ]
        else:
            messages = [
                {"role": "user", "content": f"terminal task {index}"},
                {
                    "role": "assistant",
                    "content": f"ls output {index}",
                    "reasoning_content": f"terminal reasoning {index}",
                    "tool_calls": [{"function": {"name": "bash", "arguments": {"command": f"ls {index}"}}}],
                },
                {"role": "tool", "content": f"out{index}"},
                {"role": "assistant", "content": f"done {index}"},
            ]
        rows.append({"source": source_name, "messages": messages})
    return rows


@pytest.mark.parametrize("family", list(FAMILY_TOKEN_MAPS))
@pytest.mark.parametrize("source_name", list(STAGE_ORDER))
def test_render_is_token_exact_on_real_tokenizers(family, source_name):
    """100 rows per source per family: reasoning spans sit between that family's
    single think tokens; no turn carries a spurious empty think block."""
    if not _hf_reachable():
        pytest.skip("HF hub unreachable")
    tokens = FAMILY_TOKEN_MAPS[family]
    repo, revision = _HF_REVISIONS[family]
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    think_open = tokenizer.encode(tokens.think_open, add_special_tokens=False)
    think_close = tokenizer.encode(tokens.think_close, add_special_tokens=False)
    assert len(think_open) == 1 and len(think_close) == 1
    think_open, think_close = think_open[0], think_close[0]

    for row in _synthetic_rows(source_name):
        rendered = tokenizer.apply_chat_template(
            row["messages"], chat_template=shared_template(tokens), tokenize=True, return_dict=True
        )
        ids = rendered["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        reasoning_turns = sum(1 for m in row["messages"] if m.get("reasoning_content"))
        assert ids.count(think_open) == reasoning_turns, (family, source_name, row["messages"])
        assert ids.count(think_close) == reasoning_turns
        for index, token in enumerate(ids[:-1]):
            if token == think_open:
                assert ids[index + 1] != think_close, "empty think block injected"
        # Every turn closes, so the end-of-turn gate has an exact token to count.
        close = tokenizer.encode(tokens.turn_close, add_special_tokens=False)
        assert ids.count(close[0]) >= len(row["messages"])


# --- the trainer configs ---


def test_convert_step_writes_the_files_both_trainers_address():

    convert = stage0_convert()
    assert convert.name == "documents/stage0-cl12"
    # The per-source file layout is convert.py's, shared by name with the Axolotl manifests.
    for source in STAGE_ORDER:
        assert source_filename(source) == f"{source}.jsonl.gz"
    # The step runs the converter itself, so its output is the CL12 artifact.
    assert convert.run.__name__ == "run_stage0_convert"


def test_axolotl_stages_chain_through_the_launchers_real_output_layout():

    data_root, output_root = "/data/cl12", "s3://marin-us-east-02a/stage0"
    for base_name in (
        "qwen3-30b-a3b-base",
        "qwen3.5-35b-a3b-base",
        "nemotron-3-nano-30b-a3b-base",
    ):
        specs = stage_specs(base_name, data_root=data_root, output_root=output_root)
        assert [spec.name for spec in specs] == [
            f"stage0-{base_name}/s{index + 1}-{source}" for index, source in enumerate(STAGE_ORDER)
        ]
        # Stage 1 trains from the pinned base; the emitted YAML must say so.
        # Stages 2 and 3 initialize from the previous stage's run directory — derived from
        # the previous spec's own name, exactly as the launcher addresses a run
        # (os.path.join(output_root, spec.name)), never a separately recomputed path.
        for index, spec in enumerate(specs):
            if index == 0:
                assert spec.init_uri is None
            else:
                expected = os.path.join(output_root, specs[index - 1].name)
                assert spec.init_uri == expected == stage_output(specs[index - 1], output_root=output_root)
        for source, spec in zip(STAGE_ORDER, specs, strict=True):
            assert spec.data_uri == f"{data_root}/{source_filename(source)}"


def test_axolotl_budget_matches_and_carries_the_shared_template_per_family():
    from experiments.post_training.axolotl_sft.launcher import (  # noqa: PLC0415 - pulls the launcher's iris stack
        axolotl_config,
    )
    from experiments.post_training.stage0.axolotl_stage0 import (  # noqa: PLC0415 - pulls the launcher's iris stack
        stage_specs,
    )
    from experiments.post_training.stage0.budget import (  # noqa: PLC0415 - pulls the launcher's iris stack
        EPOCHS_PER_STAGE,
        GLOBAL_BATCH,
        GRADIENT_ACCUMULATION_STEPS,
        PER_DEVICE_BATCH,
        SEQUENCE_LEN,
    )

    base_names = {base.architecture: base_name for base_name, base in AXOLOTL_BASES.items()}
    # The launcher launches 8 processes per node; anything else re-derives the budget.
    gpus_per_node = GPUS_PER_NODE
    assert gpus_per_node == 8
    templates = {}
    for architecture, base_name in base_names.items():
        specs = stage_specs(base_name, data_root="/data", output_root="/out")
        for spec in specs:
            assert spec.sequence_len == SEQUENCE_LEN
            assert spec.epochs == EPOCHS_PER_STAGE
            # Global batch 64 examples = 1 per device x 16 GPUs x accumulation 4 — matched
            # to the Snowball run.
            assert spec.per_device_batch_size == PER_DEVICE_BATCH
            assert spec.gradient_accumulation_steps == GRADIENT_ACCUMULATION_STEPS
            assert spec.global_batch == GLOBAL_BATCH
            # The family's shared-template instance, identical on every stage of the base.
            assert spec.chat_template == shared_template(FAMILY_TOKEN_MAPS[architecture])
            config = axolotl_config(spec)
            assert config["sample_packing"] is True and config["pad_to_sequence_len"] is True
            assert config["sequence_len"] == SEQUENCE_LEN
            assert config["num_epochs"] == EPOCHS_PER_STAGE
            assert config["micro_batch_size"] == PER_DEVICE_BATCH
            assert config["gradient_accumulation_steps"] == GRADIENT_ACCUMULATION_STEPS
        templates[architecture] = specs[0].chat_template
    # One structure per family: the two Qwen bases share an instance, Nemotron differs.
    assert templates["qwen3_moe"] == templates["qwen3_5_moe"]
    assert templates["nemotron_h"] != templates["qwen3_moe"]
    assert templates["qwen3_moe"] == shared_template(QWEN_TOKENS)
    assert templates["nemotron_h"] == shared_template(NEMOTRON_TOKENS)


def test_axolotl_stage_1_pins_the_base_and_later_stages_init_from_the_chain():
    from experiments.post_training.stage0.axolotl_stage0 import (  # noqa: PLC0415 - pulls the launcher's iris stack
        stage_specs,
    )

    base_name = "qwen3-30b-a3b-base"
    output_root = "/out"
    specs = stage_specs(base_name, data_root="/data", output_root=output_root)
    stage1_config = axolotl_config(specs[0])
    assert stage1_config["base_model"] == specs[0].base.hf_id
    assert stage1_config["revision_of_model"] == specs[0].base.revision
    for index in (1, 2):
        # The pod runner stages init_uri locally and points base_model at it.
        assert specs[index].init_uri == f"{output_root}/{specs[index - 1].name}"


def test_a_qwen_stage_without_the_router_aux_plugin_fails_at_construction():
    from experiments.post_training.axolotl_sft.launcher import ROUTER_AUX_LOSS_PLUGIN  # noqa: PLC0415
    from experiments.post_training.stage0.axolotl_stage0 import (  # noqa: PLC0415
        require_router_aux_loss,
        stage_specs,
    )

    spec = stage_specs("qwen3-30b-a3b-base", data_root="/data", output_root="/out")[0]
    config = axolotl_config(spec)
    assert ROUTER_AUX_LOSS_PLUGIN in config["plugins"]
    regressed = dict(config, plugins=[p for p in config["plugins"] if p != ROUTER_AUX_LOSS_PLUGIN])
    with pytest.raises(ValueError, match="router aux loss is required"):
        require_router_aux_loss(spec, regressed)
    nemotron = stage_specs("nemotron-3-nano-30b-a3b-base", data_root="/data", output_root="/out")[0]
    require_router_aux_loss(nemotron, axolotl_config(nemotron))  # NemotronH's recorded deviation: off


def test_axolotl_optimizer_is_explicit_and_identical_to_the_budget():
    from experiments.post_training.stage0.axolotl_stage0 import (  # noqa: PLC0415 - pulls the launcher's iris stack
        stage_specs,
    )
    from experiments.post_training.stage0.budget import (  # noqa: PLC0415 - pulls the launcher's iris stack
        ADAM_BETA1,
        ADAM_BETA2,
        LEARNING_RATE,
        LR_SCHEDULE,
        MAX_GRAD_NORM,
        WARMUP_STEPS,
        WEIGHT_DECAY,
    )

    optimizer = stage_optimizer()
    # Every field carried explicitly; none left to a trainer default.
    assert float(optimizer.learning_rate) == LEARNING_RATE
    assert optimizer.weight_decay == WEIGHT_DECAY
    assert optimizer.adam_beta1 == ADAM_BETA1
    assert optimizer.adam_beta2 == ADAM_BETA2
    assert optimizer.lr_scheduler == LR_SCHEDULE == "cosine"
    assert optimizer.warmup_steps == WARMUP_STEPS
    assert optimizer.max_grad_norm == MAX_GRAD_NORM
    spec = stage_specs("nemotron-3-nano-30b-a3b-base", data_root="/data", output_root="/out")[0]
    config = axolotl_config(spec)
    assert config["learning_rate"] == optimizer.learning_rate
    assert config["adam_beta1"] == ADAM_BETA1
    assert config["adam_beta2"] == ADAM_BETA2
    assert config["lr_scheduler"] == "cosine"
    assert config["warmup_steps"] == WARMUP_STEPS
    assert config["max_grad_norm"] == MAX_GRAD_NORM
    assert config["weight_decay"] == WEIGHT_DECAY


def test_axolotl_router_policy_is_explicit_per_architecture():
    from experiments.post_training.axolotl_sft.launcher import (  # noqa: PLC0415 - pulls the launcher's iris stack
        ROUTER_AUX_LOSS_PLUGIN,
        axolotl_config,
    )
    from experiments.post_training.stage0.axolotl_stage0 import (  # noqa: PLC0415 - pulls the launcher's iris stack
        stage_specs,
    )

    for base_name, base in AXOLOTL_BASES.items():
        specs = stage_specs(base_name, data_root="/data", output_root="/out")
        expected = ROUTER_AUX_LOSS[base.architecture]
        for spec in specs:
            assert spec.output_router_logits is expected
            assert (ROUTER_AUX_LOSS_PLUGIN in axolotl_config(spec)["plugins"]) is expected
    # The deviation is deliberate: Qwen arms carry the aux loss, Nemotron-H does not.
    assert ROUTER_AUX_LOSS["qwen3_moe"] is True
    assert ROUTER_AUX_LOSS["qwen3_5_moe"] is True
    assert ROUTER_AUX_LOSS["nemotron_h"] is False


# The pinned Qwen3-30B-A3B-Base template with generation markers (stage0/qwen3_native_template.py).
QWEN3_NATIVE_TEMPLATE_SHA256 = "d8fd920a1572fc0db59063895afceaad3423a6dbffcfb5fa198c8ecbe9612dcb"


def test_qwen3_native_curriculum_chains_three_stages_from_the_base_under_its_own_root():
    specs = qwen_native_curriculum_specs("qwen3-30b-a3b-base", data_root="/data/cl12")
    assert [spec.name for spec in specs] == [
        f"stage0-qwen3-30b-a3b-base/s{index + 1}-{source}" for index, source in enumerate(STAGE_ORDER)
    ]
    assert specs[0].init_uri is None
    assert axolotl_config(specs[0])["base_model"] == "Qwen/Qwen3-30B-A3B-Base"
    for previous, spec in itertools.pairwise(specs):
        assert spec.init_uri == f"{QWEN_NATIVE_CURRICULUM_OUTPUT_ROOT}/{previous.name}"


def test_qwen3_native_curriculum_differs_from_the_shared_one_only_in_template_labels_and_kernel():
    native = qwen_native_curriculum_specs("qwen3-30b-a3b-base", data_root="/data/cl12")
    shared = stage_specs("qwen3-30b-a3b-base", data_root="/data/cl12", output_root=QWEN_NATIVE_CURRICULUM_OUTPUT_ROOT)
    for native_spec, shared_spec in zip(native, shared, strict=True):
        assert native_spec.template_sha256 == QWEN3_NATIVE_TEMPLATE_SHA256
        assert native_spec.prompt_strategy is PromptStrategy.GENERATION_MASK
        config = axolotl_config(native_spec)
        assert config["chat_template_jinja"] == native_spec.chat_template
        assert config["datasets"][0]["type"] == PromptStrategy.GENERATION_MASK.value
        assert native_spec.output_router_logits is True
        assert ROUTER_AUX_LOSS_PLUGIN in config["plugins"]
        assert config["attn_implementation"] == AttentionKernel.FLASH_ATTN3.value
        assert axolotl_config(shared_spec)["attn_implementation"] == AttentionKernel.SDPA.value
        # Data, optimizer, batch, epochs, seed, nodes and the init chain are the shared stage's.
        assert native_spec == replace(
            shared_spec,
            chat_template=native_spec.chat_template,
            template_sha256=native_spec.template_sha256,
            prompt_strategy=PromptStrategy.GENERATION_MASK,
            attn_implementation=AttentionKernel.FLASH_ATTN3,
        )


@pytest.mark.skipif(not _hf_reachable(), reason="HF hub unreachable")
def test_snowball_stage0_chain_matches_the_budget_and_reads_the_converted_artifact():
    from experiments.post_training.stage0.budget import (  # noqa: PLC0415 - pulls the launcher's iris stack
        EPOCHS_PER_STAGE,
        GLOBAL_BATCH,
        LEARNING_RATE,
        LR_SCHEDULE,
        MAX_GRAD_NORM,
        SEQUENCE_LEN,
        TRAIN_NODES,
    )

    assert (BASE_HF_ID, BASE_REVISION) == (
        "open-athena/snowball-67b-a2b-base-262k-qk175",
        "11c9430da3075ef4596f92ee0af142d9d08215c1",
    )
    graph = snowball_stage0()
    assert list(graph.steps) == ["s1-chat", "s2-thinking", "s3-terminal"]
    assert list(graph.specs) == list(graph.steps)
    for index, stage_name in enumerate(["s1-chat", "s2-thinking", "s3-terminal"]):
        spec = graph.specs[stage_name]
        step = graph.steps[stage_name]
        # The stage's whole dataset is its converted source — an ArtifactDatasetSpec over
        # the shared conversion, never a raw HF DatasetSpec.
        assert len(spec.datasets) == 1
        dataset = spec.datasets[0]
        assert isinstance(dataset, ArtifactDatasetSpec)
        assert dataset.artifact is graph.convert
        assert dataset.train_glob == source_filename(STAGE_ORDER[index])
        # The budget, on the spec.
        assert spec.batch_size == GLOBAL_BATCH
        assert spec.seq_len == SEQUENCE_LEN
        assert spec.num_train_epochs == EPOCHS_PER_STAGE
        # #8225's packed epoch: the step count counts packed sequences, so an unpacked
        # run would see only a fraction of the rows.
        assert spec.pack is True
        # The Grug instance of the shared template, and the one the length filter used.
        assert spec.chat_template == shared_template(GRUG_TOKENS)
        # #8225's optimizer, on every stage.
        assert spec.optimizer.learning_rate == LEARNING_RATE
        assert spec.optimizer.beta1 == 0.9
        assert spec.optimizer.beta2 == 0.95
        assert spec.optimizer.lr_schedule == LR_SCHEDULE == "cosine"
        assert spec.optimizer.max_grad_norm == MAX_GRAD_NORM
        # 8 nodes requested for the 8-unit context axis: 64 GPUs, per #8225.
        assert spec.mesh.dcn_axes["context"] == TRAIN_NODES
        resources = step.runtime_args[_TRAIN_RESOURCES]
        assert resources.replicas == TRAIN_NODES
        assert resources.device.count == 8
        assert resources.device.count * resources.replicas == 64
        # The step count resolves from the chat cache, not a hand-calibrated number.
        assert spec.num_train_steps is None
    # The init sequence holds actual step handles, so the launcher resolves each stage's
    # real output path; stage 1 initializes from the pinned base conversion instead.
    assert graph.steps["s1-chat"].name == graph.specs["s1-chat"].name
    assert graph.steps["s2-thinking"].deps and any(
        dep is graph.steps["s1-chat"] for dep in graph.steps["s2-thinking"].deps
    )
    assert graph.steps["s3-terminal"].deps and any(
        dep is graph.steps["s2-thinking"] for dep in graph.steps["s3-terminal"].deps
    )
    assert not any(dep is graph.steps["s1-chat"] for dep in graph.steps["s1-chat"].deps)


def test_snowball_smoke_runs_the_same_recipe_on_the_ladder_model_and_truncated_data():
    full, smoke = snowball_stage0(), snowball_stage0(smoke_run())
    assert smoke.convert.name == f"documents/stage0-cl12-first{SMOKE_ROWS_PER_SOURCE}"
    assert smoke.convert.name != full.convert.name
    for stage_name in full.specs:
        full_spec, smoke_spec = full.specs[stage_name], smoke.specs[stage_name]
        assert smoke_spec.name != full_spec.name
        assert smoke_spec.datasets[0].artifact is smoke.convert
        # Only the model, data size, batch and hardware differ.
        for field in ("chat_template", "optimizer", "seq_len", "pack", "num_train_epochs"):
            assert getattr(smoke_spec, field) == getattr(full_spec, field), field
        assert smoke.steps[stage_name].runtime_args[_TRAIN_RESOURCES].replicas == 1


def test_snowball_stage_runs_have_distinct_tracker_ids():
    ids = [
        sft_run_id(spec) for graph in (snowball_stage0(), snowball_stage0(smoke_run())) for spec in graph.specs.values()
    ]
    assert len(set(ids)) == len(ids) == 6
    assert sft_run_id(snowball_stage0().specs["s1-chat"]) == "stage0-snowball-s1-chat-2026.10.22"


def test_prebuilt_stage_cache_is_the_chains_cache_without_the_training_dependency():
    graph = snowball_stage0()
    for stage_name in ("s2-thinking", "s3-terminal"):
        (chain_cache,) = [dep for dep in graph.steps[stage_name].deps if dep.name.startswith("tokenized/")]
        early = stage_chat_cache(graph, stage_name)
        assert (early.name, early.version) == (chain_cache.name, chain_cache.version)
        assert early.deps == (graph.convert,)
    with pytest.raises(ValueError, match="init step"):
        stage_chat_cache(graph, "s1-chat")


def test_snowball_and_axolotl_budgets_agree():
    """The one cross-trainer control: every shared budget value is the same constant."""

    for name in (
        "LEARNING_RATE",
        "ADAM_BETA1",
        "ADAM_BETA2",
        "LR_SCHEDULE",
        "MAX_GRAD_NORM",
        "WEIGHT_DECAY",
        "WARMUP_STEPS",
        "SEQUENCE_LEN",
        "GLOBAL_BATCH",
    ):
        assert hasattr(budget, name), f"budget must own {name}"
    # The Axolotl per-device arithmetic and the Snowball node request both resolve to the
    # single global batch.
    assert (
        budget.PER_DEVICE_BATCH * budget.AXOLOTL_NODES * budget.GPUS_PER_NODE * budget.GRADIENT_ACCUMULATION_STEPS
        == budget.GLOBAL_BATCH
    )
    assert budget.TRAIN_NODES * budget.GPUS_PER_NODE == 64


@pytest.mark.parametrize("family", list(FAMILY_TOKEN_MAPS))
def test_completion_mask_covers_each_assistant_turn_through_its_end_of_turn(family):
    """Levanter's completions-only mask: reasoning, answer and end-of-turn train; headers do not."""
    if not _hf_reachable():
        pytest.skip("HF hub unreachable")
    tokens = FAMILY_TOKEN_MAPS[family]
    repo, revision = _HF_REVISIONS[family]
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        chat_template=shared_template(tokens),
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    trained = tokenizer.decode([i for i, m in zip(rendered["input_ids"], rendered["assistant_masks"], strict=True) if m])
    assert trained == f"{tokens.think_open}r1{tokens.think_close}a1{tokens.turn_close}a2{tokens.turn_close}"

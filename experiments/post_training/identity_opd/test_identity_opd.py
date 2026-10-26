# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import ast
import hashlib
import importlib
import importlib.metadata
import json
import os
import random
import re
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import click
import pytest
import yaml
from click.testing import CliRunner
from marin.rl import skyrl

from experiments.post_training.identity_opd import build_mix, launch
from experiments.post_training.identity_opd.other_lab import FORMS, PRODUCTS, other_lab_questions

# Products the third-person screens ask about; the training questions must not.
SCREEN_PRODUCTS = (
    "ChatGPT",
    "GPT-4",
    "DALL-E",
    "Whisper",
    "Sora",
    "o1",
    "GPT-3.5",
    "Codex",
    "Claude",
    "Gemini",
    "Bard",
    "AlphaFold",
    "BERT",
    "Gemma",
    "Llama",
    "Mixtral",
    "Grok",
    "Copilot",
    "Qwen",
    "DeepSeek",
    "Nemotron",
    "OLMo",
    "Stable Diffusion",
    "Constitutional AI",
)


def test_other_lab_questions_are_unique_and_avoid_the_screens():
    questions = other_lab_questions()
    assert len(questions) == len(PRODUCTS) * len(FORMS)
    assert len({q["id"] for q in questions}) == len({build_mix.normalize(q["text"]) for q in questions}) == 240
    for q in questions:
        assert not build_mix.BRAND.search(q["text"])
        assert not any(re.search(rf"\b{re.escape(p)}\b", q["text"]) for p in SCREEN_PRODUCTS), q["text"]
    makers = dict(PRODUCTS)
    for q in questions[3::4]:
        product, wrong = q["text"].removeprefix("Is ").removesuffix("?").split(" made by ")
        assert makers[product] != wrong


GLAIVE = {
    "id": "c1",
    "tools": json.dumps([{"type": "function", "function": {"name": "get_price", "parameters": {"type": "object"}}}]),
    "conversations": [
        {"from": "system", "value": "You are a function calling AI model. <tools>...</tools>"},
        {"from": "human", "value": "Price of Apple?"},
        {"from": "gpt", "value": '<tool_call>\n{"name": "get_price", "arguments": {"company": "Apple"}}\n</tool_call>'},
        {"from": "tool", "value": '<tool_response>\n{"name": "get_price", "content": {"price": 150}}\n</tool_response>'},
        {"from": "gpt", "value": "Apple is at 150."},
        {"from": "human", "value": "And Microsoft?"},
        {
            "from": "gpt",
            "value": '<tool_call>\n{"name": "get_price", "arguments": {"company": "Microsoft"}}\n</tool_call>',
        },
    ],
}


def test_tool_conversations_become_native_chat_messages():
    messages = build_mix.tool_messages(GLAIVE)
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant", "user", "assistant"]
    assert messages[1] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "get_price", "arguments": '{"company": "Apple"}'}}
        ],
    }
    assert messages[2] == {"role": "tool", "tool_call_id": "call_1", "content": '{"price": 150}'}
    assert messages[5]["tool_calls"][0]["id"] == "call_2"


def test_tool_cuts_are_labelled_by_the_next_assistant_turn():
    cuts = build_mix.tool_cuts(build_mix.tool_messages(GLAIVE))
    assert [(kind, len(prompt)) for kind, prompt in cuts] == [("first_call", 1), ("reply", 3), ("call_after_result", 5)]


def test_bfcl_function_names_and_8_grams_reject_conversations(tmp_path):
    (tmp_path / "bfcl-a.md").write_text(
        "# Task\nwhat is the price of apple stock today in us dollars please\n## Available Functions\n### get_price\n"
    )
    clean = {**GLAIVE, "id": "c2", "tools": GLAIVE["tools"].replace("get_price", "lookup_price")}
    rejected: list = []
    with pytest.raises(ValueError, match="tool call_after_result: 0 conversations"):
        build_mix.tool_rows([GLAIVE, clean], tmp_path, random.Random(0), rejected)
    assert [r["id"] for r in rejected] == ["c1"]
    assert "BFCL function name" in rejected[0]["reason"]


def test_identity_rows_keep_modes_and_never_reuse_a_prompt(monkeypatch):
    monkeypatch.setattr(build_mix, "STEPS", 1)
    monkeypatch.setattr(build_mix, "PER_BATCH", {**build_mix.PER_BATCH, "identity": 4})
    pool = []
    for i in range(6):
        for thinking in (True, False):
            pool.append(
                {
                    "prompt": [{"role": "user", "content": f"who made you, version {i}?"}],
                    "teacher_route": f"identity_{'thinking' if thinking else 'nonthinking'}",
                    "chat_template_kwargs": {"enable_thinking": thinking},
                    "data_source": "pool",
                }
            )
    pool.append({**pool[0], "teacher_route": "preservation_thinking"})
    rejected: list = []
    rows = build_mix.identity_rows(pool, {build_mix.normalize("who made you, version 3?")}, random.Random(0), rejected)
    texts = [r["prompt"][0]["content"] for r in rows]
    assert len(texts) == len(set(texts)) == 4
    assert sorted(r["chat_template_kwargs"]["enable_thinking"] for r in rows) == [False, False, True, True]
    assert all("version 3" not in t for t in texts)
    assert len(rejected) == 2


def test_recipe_routes_cover_the_mix_and_fill_a_batch():
    config = yaml.safe_load(launch.RECIPE.read_text())
    weights = config["data"]["sampling"]["domain_weights"]
    routes = config["teacher_routing"]["opd"]["routes"]
    assert (
        set(weights)
        == set(routes)
        == {f"{family}_{mode}" for family in build_mix.PER_BATCH for mode in ("thinking", "nonthinking")}
    )
    assert sum(weights.values()) == launch.ROLE_PLAN.train_batch_size
    for family, count in build_mix.PER_BATCH.items():
        assert weights[f"{family}_thinking"] + weights[f"{family}_nonthinking"] == count
        assert routes[f"{family}_thinking"]["teacher"] == build_mix.TEACHER[family]


def test_dry_arm_only_changes_lr_warmup_steps_probe_and_saves():
    train, dry = yaml.safe_load(launch.recipe("train")), yaml.safe_load(launch.recipe("dry"))
    assert dry["trainer"]["policy"]["optimizer_config"]["lr"] == 0.0
    assert dry["trainer"]["max_steps"] == 2
    assert dry["trainer"]["mismatch_probe"]["updates"] == 2
    assert dry["trainer"]["policy"]["optimizer_config"]["num_warmup_steps"] == 0
    for key in ("lr", "num_warmup_steps"):
        dry["trainer"]["policy"]["optimizer_config"][key] = train["trainer"]["policy"]["optimizer_config"][key]
    assert dry["generator"].pop("trajectory_retention") == {"sample_fraction": 1.0, "required": True}
    assert "trajectory_retention" not in train["generator"]
    assert dry["trainer"]["callbacks"] == [
        {**callback, "save_steps": 2} if "save_steps" in callback else callback
        for callback in train["trainer"]["callbacks"]
    ]
    for key in ("max_steps", "ckpt_interval", "hf_save_interval", "callbacks", "mismatch_probe"):
        train["trainer"].pop(key, None)
        dry["trainer"].pop(key, None)
    assert train == dry


def test_mix_pin_refuses_other_bytes(tmp_path, monkeypatch):
    for name in (launch.MIX_FILE, launch.PROBE_FILE):
        (tmp_path / name).write_bytes(name.encode())
    monkeypatch.setattr(
        launch,
        "MIX_SHA256",
        {name: hashlib.sha256(name.encode()).hexdigest() for name in (launch.MIX_FILE, launch.PROBE_FILE)},
    )
    launch.verify_mix(str(tmp_path))
    (tmp_path / launch.PROBE_FILE).write_bytes(b"changed")
    with pytest.raises(click.UsageError, match=r"probe\.parquet has sha256"):
        launch.verify_mix(str(tmp_path))
    (tmp_path / launch.PROBE_FILE).unlink()
    with pytest.raises(FileNotFoundError):
        launch.verify_mix(str(tmp_path))


def test_prefixes_are_pinned_to_the_cluster_region(monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-us-east-02a/marin")
    with pytest.raises(click.UsageError, match="MARIN_PREFIX"):
        launch.pin_prefix()
    monkeypatch.delenv("MARIN_PREFIX")
    monkeypatch.setenv("MARIN_TEMP_PREFIX", "s3://marin-us-east-08a")
    with pytest.raises(click.UsageError, match="MARIN_TEMP_PREFIX"):
        launch.pin_prefix()
    monkeypatch.delenv("MARIN_TEMP_PREFIX")
    launch.pin_prefix()
    assert os.environ["MARIN_PREFIX"] == launch.PREFIX
    assert launch.temporary_root(f"{launch.PREFIX}/users/u/checkpoints/x/dry/2026.10.26.1").startswith(
        "s3://marin-rno2a/tmp/"
    )


def test_main_plans_without_submitting_and_prints_region_roots(monkeypatch):
    monkeypatch.delenv("MARIN_PREFIX", raising=False)
    monkeypatch.delenv("MARIN_TEMP_PREFIX", raising=False)
    checked = []
    monkeypatch.setattr(launch, "verify_mix", lambda: checked.append(True))
    result = CliRunner().invoke(launch.main, ["--arm", "dry", "--version", "2026.10.26.1"])
    assert result.exit_code == 0, result.output
    assert checked == [True]
    roots = [line for line in result.output.splitlines() if line.startswith(("output root:", "temporary root:"))]
    assert len(roots) == 4
    assert all(" -> s3://marin-rno2a/" in line or line.startswith("temporary root: s3://marin-rno2a/") for line in roots)
    assert "submitted" not in result.output


V01_TOKENIZER = (
    "~/.cache/huggingface/hub/models--open-athena--Marin-Snowball-67B-A2B-v0.1/snapshots/"
    "adc0078a19a4383bc998fcdb43867b3360baf5f3"
)


def test_identity_teacher_sees_the_template_system_message_on_plain_and_tool_rows():
    path = Path(V01_TOKENIZER).expanduser()
    if not path.exists():
        pytest.skip("v0.1 tokenizer is not in the local Hugging Face cache")
    from transformers import AutoTokenizer  # noqa: PLC0415

    from experiments.post_training.identity_opd import teacher_insertion  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(path)
    plain = build_mix.row([{"role": "user", "content": "Who are you?"}], "identity", False, "test", "p")
    tool_prompt = build_mix.tool_cuts(build_mix.tool_messages(GLAIVE))[-1][1]
    tool = {**build_mix.row(tool_prompt, "tool", True, "test", "t"), "tools": GLAIVE["tools"]}
    rows = [{**r, "chat_template_kwargs": json.dumps(r["chat_template_kwargs"])} for r in (plain, tool)]
    result = teacher_insertion.check(tokenizer, rows, {"identity_nonthinking", "tool_thinking"})
    assert result["rows_checked"] == 2
    plain_teacher = result["examples"]["identity_nonthinking"]["teacher"]
    tool_teacher = result["examples"]["tool_thinking"]["teacher"]
    assert plain_teacher.startswith(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>Reasoning: /nothink<|eot_id|>"
        "<|start_header_id|>system<|end_header_id|>\nYou are Marin Snowball, an AI model created by Open Athena."
    )
    header, rest = tool_teacher.split("<|eot_id|>", 1)
    assert header.startswith("<|begin_of_text|><|start_header_id|>system<|end_header_id|>Reasoning: /think")
    assert '"name": "get_price"' in header
    assert rest.startswith("<|start_header_id|>system<|end_header_id|>\nYou are Marin Snowball")


def _launch_doc(arm, tmp_path) -> Path:
    """Write the arm's launch document as the coordinator writes it, and return its path."""
    spec = SimpleNamespace(
        config_yaml=launch.recipe(arm),
        topology=skyrl.SkyRLTopology(
            num_nodes=launch.NUM_NODES, gpus_per_node=8, gpu_variant="H100", role_plan=launch.ROLE_PLAN
        ),
        retention=skyrl.SkyRLRetentionPolicy(resume_checkpoint_count=1),
        seed=launch.SEED,
    )
    execution = SimpleNamespace(
        cluster=launch.CLUSTER,
        cluster_config=f"lib/iris/config/{launch.CLUSTER}.yaml",
        cpu=32,
        memory="1800GB",
        disk="2TB",
        priority="production",
        max_retries=0,
        target_cluster=launch.CLUSTER,
        parent_cluster_config=launch.IRIS_HUB_CLUSTER_CONFIG,
        wandb_entity="marin-community",
        job_timeout_seconds=launch.JOB_TIMEOUT_SECONDS[arm],
        container_profile=None,
    )
    out, tmp = f"{launch.PREFIX}/users/u/checkpoints/x/{arm}/v", f"{launch.TEMP_PREFIX}/tmp/x/{arm}/v"
    source = skyrl.ResolvedDirectoryDataSource(uri=f"{out}/mix", identity="mix", local_path="/tmp/mix", relative_path="")
    text = skyrl._launch_config_yaml(
        spec,
        execution,
        export_hf=True,
        run_id=f"x-{arm}",
        attempt_id="a",
        model=skyrl.ResolvedModelLocator(
            uri=f"{out}/model",
            identity="model",
            local_path="/tmp/model",
            tokenizer_uri=launch.TOKENIZER_URI,
            tokenizer_revision=launch.TOKENIZER_REVISION,
        ),
        train_data=(replace(source, relative_path=launch.MIX_FILE),),
        validation_data=(replace(source, relative_path=launch.PROBE_FILE),) if arm == "dry" else (),
        output=skyrl.SkyRLOutputPaths(
            checkpoint_root=f"{tmp}/checkpoints",
            export_root=f"{out}/exports",
            attempts_root=f"{tmp}/attempts",
            resolved_config_uri=f"{out}/resolved-launch.yaml",
            terminal_manifest_uri=f"{out}/terminal.json",
        ),
    )
    (tmp_path / "launch.yaml").write_text(text)
    return tmp_path / "launch.yaml"


def _composed_skyrl(arm, tmp_path):
    """The arm's ``skyrl`` config as the coordinator's launcher composes it (``load_launch_config``)."""
    launch_config = pytest.importorskip("cloud.iris.launch_config")
    return launch_config.load_launch_config(_launch_doc(arm, tmp_path)).skyrl


@pytest.mark.parametrize("arm", ["dry", "train"])
def test_arm_composes_through_the_skyrl_launcher(arm, tmp_path):
    """The coordinator's launcher composes the recipe against SkyRL's declared keys and export rules."""
    composed = _composed_skyrl(arm, tmp_path)
    # The served student has no tool-call parser, so "auto" (vLLM's default when a row has tools) is rejected.
    assert composed.generator.tool_choice == "none"
    trainer = composed.trainer
    steps = 2 if arm == "dry" else 20
    assert trainer.max_steps == trainer.hf_save_interval == steps
    assert trainer.export_hf_artifact
    # Megatron's param scheduler asserts warmup < decay steps, and decay steps are at most max_steps.
    assert trainer.policy.optimizer_config.num_warmup_steps < trainer.max_steps
    assert {callback.type: callback.get("save_steps") for callback in trainer.callbacks}["checkpoint"] == steps


def _step2_export(training, monkeypatch, tmp_path):
    """Stub the export launcher's checkpoint reads with a step-2 request as the dry run wrote it; collect submits."""
    export_hf = pytest.importorskip("cloud.iris.export_hf_checkpoint")
    HFExportRequest = pytest.importorskip("skyrl_train.hf_export_schema").HFExportRequest
    placement = training.skyrl.trainer.placement
    checkpoint_root = str(training.artifacts.checkpoint_root)
    request = HFExportRequest(
        step=2,
        checkpoint_base_path=checkpoint_root,
        checkpoint_path=f"{checkpoint_root}/global_step_2",
        export_path=str(training.artifacts.export_root),
        model_path="/tmp/marinskyrl/model_metadata/model",
        model_source_uri=str(training.inputs.model.uri),
        model_source_identity=str(training.inputs.model.identity),
        num_nodes=placement.policy_num_nodes,
        gpus_per_node=placement.policy_num_gpus_per_node,
    )
    submitted = []
    monkeypatch.setattr(export_hf.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(export_hf, "terminal_checkpoint_step", lambda root: 2)
    monkeypatch.setattr(export_hf, "_read_hf_export_request", lambda path: request)
    monkeypatch.setattr(export_hf, "submit_requested_export", lambda spec, command: submitted.append(spec))
    return export_hf, request, submitted


@pytest.mark.parametrize("arm", ["dry", "train"])
def test_terminal_export_config_passes_launch_validation(arm, tmp_path, monkeypatch):
    """The terminal export's launch document, derived by the coordinator from the arm's, passes launch validation.

    Runs ``export_terminal_policy`` on the arm's launch document up to the submit.
    """
    launch_config = pytest.importorskip("cloud.iris.launch_config")
    training_path = _launch_doc(arm, tmp_path)
    training = launch_config.load_launch_config(training_path)
    export_hf, _, submitted = _step2_export(training, monkeypatch, tmp_path)

    export_hf.export_terminal_policy(training_path)

    [spec] = submitted
    exported = launch_config.load_launch_config(Path(spec.launch_config_path))
    assert exported.run.mode == "checkpoint_export"
    assert training.skyrl.trainer.mismatch_probe.enabled == (arm == "dry")
    assert "mismatch_probe" not in exported.skyrl.trainer
    launch_config.validate_launch_config(exported)


def test_export_step_job_is_bounded_by_its_timeout(tmp_path, monkeypatch):
    """The export-only job, derived from the dry arm's resolved launch record, carries the budget timeout."""
    launch_config = pytest.importorskip("cloud.iris.launch_config")
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    export_step = pytest.importorskip("experiments.post_training.identity_opd.export_step")

    training = launch_config.load_launch_config(_launch_doc("dry", tmp_path))
    record = {"config": OmegaConf.to_container(training, resolve=True)}
    launch_doc = export_step.training_launch_doc(record, tmp_path / "training-launch.yaml")
    export_hf, request, submitted = _step2_export(training, monkeypatch, tmp_path)
    args = export_step.export_args(launch_doc, request.checkpoint_path, dry_run=False)
    monkeypatch.setattr(sys, "argv", ["export_hf_checkpoint", *args])

    export_hf.main()

    [spec] = submitted
    exported = launch_config.load_launch_config(Path(spec.launch_config_path))
    assert export_step.EXPORT_TIMEOUT_SECONDS == 2400
    assert exported.iris.timeout == 2400
    assert exported.iris.max_retries == 0
    assert exported.iris.priority == "production"
    assert (exported.iris.allocation.num_nodes, exported.iris.allocation.gpus_per_node) == (4, 8)
    assert (exported.iris.target_cluster, exported.iris.parent_cluster_config) == (
        launch.CLUSTER,
        launch.IRIS_HUB_CLUSTER_CONFIG,
    )
    assert "mismatch_probe" not in exported.skyrl.trainer
    launch_config.validate_launch_config(exported)


SKYRL_OPTIMIZER = Path(__file__).parents[3] / "vendor/skyrl/skyrl-train/skyrl_train/distributed/megatron/optimizer.py"
SKYRL_OPTIMIZER_NAMES = (
    "_GRUG_MUONH_NAME",
    "_GRUG_EMERGING_ROUTES",
    "_grug_muonh_extra",
    "_MegatronParamScheduler",
    "_init_megatron_optim_config",
    "get_megatron_optimizer_param_scheduler",
)


def _megatron_file(monkeypatch, relative_path: str, module_name: str, globals_: dict) -> dict:
    """Run one megatron-core 0.18.0 source file as ``module_name`` with only ``megatron.core.utils`` stubbed."""
    try:
        source = Path(importlib.metadata.distribution("megatron-core").locate_file(relative_path))
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("megatron-core is not installed")
    module = ModuleType(module_name)
    module.__dict__.update(globals_, __package__=module_name.rpartition(".")[0])
    monkeypatch.setitem(sys.modules, module_name, module)
    exec(compile(source.read_text(), str(source), "exec"), module.__dict__)
    return module.__dict__


def _skyrl_megatron_builders(monkeypatch) -> SimpleNamespace:
    """SkyRL's Megatron optimizer-config and LR-scheduler builders, running on Megatron's own classes.

    In the SkyRL venv these are imported. Elsewhere, importing the ``megatron.core`` package needs triton, so the
    builders are taken from the vendored SkyRL source by name and run against megatron-core's own
    ``optimizer_config.py`` and ``optimizer_param_scheduler.py``, with only ``megatron.core.utils`` stubbed.
    """
    try:
        module = importlib.import_module("skyrl_train.distributed.megatron.optimizer")
    except ImportError:
        pass
    else:
        return SimpleNamespace(source="skyrl", **{name: getattr(module, name) for name in SKYRL_OPTIMIZER_NAMES})
    for name in ("megatron", "megatron.core", "megatron.core.optimizer"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    utils = SimpleNamespace(log_single_rank=lambda *args: None, is_te_min_version=lambda *args, **kwargs: False)
    monkeypatch.setitem(sys.modules, "megatron.core.utils", utils)
    config = _megatron_file(
        monkeypatch, "megatron/core/optimizer/optimizer_config.py", "megatron.core.optimizer.optimizer_config", {}
    )
    scheduler = _megatron_file(
        monkeypatch, "megatron/core/optimizer_param_scheduler.py", "megatron.core.optimizer_param_scheduler", {}
    )
    tree = ast.parse(SKYRL_OPTIMIZER.read_text())
    wanted = [
        node
        for node in tree.body
        if getattr(node, "name", None) in SKYRL_OPTIMIZER_NAMES
        or (isinstance(node, ast.Assign) and node.targets[0].id in SKYRL_OPTIMIZER_NAMES)
    ]
    namespace = {
        "Mapping": Mapping,
        "torch": pytest.importorskip("torch"),
        "OptimizerConfig": config["OptimizerConfig"],
        "OptimizerParamScheduler": scheduler["OptimizerParamScheduler"],
    }
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(SKYRL_OPTIMIZER), "exec"), namespace)
    return SimpleNamespace(source="extracted", **{name: namespace[name] for name in SKYRL_OPTIMIZER_NAMES})


@pytest.mark.parametrize("arm", ["dry", "train"])
def test_megatron_optimizer_config_and_scheduler_accept_each_arm(arm, tmp_path, monkeypatch):
    """Megatron checks its optimizer config and LR scheduler at policy init; build both from the composed config.

    SkyRL's decay steps are the run's total training steps, min(steps per epoch, max_steps); the mix has well over
    max_steps batches per epoch, so they equal max_steps here. The scheduler is stepped once per update; whether
    the first update trains at the LR before or after the first step is SkyRL's call order, not checked here.
    """
    torch = pytest.importorskip("torch")
    trainer = _composed_skyrl(arm, tmp_path).trainer
    builders = _skyrl_megatron_builders(monkeypatch)
    settings = trainer.policy.optimizer_config
    config = builders._init_megatron_optim_config(settings, trainer.policy.megatron_config.optimizer_config_kwargs)
    assert (config.optimizer, config.lr, config.min_lr, config.weight_decay, config.clip_grad) == (
        "adam",
        settings.lr,
        0.0,
        0.0,
        0.5,
    )
    assert (config.adam_beta1, config.adam_beta2, config.use_distributed_optimizer) == (0.9, 0.95, True)
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    scheduler = builders.get_megatron_optimizer_param_scheduler(
        optimizer=optimizer, config=settings, num_training_steps=trainer.max_steps
    )
    assert isinstance(scheduler, builders._MegatronParamScheduler)
    applied = []
    for _ in range(trainer.max_steps):
        scheduler.step(1)
        applied.append(optimizer.param_groups[0]["lr"])
    assert applied == pytest.approx([0.0, 0.0] if arm == "dry" else [1e-6, 2e-6] + [3e-6] * 18)

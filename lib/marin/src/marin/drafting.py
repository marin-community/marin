# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone Grug EAGLE3 training through the pinned extraction/training tools.

Run training and serving inside one eight-H100 Iris task. The run directory owns
frozen responses, offline states, completed training checkpoints and serving
exports. A resume restores the last published epoch; an initial draft starts a
new optimizer schedule.
"""

import contextlib
import hashlib
import json
import logging
import math
import os
import runpy
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

from huggingface_hub import hf_hub_download
from iris.cluster.client.job_info import get_job_info
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.transfer import execute_sync, sync_plan

from marin.external_dependencies import VLLM_GPU_RELEASE
from marin.inference.config import (
    ResolvedModelLocator,
    ServedModelConfig,
    SpeculativeMethod,
    SpeculativeServingConfig,
    VllmEngineConfig,
    VllmLauncherType,
    VllmSource,
)
from marin.inference.serve import local_inference

logger = logging.getLogger(__name__)
TOOLKIT_COMMIT = "8b1558baa74cafb6c7440f1dca2ebb9ab71dc054"
TRAINER_COMMIT = "6496b4db007de00dd4741349643d23b23e29dce8"
CODING_SOURCE = "sahil2801/CodeAlpaca-20k"
CODING_REVISION = "152bb5e9a29651266b018106053980070a0521a1"
AUXILIARY_LAYERS = (2, 13, 23)
FINAL_LAYER = 26
DRAFT_VOCABULARY = 32000
CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "optimizer_state_dict.pt",
    "scheduler_state_dict.pt",
    "val_metrics.json",
    "training_state.json",
)
GENERATION = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": None}


@dataclass(frozen=True)
class DraftTrainingConfig:
    """Inputs to a new standalone fit. Target/data/output are storage URLs or local paths.

    ``data`` is prompt JSONL with unique ``id``, canonical problem ``task`` and
    OpenAI ``messages``. Tokenizer revision must be immutable and its bytes must
    match the target. ``initial_draft`` accepts a compatible serving export with
    its original vocabulary-map files. A new output directory is required.
    """

    target: str
    tokenizer: str
    tokenizer_revision: str
    data: str
    output: str
    initial_draft: str | None = None
    epochs: int = 2
    sequence_length: int = 512
    max_new_tokens: int = 128
    holdout_fraction: float = 1 / 3
    learning_rate: float = 5e-5
    muon_learning_rate: float = 5e-4


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _read_json(uri: str) -> Any:
    filesystem, path = filesystem_for(uri)
    with filesystem.open(path, "r") as handle:
        return json.load(handle)


def _write_json(uri: str, value: Any) -> None:
    filesystem, path = filesystem_for(uri)
    filesystem.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
    with filesystem.open(path, "w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def _stage_directory(uri: str, destination: Path) -> None:
    # fsspec nests the source basename when destination already exists.
    filesystem, path = filesystem_for(uri)
    filesystem.get(path, str(destination), recursive=True, batch_size=16)


def _publish_directory(source: Path, destination: str) -> None:
    execute_sync(sync_plan(str(source), destination, delete=False, checksum=False))


def prepare_coding_sample(output: str, *, sample_size: int = 12, revision: str = CODING_REVISION) -> str:
    """Write a deterministic CodeAlpaca prompt sample and its source receipt.

    Deduplicate stripped instruction/input pairs, then rank problem identities
    by a seed42 hash. Original corpus responses are discarded. The training
    workflow generates new responses with the selected target.
    """
    if sample_size < 2:
        raise ValueError("At least two problems are needed for a task-disjoint holdout")
    raw = Path(hf_hub_download(CODING_SOURCE, "code_alpaca_20k.json", repo_type="dataset", revision=revision))
    problems = {}
    for row in json.loads(raw.read_text()):
        problem = [row["instruction"].strip(), row["input"].strip()]
        identity = hashlib.sha256(json.dumps(problem, ensure_ascii=False).encode()).hexdigest()
        problems[identity] = problem
    identities = sorted(problems, key=lambda key: hashlib.sha256(("42:" + key).encode()).digest())[:sample_size]
    if len(identities) < 2:
        raise ValueError("At least two problems are needed for a task-disjoint holdout")
    rows = [
        {
            "id": f"codealpaca:{identity}",
            "task": f"codealpaca:{identity}",
            "messages": [{"role": "user", "content": "\n\n".join(part for part in problems[identity] if part)}],
        }
        for identity in identities
    ]
    content = "".join(json.dumps(row) + "\n" for row in rows)
    filesystem, path = filesystem_for(output)
    if filesystem.exists(path):
        raise FileExistsError(output)
    filesystem.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
    with filesystem.open(path, "w") as handle:
        handle.write(content)
    _write_json(
        output + ".source.json",
        {
            "dataset": CODING_SOURCE,
            "revision": revision,
            "source_sha256": _sha256(raw),
            "sample_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "problem_ids": identities,
            "selection": "unique stripped instruction/input pairs; first N by sha256(42:problem_sha256)",
        },
    )
    return output


def _toolkit(workspace: Path) -> Path:
    archive = workspace / "toolkit.tar.gz"
    urlretrieve(f"https://codeload.github.com/lukedhlee/grugmoe-eagle3/tar.gz/{TOOLKIT_COMMIT}", archive)
    with tarfile.open(archive) as handle:
        handle.extractall(workspace, filter="data")
    return workspace / f"grugmoe-eagle3-{TOOLKIT_COMMIT}"


def _tool(toolkit: Path, script: str, *arguments: str | Path) -> None:
    subprocess.run([sys.executable, str(toolkit / "scripts" / script), *map(str, arguments)], check=True)


def _trainer_python() -> list[str]:
    return [
        "uv",
        "run",
        "--no-project",
        "--python",
        "3.12",
        "--with",
        f"speculators @ git+https://github.com/lukedhlee/speculators.git@{TRAINER_COMMIT}",
        "--with",
        f"hs-connectors @ git+https://github.com/lukedhlee/speculators.git@{TRAINER_COMMIT}#subdirectory=hs_connectors",
        "--with",
        "torch==2.13.0+cu132",
        "--with",
        "transformers==5.16.1",
        "--with",
        "datasets==4.8.5",
        "--index",
        "https://download.pytorch.org/whl/cu132",
        "--index",
        "https://download.pytorch.org/whl/cpu",
        "--index-strategy",
        "unsafe-best-match",
        "python",
    ]


def _engine() -> VllmEngineConfig:
    return VllmEngineConfig(
        launcher=VllmLauncherType.CUDA,
        source=VllmSource.MARIN_FORK,
        max_num_batched_tokens=7168,
        max_num_seqs=32,
        extra_args=("--data-parallel-size", "8", "--enable-expert-parallel", "--gpu-memory-utilization", "0.85"),
    )


def _model(config: DraftTrainingConfig, weights: str) -> ServedModelConfig:
    return ServedModelConfig(
        weights=weights,
        api_model="target",
        tokenizer=config.tokenizer,
        tokenizer_revision=config.tokenizer_revision,
        max_model_len=65536,
        tensor_parallel_size=1,
    )


def _prepare_run(config: DraftTrainingConfig, workspace: Path, root: Path, toolkit: Path) -> dict:
    if not 0 < config.holdout_fraction < 1 or config.epochs < 1 or not 1 <= config.sequence_length <= 32768:
        raise ValueError("Use a nonempty holdout, positive epochs and training sequence length <=32768")
    root.mkdir()
    filesystem, path = filesystem_for(config.data)
    filesystem.get_file(path, str(root / "prompts.jsonl"))
    prompts = [json.loads(line) for line in (root / "prompts.jsonl").read_text().splitlines() if line.strip()]
    if len({row["id"] for row in prompts}) != len(prompts) or len({row["task"] for row in prompts}) < 2:
        raise ValueError("Prompts need unique ids and at least two canonical problem identities")
    if any(not row["messages"] for row in prompts):
        raise ValueError("Every prompt needs messages")
    target_config = _read_json(config.target.rstrip("/") + "/config.json")
    if (target_config["model_type"], target_config["num_hidden_layers"], target_config["hidden_size"]) != (
        "grug_moe",
        FINAL_LAYER,
        2560,
    ):
        raise ValueError("This recipe supports the 26-layer Snowball/Grug EAGLE3 case")
    if target_config["max_position_embeddings"] != 65536:
        raise ValueError("This recipe preserves Snowball's native 65536-token context")
    target_filesystem, target_path = filesystem_for(config.target)
    target_objects = target_filesystem.find(target_path, detail=True)
    target_bytes = sum(info["size"] for info in target_objects.values() if info["type"] == "file")
    if shutil.disk_usage(workspace).free < target_bytes + 20 * 1024**3:
        raise OSError("Insufficient local disk for target staging and the bounded draft run")
    target = workspace / "target"
    _stage_directory(config.target, target)
    tokenizer = Path(hf_hub_download(config.tokenizer, "tokenizer.json", revision=config.tokenizer_revision))
    tokenizer_hash = _sha256(tokenizer)
    if _sha256(target / "tokenizer.json") != tokenizer_hash:
        raise ValueError("Pinned tokenizer bytes differ from the target checkpoint")
    model, engine = _model(config, str(target)), _engine()
    with local_inference(model, engine, num_chips=8) as session:
        _tool(
            toolkit,
            "generate_rollouts.py",
            "--input",
            root / "prompts.jsonl",
            "--out",
            root / "rollouts.jsonl",
            "--target",
            config.target,
            "--url",
            session.model.endpoint.base_url.removesuffix("/v1"),
            "--conc",
            "2",
            "--max-tokens",
            str(config.max_new_tokens),
        )
    (root / "excluded_tasks.txt").write_text("")
    _tool(
        toolkit,
        "prepare_inputs.py",
        "--input",
        root / "rollouts.jsonl",
        "--out",
        root / "inputs.jsonl",
        "--target",
        config.target,
        "--exclude-tasks",
        root / "excluded_tasks.txt",
        "--val-ratio",
        str(config.holdout_fraction),
        "--max-len",
        str(config.sequence_length),
    )
    _tool(toolkit, "make_target_stub.py", target, root / "stub")
    states = root / "states"
    states.mkdir()
    extraction = {
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [*AUXILIARY_LAYERS, FINAL_LAYER]}},
    }
    connector = {
        "kv_connector": "ExampleHiddenStatesConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {"shared_storage_path": str(states), "allow_custom_save_path": True},
    }
    extraction_engine = replace(
        engine,
        max_num_batched_tokens=65536,
        extra_args=(
            *engine.extra_args,
            "--no-enable-chunked-prefill",
            "--speculative-config",
            json.dumps(extraction),
            "--kv-transfer-config",
            json.dumps(connector),
        ),
    )
    with local_inference(model, extraction_engine, num_chips=8) as session:
        _tool(
            toolkit,
            "extract.py",
            "--input",
            root / "inputs.jsonl",
            "--out",
            root / "manifest.jsonl",
            "--hs-dir",
            states,
            "--url",
            session.model.endpoint.base_url.removesuffix("/v1"),
            "--conc",
            "2",
        )
    _tool(
        toolkit,
        "build_dataset.py",
        "--manifest",
        root / "manifest.jsonl",
        "--out",
        root / "data",
        "--hs-out",
        root / "data_states",
        "--max-len",
        str(config.sequence_length),
    )
    job = get_job_info()
    record = {
        "config": asdict(config),
        "tokenizer_sha256": tokenizer_hash,
        "target_config": target_config,
        "target_objects": {
            name: {key: info[key] for key in ("size", "ETag") if key in info}
            for name, info in target_objects.items()
            if info["type"] == "file"
        },
        "prompts_sha256": _sha256(root / "prompts.jsonl"),
        "rollouts_sha256": _sha256(root / "rollouts.jsonl"),
        "generation": {**GENERATION, "max_tokens": config.max_new_tokens, "concurrency": 2},
        "split": json.loads((root / "data/split.json").read_text()),
        "teacher_states": {
            "auxiliary_layers": AUXILIARY_LAYERS,
            "final_layer": FINAL_LAYER,
            "convention": "raw final residual; trainer applies target RMSNorm and learned sigmoid gate",
        },
        "runtime": {
            "toolkit": TOOLKIT_COMMIT,
            "trainer": TRAINER_COMMIT,
            "torch": "2.13.0+cu132",
            "transformers": "5.16.1",
            "datasets": "4.8.5",
            "vllm": asdict(VLLM_GPU_RELEASE),
        },
        "job_id": str(job.job_id) if job else None,
    }
    source_filesystem, source_path = filesystem_for(config.data + ".source.json")
    if source_filesystem.exists(source_path):
        record["data_source"] = _read_json(config.data + ".source.json")
    _write_json(str(root / "run.json"), record)
    _publish_directory(root, config.output)
    return record


def _mapping_paths(root: Path, size: int) -> tuple[Path, Path]:
    return root / "data" / f"d2t-{size}.npy", root / "data" / f"t2d-{size}.npy"


def _initializer(source: Path, root: Path, toolkit: Path) -> Path:
    size = json.loads((source / "config.json").read_text())["draft_vocab_size"]
    d2t, t2d = _mapping_paths(root, size)
    destination = root / "initializer"
    # The pinned toolkit exposes this ordinary Python function, including map validation.
    prepare = runpy.run_path(str(toolkit / "scripts/prepare_warmstart.py"))["prepare"]
    prepare(str(source), str(root / "stub"), str(destination), list(AUXILIARY_LAYERS), str(d2t), str(t2d))
    return destination


def _publish_checkpoint(root: Path, config: DraftTrainingConfig, record: dict, epoch: int) -> None:
    checkpoint = root / "checkpoints" / str(epoch)
    state = json.loads((checkpoint / "training_state.json").read_text())
    metrics = json.loads((checkpoint / "val_metrics.json").read_text())
    if state["epoch"] != epoch or state["local_step"] != 0 or not math.isfinite(metrics["loss_epoch"]):
        raise ValueError("Only completed epochs with finite holdout loss can be published")
    size = json.loads((checkpoint / "config.json").read_text())["draft_vocab_size"]
    d2t, t2d = _mapping_paths(root, size)
    maps = {path.name: _sha256(path) for path in (d2t, t2d)}
    if "vocabulary_maps" in record and record["vocabulary_maps"] != maps:
        raise ValueError("Vocabulary maps changed within a training run")
    for path in (d2t, t2d):
        filesystem, destination = filesystem_for(config.output.rstrip("/") + "/data/" + path.name)
        filesystem.makedirs(destination.rsplit("/", 1)[0], exist_ok=True)
        filesystem.put_file(str(path), destination)
    plan = sync_plan(str(checkpoint), config.output.rstrip("/") + f"/checkpoints/{epoch}", delete=False, checksum=False)
    # Publish native progress last. A partially copied epoch is never selected by run.json.
    copies = tuple(sorted(plan.copies, key=lambda action: action.source.name == "training_state.json"))
    execute_sync(replace(plan, copies=copies))
    record["vocabulary_maps"] = maps
    record["checkpoint"] = {**state, "validation_loss": metrics["loss_epoch"]}
    _write_json(str(root / "run.json"), record)
    _write_json(config.output.rstrip("/") + "/run.json", record)


def restore_training_run(output: str, destination: Path) -> tuple[DraftTrainingConfig, dict]:
    """Stage a completed training checkpoint and restore its native best pointer.

    ``destination`` must be absent. Serving exports alone are not resumable.
    Missing state or changed vocabulary-map bytes fail before trainer launch.
    """
    _stage_directory(output, destination)
    record = json.loads((destination / "run.json").read_text())
    config = DraftTrainingConfig(**record["config"])
    if "checkpoint" not in record:
        raise ValueError("No completed training checkpoint was published; a serving export cannot resume training")
    epoch = record["checkpoint"]["epoch"]
    checkpoint = destination / "checkpoints" / str(epoch)
    for name in CHECKPOINT_FILES:
        if not (checkpoint / name).is_file():
            raise ValueError(f"Incomplete training checkpoint: missing {name}")
    for name, digest in record["vocabulary_maps"].items():
        if _sha256(destination / "data" / name) != digest:
            raise ValueError(f"Saved vocabulary map changed: {name}")
    if json.loads((checkpoint / "training_state.json").read_text()) != {
        key: value for key, value in record["checkpoint"].items() if key != "validation_loss"
    }:
        raise ValueError("Native training progress differs from the published checkpoint")
    # A failed upload can leave a newer numeric directory. The native trainer
    # selects the largest directory name, so keep unpublished epochs outside it.
    for candidate in (destination / "checkpoints").iterdir():
        if candidate.name.isdigit() and int(candidate.name) > epoch:
            candidate.rename(destination / f"unpublished-epoch-{candidate.name}")
    # Object stores preserve numeric checkpoint files, not directory symlinks.
    losses = [
        (int(path.parent.name), json.loads(path.read_text())["loss_epoch"])
        for path in (destination / "checkpoints").glob("[0-9]*/val_metrics.json")
        if int(path.parent.name) <= epoch
    ]
    best_epoch = min(losses, key=lambda pair: pair[1])[0]
    (destination / "checkpoints/checkpoint_best").symlink_to(str(best_epoch), target_is_directory=True)
    return config, record


def _train(root: Path, config: DraftTrainingConfig, record: dict, toolkit: Path, initializer: Path | None) -> None:
    split = record["split"]
    command = [
        *_trainer_python(),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        "1",
        str(toolkit / "scripts/adapt_train_entry.py"),
        "--verifier-name-or-path",
        str(root / "stub"),
        "--data-path",
        str(root / "data"),
        "--hidden-states-path",
        str(root / "data_states"),
        "--save-path",
        str(root / "checkpoints"),
        "--epochs",
        str(config.epochs),
        "--lr",
        str(config.learning_rate),
        "--muon-lr",
        str(config.muon_learning_rate),
        "--total-seq-len",
        str(config.sequence_length),
        "--ttt-steps",
        "3",
        "--seed",
        "42",
        "--hidden-states-dtype",
        "bfloat16",
        "--scheduler-type",
        "cosine",
        "--scheduler-warmup-ratio",
        "0.05",
        "--checkpoint-freq",
        "1",
        "--train-data-ratio",
        str(split["train_ratio"]),
        "--on-missing",
        "raise",
        "--target-layer-ids",
        *map(str, AUXILIARY_LAYERS),
        "--draft-attn-impl",
        "sdpa",
        "--loss-implementation",
        "eager",
        "--num-workers",
        "1",
    ]
    if initializer is None:
        command += ["--num-layers", "1", "--draft-vocab-size", str(DRAFT_VOCABULARY), "--sliding-window", "2048"]
    else:
        size = json.loads((initializer / "config.json").read_text())["draft_vocab_size"]
        d2t, t2d = _mapping_paths(root, size)
        command += ["--from-pretrained", str(initializer), "--d2t-path", str(d2t), "--t2d-path", str(t2d)]
    published = {int(path.name) for path in (root / "checkpoints").glob("[0-9]*") if path.is_dir()}

    def publish_completed() -> None:
        for checkpoint in sorted((root / "checkpoints").glob("[0-9]*"), key=lambda path: int(path.name)):
            if int(checkpoint.name) not in published and all((checkpoint / name).is_file() for name in CHECKPOINT_FILES):
                _publish_checkpoint(root, config, record, int(checkpoint.name))
                published.add(int(checkpoint.name))

    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="1")
    with subprocess.Popen(command, env=environment) as process:
        try:
            while process.poll() is None:
                try:
                    publish_completed()
                except json.JSONDecodeError:
                    # Native metadata may be observed during write_text; retry while the writer is live.
                    logger.debug("Checkpoint metadata is still being written")
                time.sleep(1)
            publish_completed()
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def _export(root: Path, config: DraftTrainingConfig, record: dict, toolkit: Path) -> ResolvedModelLocator:
    epoch = record["checkpoint"]["epoch"]
    relative = f"exports/epoch-{epoch}"
    export = root.parent / "export"
    card = root / "model-card.md"
    card.write_text(
        f"# Standalone EAGLE3 draft\n\nTarget: {config.target}\n\n"
        f"Data: {config.data}; {record['split']['train_rows']} train / {record['split']['val_rows']} holdout rows.\n\n"
        "This bounded fit establishes compatibility. Quality, convergence and speedup are unmeasured.\n"
    )
    subprocess.run(
        [
            *_trainer_python(),
            str(toolkit / "scripts/export_draft.py"),
            "--source",
            str(root / "checkpoints" / str(epoch)),
            "--out",
            str(export),
            "--model-card",
            str(card),
        ],
        check=True,
    )
    _publish_directory(export, config.output.rstrip("/") + "/" + relative)
    locator = ResolvedModelLocator(
        uri=config.output.rstrip("/") + "/" + relative, identity=_sha256(export / "model.safetensors")
    )
    record["export"] = asdict(locator)
    _write_json(config.output.rstrip("/") + "/run.json", record)
    return locator


def train_draft(config: DraftTrainingConfig) -> ResolvedModelLocator:
    """Generate target data, extract states, fit and durably export a new draft.

    Owns tool environments, target staging and checkpoint publication. Call in
    an eight-H100 Iris task with marin-core's CPU extra. Completed epochs are
    resumable even if a later epoch or export fails.
    """
    filesystem, output = filesystem_for(config.output)
    if filesystem.exists(output):
        raise FileExistsError(f"Use a new output for training or resume_draft({config.output!r})")
    with tempfile.TemporaryDirectory(prefix="marin-draft-") as temporary:
        workspace = Path(temporary)
        toolkit = _toolkit(workspace)
        root = workspace / "run"
        record = _prepare_run(config, workspace, root, toolkit)
        initializer = None
        if config.initial_draft is not None:
            initial = workspace / "initial"
            _stage_directory(config.initial_draft, initial)
            size = json.loads((initial / "config.json").read_text())["draft_vocab_size"]
            for destination in _mapping_paths(root, size):
                shutil.copyfile(initial / destination.name, destination)
            initializer = _initializer(initial, root, toolkit)
            record["initial_draft_sha256"] = _sha256(initial / "model.safetensors")
        _train(root, config, record, toolkit, initializer)
        return _export(root, config, record, toolkit)


def resume_draft(output: str) -> ResolvedModelLocator:
    """Resume the last published completed epoch, preserving maps and optimizer state.

    Uses the saved configuration and frozen data. It retargets model construction
    to the staged stub; the native trainer loads learned weights, optimizer,
    scheduler and progress from the original checkpoint root.
    """
    with tempfile.TemporaryDirectory(prefix="marin-draft-resume-") as temporary:
        workspace = Path(temporary)
        root = workspace / "run"
        config, record = restore_training_run(output, root)
        toolkit = _toolkit(workspace)
        epoch = record["checkpoint"]["epoch"]
        if epoch + 1 < config.epochs:
            initializer = _initializer(root / "checkpoints" / str(epoch), root, toolkit)
            _train(root, config, record, toolkit, initializer)
        return _export(root, config, record, toolkit)


@contextlib.contextmanager
def draft_serving(output: str) -> Iterator[tuple[ServedModelConfig, VllmEngineConfig]]:
    """Stage a durable draft export and yield the recorded target/tokenizer pair.

    Pass the pair to Marin's existing local_inference or IrisServiceConfig APIs
    inside an eight-H100 task. The serving export never selects the training stub.
    """
    record = _read_json(output.rstrip("/") + "/run.json")
    config = DraftTrainingConfig(**record["config"])
    locator = ResolvedModelLocator(**record["export"])
    with tempfile.TemporaryDirectory(prefix="marin-draft-serve-") as temporary:
        draft = Path(temporary) / "draft"
        _stage_directory(locator.uri, draft)
        if _sha256(draft / "model.safetensors") != locator.identity:
            raise ValueError("Serving draft bytes differ from the recorded export")
        engine = replace(
            _engine(),
            speculative=SpeculativeServingConfig(
                method=SpeculativeMethod.EAGLE3,
                model=replace(locator, uri=str(draft)),
                num_speculative_tokens=3,
            ),
        )
        yield _model(config, config.target), engine

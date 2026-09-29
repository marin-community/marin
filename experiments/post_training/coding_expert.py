# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train and evaluate a Python coding expert from the latest Snowball SFT checkpoint.

The data stage prepares three disjoint groups before any model-based selection. The
development set can guide the spike. The final set stays sealed until a candidate is
selected. Public code evaluations give a stable comparison with the parent model.

Build the data stage::

    uv run python -m experiments.post_training.coding_expert_data --run

Run ``coding_expert_baseline`` before RL. Then run ``--stage smoke`` for one optimizer update.
Use ``--stage pilot`` only after the smoke has a finite loss and writes an HF export.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

import click
from datasets import Dataset, load_dataset
from fray.types import ResourceConfig
from marin.evaluation.model_config import GenerationConfig, ModelConfig, ResourceHint, ServeConfig
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.execution.remote import remote
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
    IrisSkyRLExecution,
    PinnedHfModel,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRun,
    SkyRLRuntime,
    SkyRLRuntimeProfile,
    SkyRLSpec,
    SkyRLTopology,
    artifact_data_staging_path,
    skyrl_step,
)
from rigging.filesystem.storage_path import StoragePath
from transformers import AutoTokenizer

from experiments.evaluation.models import SNOWBALL_VLLM_ARGS
from experiments.evaluation.pipeline import EvaluationResult, eval_step
from experiments.post_training.skyrl_evaluation import SKYRL_POLICY_LOCATION, skyrl_eval_step

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "snowball-python-expert"
PARENT_MODEL = "open-athena/Grug-67B-A2B-GLM53-RLVR-SFT-2026.09.23"
PARENT_REVISION = "227263f32121ac79ba96de6136b86221e58b05db"
DATASET = "open-r1/verifiable-coding-problems-python_decontaminated-tested-shuffled"
DATASET_REVISION = "98191eb6eefd276b7ebb4eb8d25c4a167cc65605"
SKYRL_RUNTIME_COMMIT = "92fbe5126329c38659b54913424e64c39bbeab89"
SHELLBOX_COMMIT = "ee9e2b9f06a9da1c43dc3b3462460d933f81d32a"
SHELLBOX_REQUIREMENT = (
    "marin-shellbox[shellsim] @ "
    f"git+https://github.com/marin-community/marin.git@{SHELLBOX_COMMIT}#subdirectory=lib/shellbox"
)
SHELLSIM_VERSION = "0.1.20"
SHELLSIM_REQUIREMENT = f"shellsim=={SHELLSIM_VERSION}"

CLUSTER = "cw-us-east-02a"
GPU_VARIANT = "H100"
GPUS_PER_NODE = 8
SEED = 9528
MAX_PROMPT_TOKENS = 4096
MAX_TEST_CASES = 50
MAX_CASE_BYTES = 64 * 1024
MAX_NEW_TOKENS = 4096
TRAIN_ROWS = 2400
DEVELOPMENT_ROWS = 300
FINAL_ROWS = 300
CODE_EVALS = "coding-expert-code"
COLLATERAL_EVALS = "math500,coding-expert-ifbench"
TRAIN_DIRECTORY = "train"
DEVELOPMENT_DIRECTORY = "development"
RUNTIME_PACKAGES_DIRECTORY = "runtime_packages"
PROVENANCE_FILENAME = "provenance.json"
RUNTIME_MODULES = (
    "coding_expert_runtime.py",
    "coding_expert_verifier.py",
)


@dataclass(frozen=True)
class CodeProblem:
    """The identities used to keep related problems in one split."""

    problem_id: str
    source_identity: str | None
    prompt_fingerprint: str


@dataclass(frozen=True)
class CodeDataConfig:
    output_path: str
    dataset: str = DATASET
    dataset_revision: str = DATASET_REVISION
    tokenizer: str = PARENT_MODEL
    tokenizer_revision: str = PARENT_REVISION
    train_rows: int = TRAIN_ROWS
    development_rows: int = DEVELOPMENT_ROWS
    final_rows: int = FINAL_ROWS
    max_prompt_tokens: int = MAX_PROMPT_TOKENS
    max_test_cases: int = MAX_TEST_CASES
    max_case_bytes: int = MAX_CASE_BYTES
    marinskyrl_commit: str = SKYRL_RUNTIME_COMMIT
    shellbox_commit: str = SHELLBOX_COMMIT
    shellsim_version: str = SHELLSIM_VERSION
    seed: int = SEED


@dataclass(frozen=True)
class Scale:
    name: str
    max_steps: int
    checkpoint_interval: int
    eval_interval: int
    temporary_storage_ttl_days: int


SCALES = {
    "smoke": Scale("smoke", max_steps=1, checkpoint_interval=1, eval_interval=-1, temporary_storage_ttl_days=1),
    "pilot": Scale("pilot", max_steps=4, checkpoint_interval=4, eval_interval=4, temporary_storage_ttl_days=7),
    "train": Scale("train", max_steps=40, checkpoint_interval=10, eval_interval=10, temporary_storage_ttl_days=14),
}

ROLE_PLAN = SkyRLRolePlan(
    colocate_all=False,
    policy_num_nodes=4,
    policy_num_gpus_per_node=GPUS_PER_NODE,
    num_inference_engines=1,
    inference_engine_tensor_parallel_size=1,
    inference_engine_pipeline_parallel_size=1,
    inference_engine_data_parallel_size=GPUS_PER_NODE,
    inference_engine_expert_parallel_size=GPUS_PER_NODE,
    train_batch_size=64,
    policy_mini_batch_size=64,
    micro_train_batch_size_per_gpu=1,
    n_samples_per_prompt=4,
)


def normalized_prompt_fingerprint(prompt: str) -> str:
    """Return a stable fingerprint after Unicode and white-space normalization."""
    normalized = " ".join(unicodedata.normalize("NFKC", prompt).casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def partition_problems(
    problems: Iterable[CodeProblem], split_sizes: Mapping[str, int], seed: int
) -> dict[str, tuple[CodeProblem, ...]]:
    """Select exact split sizes while keeping source and prompt groups together."""
    problem_list = list(problems)
    if len({problem.problem_id for problem in problem_list}) != len(problem_list):
        raise ValueError("problem_id values must be unique")
    if any(size <= 0 for size in split_sizes.values()):
        raise ValueError("split sizes must be positive")

    parents = list(range(len(problem_list)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = root(left)
        right_root = root(right)
        if left_root != right_root:
            parents[right_root] = left_root

    group_owner: dict[tuple[str, str], int] = {}
    for index, problem in enumerate(problem_list):
        keys = [("prompt", problem.prompt_fingerprint)]
        if problem.source_identity:
            keys.append(("source", problem.source_identity))
        for key in keys:
            owner = group_owner.get(key)
            if owner is None:
                group_owner[key] = index
            else:
                union(index, owner)

    components: dict[int, list[CodeProblem]] = defaultdict(list)
    for index, problem in enumerate(problem_list):
        components[root(index)].append(problem)
    ordered = sorted(
        components.values(),
        key=lambda component: hashlib.sha256(
            f"{seed}:".encode() + ",".join(sorted(problem.problem_id for problem in component)).encode()
        ).digest(),
    )

    selected: dict[str, tuple[CodeProblem, ...]] = {}
    remaining = ordered
    for split, target_size in split_sizes.items():
        chosen: list[list[CodeProblem]] = []
        unchosen: list[list[CodeProblem]] = []
        count = 0
        for component in remaining:
            if count + len(component) <= target_size:
                chosen.append(component)
                count += len(component)
            else:
                unchosen.append(component)
        if count != target_size:
            raise ValueError(f"could not fill {split!r} with {target_size} grouped problems; selected {count}")
        selected[split] = tuple(problem for component in chosen for problem in component)
        remaining = unchosen
    return selected


def _source_identity(row: Mapping[str, Any]) -> str | None:
    source = row.get("source")
    in_source_id = row.get("in_source_id")
    metadata = row.get("metadata")
    problem_url = metadata.get("problem_url") if isinstance(metadata, Mapping) else None
    value = in_source_id or problem_url
    if not isinstance(source, str) or not isinstance(value, str) or not value:
        return None
    return f"{source}:{value}"


def _percentile(values: list[int], fraction: float) -> int:
    return sorted(values)[round((len(values) - 1) * fraction)]


def _extract_python_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        raise ValueError("missing_python_code")
    return match.group(1).strip() + "\n"


def _normalized_cases(verification_info: Any, max_test_cases: int, max_case_bytes: int) -> list[dict[str, str]]:
    if not isinstance(verification_info, Mapping) or verification_info.get("language") != "python":
        raise ValueError("invalid_verification_info")
    raw_cases = verification_info.get("test_cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("missing_test_cases")
    if len(raw_cases) > max_test_cases:
        raise ValueError("too_many_test_cases")

    cases = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, Mapping):
            raise ValueError("invalid_test_case")
        if raw_case.get("type") != "stdin_stdout" or raw_case.get("fn_name") is not None:
            raise ValueError("unsupported_test_case")
        test_input = raw_case.get("input")
        expected = raw_case.get("output")
        if not isinstance(test_input, str) or not isinstance(expected, str):
            raise ValueError("invalid_test_case")
        if max(len(test_input.encode()), len(expected.encode())) > max_case_bytes:
            raise ValueError("test_case_too_large")
        cases.append({"input": test_input, "output": expected})
    return cases


def _task_instruction(problem: str) -> str:
    return (
        f"{problem.rstrip()}\n\n"
        "Use the Bash tool to inspect and run files in the workspace. "
        "Write the complete executable Python program to /workspace/solution.py. "
        "The program must read standard input and write standard output."
    )


def _task_toml(problem_id: str, source: str) -> str:
    return f"""\
version = "1.0"

[metadata]
problem_id = {json.dumps(problem_id)}
source = {json.dumps(source)}

[agent]
timeout_sec = 600.0

[verifier]
timeout_sec = 120.0

[verifier.env]
MARIN_CODING_EXPERT_VERIFY = "1"

[environment]
allow_internet = false
workdir = "/workspace"
"""


def _write_task(root: Path, row: Mapping[str, Any]) -> None:
    task = root / str(row["problem_id"])
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "instruction.md").write_text(_task_instruction(str(row["problem"])))
    (task / "task.toml").write_text(_task_toml(str(row["problem_id"]), str(row["source"])))
    (task / "environment" / ".keep").write_text("")
    (task / "tests" / "cases.json").write_text(json.dumps(row["test_cases"], separators=(",", ":")))
    (task / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n")


def _copy_installed_package(package: str, destination: Path) -> None:
    spec = importlib.util.find_spec(package)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"could not find installed package {package!r}")
    source = Path(spec.origin).parent
    shutil.copytree(source, destination / package, symlinks=False)


def _bundle_runtime(root: Path) -> None:
    runtime = root / RUNTIME_PACKAGES_DIRECTORY
    runtime.mkdir()
    _copy_installed_package("shellbox", runtime)
    _copy_installed_package("shellsim", runtime)
    module_root = Path(__file__).parent
    for module in RUNTIME_MODULES:
        shutil.copy2(module_root / module, runtime / module)


def _preflight_gold_solution(row: Mapping[str, Any]) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="coding-expert-preflight-") as temporary:
        root = Path(temporary)
        solution = root / "solution.py"
        cases = root / "cases.json"
        solution.write_text(str(row["gold_source"]))
        cases.write_text(json.dumps(row["test_cases"], separators=(",", ":")))
        verifier = Path(__file__).with_name("coding_expert_verifier.py")
        try:
            result = subprocess.run(
                [sys.executable, str(verifier), str(solution), str(cases)],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return False, "shellsim_preflight_timeout"
        except subprocess.CalledProcessError:
            return False, "shellsim_preflight_error"
        payload = json.loads(result.stdout)
        if int(payload["reward"]) != 1:
            return False, "shellsim_gold_failure"
        return True, ""


def prepare_code_data(config: CodeDataConfig) -> None:
    """Build private Harbor tasks and a pinned Shellbox runtime bundle."""
    if (
        config.marinskyrl_commit != SKYRL_RUNTIME_COMMIT
        or config.shellbox_commit != SHELLBOX_COMMIT
        or config.shellsim_version != SHELLSIM_VERSION
    ):
        raise ValueError("data builder runtime pins do not match the experiment")
    source = cast(Dataset, load_dataset(config.dataset, split="train", revision=config.dataset_revision))
    tokenizer = cast(Any, AutoTokenizer.from_pretrained(config.tokenizer, revision=config.tokenizer_revision))

    rows_by_id: dict[str, dict[str, Any]] = {}
    problems: list[CodeProblem] = []
    rejected: Counter[str] = Counter()
    prompt_tokens: list[int] = []
    source_counts: Counter[str] = Counter()
    for raw_row in source:
        row = cast(dict[str, Any], raw_row)
        try:
            problem_id = row["problem_id"]
            problem = row["problem"].strip()
            if (
                row["task_type"] != "verifiable_code"
                or not isinstance(problem_id, str)
                or re.fullmatch(r"[A-Za-z0-9_.-]+", problem_id) is None
                or not problem
            ):
                raise ValueError("invalid_problem")
            cases = _normalized_cases(row["verification_info"], config.max_test_cases, config.max_case_bytes)
            gold_source = _extract_python_code(row["gold_standard_solution"])
            prompt = _task_instruction(problem)
            token_count = len(tokenizer(prompt, add_special_tokens=False).input_ids)
            if token_count > config.max_prompt_tokens:
                raise ValueError("prompt_too_long")
            if problem_id in rows_by_id:
                raise ValueError("duplicate_problem_id")
        except (KeyError, TypeError, ValueError) as error:
            rejected[str(error)] += 1
            continue

        fingerprint = normalized_prompt_fingerprint(problem)
        identity = _source_identity(row)
        source_name = str(row["source"])
        rows_by_id[problem_id] = {
            "problem_id": problem_id,
            "problem": problem,
            "source": source_name,
            "source_identity": identity or "",
            "prompt_fingerprint": fingerprint,
            "test_cases": cases,
            "gold_source": gold_source,
        }
        prompt_tokens.append(token_count)
        source_counts[source_name] += 1

    candidate_rows = list(rows_by_id.values())
    with ThreadPoolExecutor(max_workers=8) as executor:
        preflight = executor.map(_preflight_gold_solution, candidate_rows)
        for row, (passed, reason) in zip(candidate_rows, preflight, strict=True):
            if not passed:
                rejected[reason] += 1
                del rows_by_id[str(row["problem_id"])]
                continue
            problems.append(
                CodeProblem(
                    str(row["problem_id"]),
                    str(row["source_identity"]) or None,
                    str(row["prompt_fingerprint"]),
                )
            )

    splits = partition_problems(
        problems,
        {"final": config.final_rows, "development": config.development_rows, "train": config.train_rows},
        config.seed,
    )
    emitted: dict[str, list[dict[str, Any]]] = {}
    for split, selected in splits.items():
        split_rows = []
        for selected_problem in selected:
            row = rows_by_id[selected_problem.problem_id]
            split_rows.append(row)
        emitted[split] = split_rows

    provenance = {
        "schema_version": 2,
        "source": {"dataset": config.dataset, "revision": config.dataset_revision, "rows": len(source)},
        "runtime": {
            "marinskyrl_commit": config.marinskyrl_commit,
            "shellbox_commit": config.shellbox_commit,
            "shellsim_version": config.shellsim_version,
            "environment": "coding_expert_runtime:CodingExpertShellSimEnvironment",
            "agent": "shellbox.agent:BashAgent",
            "agent_bridge": "coding_expert_runtime registers BashAgent as Harbor oracle",
            "verifier": "fresh ShellSim Python process per private case",
            "reward_mode": "binary",
        },
        "parent": {"model": config.tokenizer, "revision": config.tokenizer_revision},
        "selection": {
            "seed": config.seed,
            "max_prompt_tokens": config.max_prompt_tokens,
            "max_test_cases": config.max_test_cases,
            "max_case_bytes": config.max_case_bytes,
            "split_order": ["final", "development", "train"],
            "group_keys": ["normalized_prompt_sha256", "source_and_in_source_id_or_problem_url"],
            "gold_preflight": "all selected candidates pass all private cases in ShellSim",
        },
        "counts": {
            "accepted_after_shellsim_preflight": len(problems),
            "rejected": dict(sorted(rejected.items())),
            "available_by_source": dict(sorted(source_counts.items())),
            "emitted": {split: len(rows) for split, rows in emitted.items()},
        },
        "prompt_tokens": {
            "min": min(prompt_tokens),
            "p50": _percentile(prompt_tokens, 0.5),
            "p95": _percentile(prompt_tokens, 0.95),
            "max": max(prompt_tokens),
        },
        "contamination": {
            "upstream_release": "decontaminated-tested-shuffled",
            "parent_sft_overlap": "unknown: parent SFT source problem IDs are not published",
            "split_isolation": "exact normalized prompts and source identities do not cross splits",
        },
        "split_problem_id_sha256": {
            split: hashlib.sha256("\n".join(sorted(problem.problem_id for problem in selected)).encode()).hexdigest()
            for split, selected in splits.items()
        },
    }

    with tempfile.TemporaryDirectory(prefix=f"{EXPERIMENT_NAME}-") as temporary:
        root = Path(temporary)
        for split, rows in emitted.items():
            for row in rows:
                _write_task(root / split, row)
        _bundle_runtime(root)
        (root / PROVENANCE_FILENAME).write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
        StoragePath(config.output_path).upload_from(f"{root}/", recursive=True)
    logger.info(
        "Published %d train, %d development, and %d final rows",
        config.train_rows,
        config.development_rows,
        config.final_rows,
    )


def data_step(version: str) -> ArtifactStep[Artifact]:
    """Build the pinned coding data artifact."""
    return ArtifactStep(
        name=user_owned_name(f"documents/{EXPERIMENT_NAME}"),
        version=version,
        artifact_type=Artifact,
        run=remote(
            prepare_code_data,
            resources=ResourceConfig.with_cpu(cpu=8, ram="64g", disk="32g"),
            pip_packages=[SHELLBOX_REQUIREMENT, SHELLSIM_REQUIREMENT],
        ),
        build_config=lambda ctx: CodeDataConfig(output_path=ctx.output_path),
    )


def rl_config_yaml(scale: Scale, runtime_packages_path: str) -> str:
    """Return the Shellbox terminal-bench recipe for one scale."""
    return f"""\
entrypoint: terminal_bench

config_groups:
  terminal_bench_config: terminal_bench

context_budget:
  request_window_tokens: 16384
  max_new_tokens_per_turn: {MAX_NEW_TOKENS}
  max_turns: 8

terminal_bench:
  harbor:
    name: oracle
    store_all_messages: true
    extra_body:
      chat_template_kwargs:
        enable_thinking: false
    collect_rollout_details: false
    upload_agent_logs: false
    override_timeout_sec: 600
    override_cpus: 1
    override_memory_mb: 512
    override_storage_mb: 128
    import_path: coding_expert_runtime:CodingExpertShellSimEnvironment
    env_network_policy: deny
    verifier_override_timeout_sec: 120
    max_retries: 0
    n_concurrent_trials: 32
    log_level: INFO
    enable_reward_shaping: false
    enable_error_classification: true
    passthrough_exceptions:
      - AgentTimeoutError
      - TurnCapExhaustedError
    mask_exceptions:
      - EnvironmentStartTimeoutError
      - VerifierTimeoutError
      - RewardFileNotFoundError
      - RewardFileEmptyError
      - VerifierOutputParseError
      - VerifierRuntimeError
    default_error_treatment: zero
  model_info: {{}}
  archiving:
    enabled: false
  trace_upload:
    enabled: false

trainer:
  strategy: megatron
  flash_attn: false
  use_sample_packing: false
  offload_optimizer_during_rollouts: true
  gradient_checkpointing: true
  algorithm:
    advantage_estimator: grpo
    use_kl_loss: false
  epochs: 1
  max_steps: {scale.max_steps}
  update_epochs_per_batch: 1
  eval_batch_size: 64
  micro_forward_batch_size_per_gpu: 1
  eval_before_train: {str(scale.eval_interval > 0).lower()}
  eval_interval: {scale.eval_interval}
  ckpt_interval: {scale.checkpoint_interval}
  resume_mode: latest
  logger: console
  project_name: marin-coding-expert
  hf_hub_repo_id: null
  policy:
    optimizer_config:
      lr: 5.0e-7
      max_grad_norm: 1.0
    megatron_config:
      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 2
      context_parallel_size: 1
      expert_model_parallel_size: 8
      expert_tensor_parallel_size: 1
      optimizer_checkpoint_sharding_type: dp_reshardable
      ddp_config:
        overlap_grad_reduce: true
        overlap_param_gather: true
        grad_reduce_in_fp32: false
generator:
  backend: vllm
  model_dtype: bfloat16
  vllm_attention_backend: FLASH_ATTN
  gpu_memory_utilization: 0.75
  max_num_batched_tokens: 16384
  enforce_eager: false
  run_engines_locally: true
  weight_sync_backend: nccl
  enable_http_endpoint: true
  engine_init_kwargs:
    moe_backend: triton
    enable_auto_tool_choice: true
    tool_call_parser: hermes
  sampling_params:
    temperature: 1.0
    top_p: 1.0

data:
  kind: tasks
  train_data: []
  val_data: []

extra_env:
  PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
  PYTHONPATH: {runtime_packages_path}
"""


def evaluation_model(name: str, location: str, revision: str | None) -> ModelConfig:
    """Serve a pinned or resolved Snowball checkpoint for evaluation."""
    return ModelConfig(
        name=name,
        location=location,
        revision=revision,
        tokenizer=PARENT_MODEL,
        tokenizer_revision=PARENT_REVISION,
        apply_chat_template=True,
        resource_hint=ResourceHint(gpu={GPU_VARIANT: 8}, memory="512g"),
        serve=ServeConfig(
            tensor_parallel_size=1,
            data_parallel_size=8,
            max_model_len=32768,
            max_num_batched_tokens=8192,
            max_num_seqs=32,
            auto_overrides=False,
            vllm_extra_args=SNOWBALL_VLLM_ARGS,
        ),
        generation=GenerationConfig(
            max_gen_toks=8192,
            extra_gen_kwargs={"skip_special_tokens": "false", "repetition_penalty": "1.1"},
        ),
    )


def _baseline_evaluation_step(version: str, evals: str) -> ArtifactStep[EvaluationResult]:
    return eval_step(
        evaluation_model(f"{EXPERIMENT_NAME}-parent", PARENT_MODEL, PARENT_REVISION),
        evals,
        version=version,
        accelerator="H100x8",
        submission_cluster=CLUSTER,
        federated_cluster=CLUSTER,
    )


def baseline_step(version: str) -> ArtifactStep[EvaluationResult]:
    return _baseline_evaluation_step(version, CODE_EVALS)


def baseline_collateral_step(version: str) -> ArtifactStep[EvaluationResult]:
    """Evaluate parent math and instruction-following retention."""
    return _baseline_evaluation_step(version, COLLATERAL_EVALS)


def rl_step(data: ArtifactStep[Artifact], scale: Scale, version: str) -> ArtifactStep[SkyRLRun]:
    """Build one coding RL run from the pinned parent and data artifact."""
    runtime_packages_path = str(PurePosixPath(artifact_data_staging_path(data)) / RUNTIME_PACKAGES_DIRECTORY)
    return skyrl_step(
        SkyRLSpec(
            name=user_owned_name(f"checkpoints/{EXPERIMENT_NAME}-{scale.name}"),
            version=version,
            config_yaml=rl_config_yaml(scale, runtime_packages_path),
            runtime=SkyRLRuntime(profile=SkyRLRuntimeProfile.MEGATRON, commit=SKYRL_RUNTIME_COMMIT),
            model=PinnedHfModel(
                repository=PARENT_MODEL,
                revision=PARENT_REVISION,
                tokenizer_repository=PARENT_MODEL,
                tokenizer_revision=PARENT_REVISION,
            ),
            train_data=(ArtifactDataSource(data, relative_path=TRAIN_DIRECTORY),),
            validation_data=(ArtifactDataSource(data, relative_path=DEVELOPMENT_DIRECTORY),),
            topology=SkyRLTopology(
                num_nodes=5,
                gpus_per_node=GPUS_PER_NODE,
                gpu_variant=GPU_VARIANT,
                role_plan=ROLE_PLAN,
            ),
            retention=SkyRLRetentionPolicy(
                resume_checkpoint_count=2,
                temporary_storage_ttl_days=scale.temporary_storage_ttl_days,
            ),
            seed=SEED,
        ),
        IrisSkyRLExecution(
            cluster=CLUSTER,
            cluster_config=f"lib/iris/config/{CLUSTER}.yaml",
            cpu=32,
            memory="512GB",
            disk="2TB",
            priority="interactive",
            max_retries=0 if scale.name == "smoke" else 1,
            target_cluster=CLUSTER,
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=12 if scale.name == "smoke" else 72,
            wandb_entity=None,
        ),
        export_hf=True,
    )


def _candidate_evaluation_step(
    trained: ArtifactStep[SkyRLRun], scale: Scale, version: str, evals: str
) -> ArtifactStep[EvaluationResult]:
    model = evaluation_model(
        f"{EXPERIMENT_NAME}-{scale.name}",
        SKYRL_POLICY_LOCATION,
        None,
    )
    return skyrl_eval_step(
        trained,
        model,
        evals,
        version=version,
        accelerator="H100x8",
        submission_cluster=CLUSTER,
        federated_cluster=CLUSTER,
    )


def candidate_evaluation_step(
    trained: ArtifactStep[SkyRLRun], scale: Scale, version: str
) -> ArtifactStep[EvaluationResult]:
    return _candidate_evaluation_step(trained, scale, version, CODE_EVALS)


def candidate_collateral_evaluation_step(
    trained: ArtifactStep[SkyRLRun], scale: Scale, version: str
) -> ArtifactStep[EvaluationResult]:
    """Evaluate candidate math and instruction-following retention."""
    return _candidate_evaluation_step(trained, scale, version, COLLATERAL_EVALS)


@click.command(help=__doc__)
@click.option(
    "--stage",
    type=click.Choice(("smoke", "pilot", "train", "candidate-evaluation", "full")),
    default="smoke",
    show_default=True,
)
@click.option("--scale", type=click.Choice(tuple(SCALES)), default="pilot", show_default=True)
@rl_build_options
def main(stage: str, scale: str) -> ArtifactStep | dict[str, ArtifactStep]:
    data_version = resolve_version(f"documents/{EXPERIMENT_NAME}", None)
    version = resolve_version(EXPERIMENT_NAME, None)
    data = data_step(data_version)
    selected_scale = SCALES[stage] if stage in SCALES else SCALES[scale]
    trained = rl_step(data, selected_scale, version)
    if stage in SCALES:
        return trained
    evaluation = candidate_evaluation_step(trained, selected_scale, version)
    collateral = candidate_collateral_evaluation_step(trained, selected_scale, version)
    if stage == "candidate-evaluation":
        return {"code": evaluation, "collateral": collateral}
    return {
        "baseline-code": baseline_step(version),
        "baseline-collateral": baseline_collateral_step(version),
        "candidate-code": evaluation,
        "candidate-collateral": collateral,
    }


if __name__ == "__main__":
    main()

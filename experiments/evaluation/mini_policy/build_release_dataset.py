# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build nested TaskCompendium validation sets and the unchanged-policy task index."""

import asyncio
import hashlib
import json
import os
import shutil
import string
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from rigging.filesystem.buckets import filesystem_for
from rolloutengine.models import MachineRuntimeSpec, RolloutLimits, TaskRolloutSpec
from shellbox.machine import CommandSemantics, NetworkPolicy, PackageSpec
from taskcompendium.convert.script_grader import grade_script, script_package
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    PlainText,
    ResourceGroups,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.resources import inline_resource

from experiments.evaluation.mini_policy.release_grader import grade_response

BASE = "s3://marin-us-east-02a/marin/experiments/mini-eval-policy/dataset-release-20261030"
ROOT = Path(__file__).resolve().parents[3]


def read(url):
    fs, path = filesystem_for(url)
    return fs.cat(path)


def read_json(url):
    return json.loads(read(url))


def original_prompt(spec, benchmark):
    if benchmark == "truthfulqa":
        doc = spec["doc"]
        choices = doc["mc2_targets"]["choices"]
        if len(choices) > len(string.ascii_uppercase):
            raise ValueError("TruthfulQA choices exceed the published single-letter interface")
        prompt = doc["question"] + "\n\n" + "\n".join(f"{string.ascii_uppercase[i]}. {v}" for i, v in enumerate(choices))
        return [
            {"role": "user", "content": prompt + "\n\nChoose one truthful answer. End with its letter in \\boxed{...}."}
        ]
    prompt = spec["requests"][0]["prompt"]
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    if not isinstance(prompt, list) or any(not isinstance(m, dict) or not isinstance(m["content"], str) for m in prompt):
        raise ValueError(f"Unsupported public source prompt: {benchmark}")
    messages = []
    for message in prompt:
        if set(message) - {"role", "content", "reasoning_content"}:
            raise ValueError(f"Unsupported source conversation fields: {benchmark}")
        if message.get("reasoning_content"):
            raise ValueError(f"Source contains unrepresentable provider reasoning: {benchmark}")
        messages.append({"role": message["role"], "content": message["content"]})
    return messages


def environment(packages):
    return EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS,
        capabilities=("shell", "filesystem", "process"),
        package_spec=PackageSpec(pypi_packages=tuple(packages)),
    )


def smoke_response(benchmark, spec):
    if benchmark == "truthfulqa":
        return string.ascii_uppercase[spec["doc"]["mc2_targets"]["labels"].index(1)]
    if benchmark == "NUPA":
        return str(spec["target"])
    if benchmark == "gsm8k-0shot":
        return "\\boxed{" + spec["doc"]["answer"].rsplit("#### ", 1)[-1] + "}"
    if benchmark == "triviaqa":
        return spec["doc"]["answer"]["aliases"][0]
    if benchmark == "mrcr":
        value = str(spec["target"])
        nonce = spec["grader"]["reference"]["nonce"]
        return value if value.startswith(nonce) else nonce + value
    if benchmark == "cruxeval":
        return "[ANSWER]" + str(spec["doc"]["output"]) + "[/ANSWER]"
    if benchmark == "mbppplus":
        problem = spec["grader"]["reference"]["problem"]
        return "```python\n" + "\n".join(problem.get("test_imports", [])) + "\n" + problem["code"] + "\n```"
    if benchmark == "humanevalplus":
        problem = spec["grader"]["reference"]["problem"]
        return "```python\n" + problem["prompt"] + problem["canonical_solution"] + "\n```"
    if benchmark == "olympiadbench":
        return "\\boxed{" + spec["grader"]["reference"]["answers"][0] + "}"
    if benchmark in ("math500", "aime24"):
        return "\\boxed{" + str(spec["target"]) + "}"
    if benchmark in ("gpqa-diamond", "mmlu-pro"):
        return str(spec["target"])
    if benchmark == "ifbench":
        return None
    raise ValueError(benchmark)


async def main():
    for name in ("evalchemy", "harbor"):
        assert read_json(BASE + "/" + name + "-complete.json")["complete"]
    selection = read_json(BASE + "/selection.json")
    tasks = selection["selection"]["840"]
    micro = selection["selection"]["210"]
    assert tasks[:210] == micro and len(tasks) == 840
    profile_path = ROOT / "experiments/evaluation/mini_policy/runtime-configs/validation-profile.yaml"
    profile = yaml.safe_load(profile_path.read_text())
    assert profile["model"]["total_context_length"] == 65536 and profile["model"]["max_output_tokens_per_turn"] == 16384
    assert profile["model"]["thinking"] is True
    runtime = read_json(BASE + "/runtime-upload.json")
    base_url = (
        "https://huggingface.co/datasets/"
        + runtime["repo"]
        + "/resolve/"
        + runtime["runtime_revision"]
        + "/"
        + runtime["asset_path"]
        + "/"
    )
    manifest = read_json(BASE + "/" + runtime["asset_path"] + "/manifest.json")
    package_names = {
        "evalchemy": "evalchemy[ifbench,mrcr,nupa5k-loose]",
        "marin_rigging": "marin-rigging",
        "marin_finestore": "marin-finestore",
    }
    requirements = read(BASE + "/" + runtime["asset_path"] + "/requirements.txt").decode()
    packages = [line.strip() for line in requirements.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    for item in manifest:
        if not item["file"].endswith(".whl"):
            continue
        name = next(value for prefix, value in package_names.items() if item["file"].startswith(prefix + "-"))
        packages.append(name + " @ " + base_url + item["file"] + "#sha256=" + item["sha256"])
    grading_environment = environment(packages)
    metric_resource = inline_resource(
        "marin/evaluation/metric_selection.py",
        (ROOT / "lib/marin/src/marin/evaluation/metric_selection.py").read_bytes(),
    )
    imports = (
        inline_resource("marin/__init__.py", b""),
        inline_resource("marin/evaluation/__init__.py", b""),
        metric_resource,
    )
    grader_files = grade_script(ROOT / "experiments/evaluation/mini_policy/release_grader.py", *imports)
    output = Path(os.environ["IRIS_OUTPUT_DIR"]) / "release"
    output.mkdir(parents=True, exist_ok=True)
    (output / "runtime").mkdir(exist_ok=True)
    shutil.copy2(profile_path, output / "runtime/validation-profile.yaml")
    (output / "runtime/grading-requirements.json").write_text(json.dumps(packages, indent=2))
    (output / "runtime/release_grader.py").write_bytes(
        (ROOT / "experiments/evaluation/mini_policy/release_grader.py").read_bytes()
    )
    (output / "runtime/metric_selection.py").write_bytes(
        (ROOT / "lib/marin/src/marin/evaluation/metric_selection.py").read_bytes()
    )
    (output / "runtime/harbor_import.py").write_bytes(
        (ROOT / "vendor/skyrl/skyrl-train/skyrl_train/dataset/harbor_import.py").read_bytes()
    )
    profile_overrides = {
        "trainer": {"max_prompt_length": 49152},
        "generator": {
            "max_input_length": 49152,
            "engine_init_kwargs": {"max_model_len": 65536},
            "sampling_params": {"max_generate_length": 16384},
            "eval_sampling_params": {"max_generate_length": 16384},
            "chat_template_kwargs": {"enable_thinking": True},
            "enforce_eager": False,
        },
    }
    (output / "runtime/skyrl-overrides.yaml").write_text(yaml.safe_dump(profile_overrides, sort_keys=False))
    pi = {
        "name": "pi",
        "override_timeout_sec": 1800,
        "override_setup_timeout_sec": 360,
        "kwargs": {
            "model_info": {"max_input_tokens": 65536, "max_output_tokens": 16384},
            "thinking": "high",
            "thinking_format": "chat-template",
        },
    }
    pi["kwargs"]["version"] = profile["agentic"]["version"]
    (output / "runtime/harbor-agent.yaml").write_text(yaml.safe_dump(pi, sort_keys=False))
    docs = {}
    harbor = {}
    parent = {}
    controls = []
    for benchmark in sorted({row["benchmark"] for row in tasks}):
        kind = (
            "harbor"
            if benchmark
            in (
                "bfclparity-pi",
                "bixbench-pi",
                "ds-1000-local",
                "ot-tblite-recovery",
                "swebench-verified",
                "tau3-pi",
                "tb2-recovery",
            )
            else "evalchemy"
        )
        source = read_json(BASE + "/" + kind + "-" + benchmark + ".json")
        parent[benchmark] = source["parent"]
        groups = defaultdict(list)
        for row in source["rows"]:
            groups[row["task_id"]].append(row)
        for task_id, values in groups.items():
            if kind == "harbor":
                harbor[benchmark, task_id] = values[0]
            else:
                docs[benchmark, task_id] = values
        if kind == "evalchemy":
            spec = source["rows"][0]["spec"]
            metric = {
                "NUPA": "exact_match",
                "ifbench": "strict_prompt_accuracy",
                "gsm8k-0shot": "exact_match",
                "triviaqa": "exact_match",
            }.get(benchmark, "accuracy")
            config = {"benchmark": benchmark, "spec": spec, "metric": metric, "judge_model": profile["judge"]["model"]}
            empty = await grade_response(config, "")
            assert empty == 0, (benchmark, "empty", empty)
            golden = smoke_response(benchmark, spec)
            if golden is not None:
                score = await grade_response(config, golden)
                assert score >= 0.999999, (benchmark, "golden", score, golden[:150])
            else:
                score = None
            controls.append({"benchmark": benchmark, "empty": empty, "reference_response": score})
            print(json.dumps(controls[-1]), flush=True)
    rows = []
    index = []
    machine = MachineRuntimeSpec(
        backend="local",
        network=NetworkPolicy.ALLOW,
        cpus=2,
        memory_mb=8192,
        storage_mb=16384,
        gpus=0,
        user=None,
        startup_timeout=600,
        cleanup_timeout=60,
    )
    for task in tasks:
        benchmark, task_id = task["benchmark"], task["task_id"]
        identifier = benchmark + "/" + task_id
        if (benchmark, task_id) in harbor:
            original = harbor[benchmark, task_id]
            spec = TaskSpec.model_validate_json(original["task_spec"])
            harness = "harbor_trial"
            verifier_machine = None
            config = yaml.safe_load(parent[benchmark]["config_text"])
            repetitions = int(config.get("n_attempts", 1))
            source_keys = []
            locator = original["source"]["identity"]
            policy = {
                "kind": "harbor",
                "source": locator,
                "original_config": parent[benchmark]["config_text"],
                "task_name": original["source"]["task_name"],
                "file_sha256": original["file_sha256"],
            }
        else:
            originals = docs[benchmark, task_id]
            raw = originals[0]["spec"]
            metric = {
                "NUPA": "exact_match",
                "ifbench": "strict_prompt_accuracy",
                "gsm8k-0shot": "exact_match",
                "triviaqa": "exact_match",
            }.get(benchmark, "accuracy")
            config = {"benchmark": benchmark, "spec": raw, "metric": metric, "judge_model": profile["judge"]["model"]}
            package = script_package(
                grader_files, config, environment=grading_environment, timeout=300, answer_path="/app/answer.txt"
            )
            document_hash = hashlib.sha256(json.dumps(raw["doc"], sort_keys=True).encode()).hexdigest()
            source = Source(
                dataset=raw["source"]["dataset"],
                revision=raw["source"]["revision"] or "embedded-doc-sha256:" + document_hash,
                row=str(raw["source_id"]),
                importer_revision="marin-815c4b2c81:mini-policy-validation-v1",
            )
            events = tuple(TextMessage(**m) for m in original_prompt(raw, benchmark))
            spec = TaskSpec(
                id=identifier,
                context=ConversationInput(events=events),
                environment_requirements=EnvironmentRequirements(),
                answer_type=AnswerType.TEXT,
                answer_format=PlainText(),
                grader=package.grader,
                resources=ResourceGroups(verifier=package.resources),
                source=source,
                tags=("validation", "mini-eval-policy", benchmark),
            )
            harness = "shellbox:" + json.dumps({"command_timeout": None, "shell_tool": {}}, sort_keys=True)
            verifier_machine = machine
            repetitions = len(originals)
            source_keys = [r["source_key"] for r in originals]
            locator = raw["source"]
            policy = {
                "kind": "evalchemy",
                "source": locator,
                "original_parent": parent[benchmark],
                "source_keys": source_keys,
                "specs": [
                    {
                        **r,
                        "requests_sha256": (
                            hashlib.sha256(json.dumps(r["spec"]["requests"], sort_keys=True).encode()).hexdigest()
                        ),
                    }
                    for r in originals
                ],
            }
        limits = RolloutLimits(
            max_turns=1024 if harness == "harbor_trial" else 1,
            max_total_tokens=None,
            model_turn_timeout=1800,
            tool_turn_timeout=1800,
            total_turn_timeout=1800,
            attempt_timeout=2460,
            verifier_timeout=300,
            cleanup_timeout=60,
        )
        lowered = TaskRolloutSpec(
            task=spec, harness=harness, task_machine=None, verifier_machine=verifier_machine, limits=limits
        )
        TaskSpec.model_validate_json(spec.model_dump_json())
        TaskRolloutSpec.model_validate_json(lowered.model_dump_json())
        rows.append(
            {
                "task_id": identifier,
                "source_task_id": task_id,
                "benchmark": benchmark,
                "selection_order": task["selection_order"],
                "is_micro": task["selection_order"] < 210,
                "repetitions": repetitions,
                "task_spec": spec.model_dump_json(),
                "task_rollout_spec": lowered.model_dump_json(),
                "agent": "pi" if harness == "harbor_trial" else "single-turn",
                "validation_profile": profile["name"],
            }
        )
        index.append(
            {
                "task_id": identifier,
                "benchmark": benchmark,
                "matrix_column": task["matrix_column"],
                "selection_order": task["selection_order"],
                "is_micro": task["selection_order"] < 210,
                "source_task_id": task_id,
                "repetitions": repetitions,
                "policy": policy,
            }
        )
    assert len(rows) == 840 and len({r["task_id"] for r in rows}) == 840
    assert len({r["benchmark"] for r in rows}) == 21
    for budget, name in [(840, "mini"), (210, "micro")]:
        selected = rows[:budget]
        folder = output / name / "data"
        folder.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(selected), folder / "validation.parquet", compression="zstd")
        expanded = []
        for row in selected:
            for repeat in range(row["repetitions"]):
                expanded.append(dict(row, repetition_index=repeat))
        pq.write_table(pa.Table.from_pylist(expanded), folder / "validation-rollouts.parquet", compression="zstd")
    idx = output / "index"
    idx.mkdir()
    with (idx / "tasks.jsonl").open("w") as handle:
        for row in index:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    pq.write_table(
        pa.Table.from_pylist(
            [
                {k: v for k, v in row.items() if k != "policy"} | {"policy_json": json.dumps(row["policy"])}
                for row in index
            ]
        ),
        idx / "tasks.parquet",
        compression="zstd",
    )
    (idx / "selection.json").write_text(json.dumps(selection, indent=2))
    (idx / "profile.json").write_text(json.dumps(profile, indent=2))
    (output / "controls.json").write_text(json.dumps(controls, indent=2))
    manifest = {
        "tasks": len(rows),
        "micro": 210,
        "benchmarks": sorted({r["benchmark"] for r in rows}),
        "counts": dict(Counter(r["benchmark"] for r in rows)),
        "micro_counts": dict(Counter(r["benchmark"] for r in rows[:210])),
        "mini_rollouts": sum(r["repetitions"] for r in rows),
        "micro_rollouts": sum(r["repetitions"] for r in rows[:210]),
        "runtime": runtime,
        "files": {
            str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in output.rglob("*")
            if p.is_file()
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    archive = output.parent / "release.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(output, arcname="release")
    fs, key = filesystem_for(BASE + "/release.tar.gz")
    fs.pipe_file(key, archive.read_bytes())
    fs, key = filesystem_for(BASE + "/release-manifest.json")
    fs.pipe_file(key, json.dumps(manifest, indent=2).encode())
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    asyncio.run(main())

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import polars as pl
from taskcompendium.grader import verifyit_package
from taskcompendium.models import (
    SCHEMA_VERSION,
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.curriculum.models import AlignmentRelease
from taskcompendium.pipeline.semantics import task_content_hash
from verifyit.spec import NumericSpec
from zephyr.context import ZephyrContext

from experiments.post_training.capability_pool.allocation import InfeasiblePolicy
from experiments.post_training.capability_pool.labels import extract_labels
from experiments.post_training.capability_pool.pool import PoolRecipe, PoolSource, RowRewrite, build_pools, rewrite_spec

CAP_A = "d01.algebra.a"
CAP_B = "d01.algebra.b"
OTHER = "d02.other"
NODES = [
    ("subject:D01", "D01", None, "subject"),
    ("d01.algebra", "D01", "subject:D01", "group"),
    (CAP_A, "D01", "d01.algebra", "capability"),
    (CAP_B, "D01", "d01.algebra", "capability"),
    ("subject:D02", "D02", None, "subject"),
    (OTHER, "D02", "subject:D02", "capability"),
]


def _spec(task_id: str, prompt: str, tags: tuple[str, ...] = ()) -> str:
    task = TaskSpec(
        id=task_id,
        context=ConversationInput(events=(TextMessage(role="user", content=prompt),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.NUMBER,
        answer_format=PlainText(),
        grader=verifyit_package(NumericSpec("12", tolerance_abs=0, tolerance_rel=0)).grader,
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
        tags=list(tags),
    )
    limits = {"max_turns": 1, "max_total_tokens": None, "verifier_timeout": 5, "cleanup_timeout": 5}
    limits |= dict.fromkeys(("model_turn_timeout", "tool_turn_timeout", "total_turn_timeout", "attempt_timeout"))
    rollout = {"task": task.model_dump(mode="json"), "harness": "shell", "task_machine": None}
    return json.dumps(rollout | {"verifier_machine": None, "limits": limits, "trace": "exact_tokens"})


def _older_spec(task_id: str, prompt: str, capability: str) -> str:
    """A Taskforge row as exported before schema 0.28, with a cumulative token budget."""
    row = json.loads(_spec(task_id, prompt, (f"capability:{capability}",)))
    row["task"]["schema_version"] = "0.27"
    row["limits"]["max_total_tokens"] = 65536
    return json.dumps(row)


def _content_hash(spec: str) -> str:
    return task_content_hash(json.loads(spec)["task"])


def _assignment(spec: str, task_id: str, capability: str, similarity: float, intended_use: str = "train") -> dict:
    subject = capability.split(".")[0].upper()
    return {
        "source_id": "trove",
        "task_id": task_id,
        "content_hash": _content_hash(spec),
        "intended_use": intended_use,
        "description_status": "ready",
        "matches": {
            "subjects": [
                {
                    "subject_id": subject,
                    "facet": "subject_domain",
                    "membership_similarity": 0.9,
                    "membership_anchor_id": subject,
                    "capability_id": capability,
                    "margin": 0.1,
                    "candidates": [
                        {"subject_id": subject, "capability_id": capability, "similarity": similarity, "anchor_id": "x"}
                    ],
                    "subject_candidates": [{"subject_id": subject, "similarity": 0.9, "anchor_id": "x"}],
                }
            ],
            "capabilities": [],
        },
    }


def _fixture(tmp_path: Path) -> PoolRecipe:
    trove, assignments = [], []
    for capability in (CAP_A, CAP_B):
        for index in range(6):
            task_id = f"trove-{capability}-{index}"
            spec = _spec(task_id, f"{capability} problem {index}")
            trove.append({"task_rollout_spec": spec, "task_id": task_id, "source_id": "trove"})
            assignments.append(_assignment(spec, task_id, capability, 0.8))
    extras = [
        ("trove-duplicate", f"{CAP_A} problem 0", CAP_A, 0.8, "train"),
        ("trove-eval", "an evaluation problem", CAP_A, 0.8, "eval"),
        ("trove-subject-only", "a vague problem", CAP_A, 0.6, "train"),
        ("trove-other", "another subject", OTHER, 0.8, "train"),
        ("trove-unlabelled", "never aligned", None, 0.0, "train"),
    ]
    for task_id, prompt, capability, similarity, intended_use in extras:
        spec = _spec(task_id, prompt)
        trove.append({"task_rollout_spec": spec, "task_id": task_id, "source_id": "trove"})
        if capability is not None:
            assignments.append(_assignment(spec, task_id, capability, similarity, intended_use))
    invalid = json.loads(_spec("trove-invalid", "a newer task schema"))
    invalid["task"]["context"]["token_ids"] = [1, 2]
    trove.append({"task_rollout_spec": json.dumps(invalid), "task_id": "trove-invalid", "source_id": "trove"})
    forge = [
        {"task_rollout_spec": _older_spec(f"forge-{capability}-{index}", f"forge {capability} {index}", capability)}
        for capability in (CAP_A, CAP_B, OTHER)
        for index in range(3)
    ]
    forge.append({"task_rollout_spec": _spec("forge-untagged", "forge untagged")})
    paths = {name: tmp_path / f"{name}.parquet" for name in ("trove", "forge", "heldout", "assignments")}
    pl.DataFrame(trove).write_parquet(paths["trove"])
    pl.DataFrame(forge).write_parquet(paths["forge"])
    pl.DataFrame(forge[:1]).write_parquet(paths["heldout"])
    pl.DataFrame(assignments).write_parquet(paths["assignments"])

    release = AlignmentRelease.model_validate(
        {
            "release_id": "release",
            "created_at": "2026-10-22T00:00:00Z",
            "curriculum_key": "curriculum",
            "population": "converted",
            "sample_size": None,
            "task_count": len(assignments),
            "ready_count": len(assignments),
            "insufficient_context_count": 0,
            "sources": [],
            "inventory": [],
            "nodes": [
                {
                    "node_id": node,
                    "subject_id": subject,
                    "parent_id": parent,
                    "kind": kind,
                    "name": node,
                    "description": "",
                }
                for node, subject, parent, kind in NODES
            ],
            "paths": [str(paths["assignments"])],
            "lineage": {},
            "embedding_spec": {
                "model": "m",
                "revision": "r",
                "tokenizer_revision": "r",
                "dimensions": 4,
                "max_tokens": 16,
                "compute_dtype": "float32",
            },
        }
    )
    release_path = tmp_path / "release.json"
    release_path.write_text(release.model_dump_json())
    with ZephyrContext(max_workers=2, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
        extract_labels(context, release, str(release_path), str(tmp_path / "labels"))
    return PoolRecipe(
        atlas_release=str(release_path),
        labels=str(tmp_path / "labels"),
        subject_distance=0.45,
        capability_distance=0.25,
        target={CAP_A: 1.0, CAP_B: 1.0},
        pool_size=8,
        seed=0,
        arms={"control": 0.0, "taskforge25": 0.25},
        infeasible=InfeasiblePolicy.ERROR,
        tasktrove=[str(paths["trove"])],
        taskforge=[str(paths["forge"])],
        exclude=[str(paths["heldout"])],
        rewrites={PoolSource.TASKFORGE: RowRewrite(limits={"max_total_tokens": None})},
    )


def _build(recipe: PoolRecipe, tmp_path: Path, name: str) -> dict:
    with ZephyrContext(max_workers=2, chunk_storage_prefix=str(tmp_path / f"chunks-{name}")) as context:
        return build_pools(context, recipe, str(tmp_path / name))


def test_arms_share_the_capability_marginal_and_differ_only_by_swapped_tasks(tmp_path: Path):
    recipe = _fixture(tmp_path)
    selection = _build(recipe, tmp_path, "pools")

    arms = {arm: pl.read_parquet(tmp_path / "pools" / arm / "train.parquet") for arm in recipe.arms}
    for frame in arms.values():
        assert dict(frame["target_node"].value_counts().iter_rows()) == {CAP_A: 4, CAP_B: 4}
        for row in frame.iter_rows(named=True):
            assert json.loads(row["task_rollout_spec"])["task"]["id"] == row["task_id"]
    control, treatment = arms["control"], arms["taskforge25"]
    assert set(control["pool_source"]) == {"tasktrove"}
    forge_rows = treatment.filter(pl.col("pool_source") == "taskforge")
    assert dict(forge_rows["target_node"].value_counts().iter_rows()) == {CAP_A: 1, CAP_B: 1}
    assert set(treatment.filter(pl.col("pool_source") == "tasktrove")["task_id"]) < set(control["task_id"])
    for spec in forge_rows["task_rollout_spec"]:
        row = json.loads(spec)
        assert row["task"]["schema_version"] == SCHEMA_VERSION
        assert row["limits"]["max_total_tokens"] is None
    assert selection["arms"]["taskforge25"]["rewritten_fields"] == {
        "limits.max_total_tokens": 2,
        f"task.schema_version:0.27->{SCHEMA_VERSION}": 2,
    }
    assert selection["arms"]["control"]["rewritten_fields"] == {}

    chosen = set(control["task_id"]) | set(treatment["task_id"])
    left_out = {"forge-d01.algebra.a-0", "trove-duplicate", "trove-eval", "trove-unlabelled", "trove-subject-only"}
    assert not chosen & left_out
    assert selection["census"]["taskforge"] == {
        "rows": 10,
        "held_out": 1,
        "untagged": 1,
        "outside_target": 3,
        "eligible": 5,
    }
    assert selection["census"]["tasktrove"] == {
        "rows": 18,
        "invalid_spec": 1,
        "unlabelled": 1,
        "evaluation_source": 1,
        "outside_target": 2,
        "duplicate": 1,
        "eligible": 12,
    }

    manifest = json.loads((tmp_path / "pools" / "control" / "artifact-manifest.json").read_text())
    for entry in manifest["files"]:
        payload = (tmp_path / "pools" / "control" / entry["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]

    candidates = pl.read_parquet(tmp_path / "pools" / "candidates.parquet")
    assert candidates.filter(pl.col("task_id") == "trove-duplicate")["excluded_by"].to_list() == ["duplicate"]

    rebuilt = _build(recipe, tmp_path, "again")
    assert rebuilt["artifact_manifest_sha256"] == selection["artifact_manifest_sha256"]


def test_raising_a_grader_deadline_keeps_the_verifier_margin():
    row = json.loads(_spec("scripted", "graded by a script"))
    environment = {"command_semantics": "linux_process", "package_spec": {"pypi_packages": [], "tools": []}}
    row["task"]["grader"] = {
        "kind": "script",
        "argv": ["python3", "/tests/grade.py"],
        "environment": EnvironmentRequirements.model_validate(environment).model_dump(mode="json"),
        "answer_path": "/app/answer.txt",
        "reward": {"kind": "file", "files": [{"path": "/logs/verifier/reward.json", "format": "json", "key": "reward"}]},
        "timeout": 120.0,
    }
    row["limits"]["verifier_timeout"] = 420.0

    spec, changed = rewrite_spec(json.dumps(row), RowRewrite(min_grader_timeout=300))

    rewritten = json.loads(spec)
    assert (rewritten["task"]["grader"]["timeout"], rewritten["limits"]["verifier_timeout"]) == (300, 600)
    assert changed == ["task.grader.timeout", "limits.verifier_timeout"]
    assert rewrite_spec(spec, RowRewrite(min_grader_timeout=300))[1] == []

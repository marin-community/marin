# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Round runner proofs and shared long-loop round invariants.

The byte-identity proofs run the real constructions against a pinned mirror of
the small storage objects they read; they are marked integration because the
mirror and credentials live in object storage. The invariant tests are offline.
"""

import dataclasses
import hashlib
import io
import json
import math
import re
import tarfile
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from experiments.post_training.long_loop import __main__ as loop_main
from experiments.post_training.long_loop import exp4, round_runner
from experiments.post_training.long_loop.round_runner import (
    DATA_RELEASES,
    RoundArchive,
    RoundSft,
    RoundSpec,
    RoundStorage,
    load_round,
    require_archive_binds_final_sft_checkpoint,
    require_permanent_round_storage,
    round_sft_spec,
)
from experiments.post_training.long_loop.tests.test_exp3_production import (
    preview_release as preview_release,
    production_recipe as production_recipe,
)
from experiments.post_training.long_loop.tests.test_initializer_stage import (
    prepared_sources as prepared_sources,
    staged_initializer as staged_initializer,
)

ROUND_DIR = Path(__file__).parents[1] / "rounds"
MIRROR_URI = (
    "s3://marin-us-east-08a/marin/users/benfeuer/long-loop-exp5/evidence/round-mirror/2026.11.19/round-mirror.tar.gz"
)
MIRROR_SHA256 = "26f306aa72c73986f0eb1094b7e1b71f7786b12d5af808b92c6a1f74297576e5"
MIRROR_BYTES = 1_450_984
TEST_REVISION = "b" * 40
# The recorded Exp5 dry run (exp5.py at commit 688c000eeb), canonicalized with the
# revision-derived fields normalized: marin_revision, runtime_source,
# resolved_sha256 and spec_sha256 all depend on the constructing workspace.
EXP5_RECORDED_PLAN_SHA256 = "af1958c3653ff0a2cd2d8ede0caa4cac1f81e01e617a4cfa9db021404200f488"
YAML_DIFF_PATHS = {
    "name",
    "version",
    "slug",
    "storage.output_prefix",
    "archive.hf_upload_repo_prefix",
    "archive.evaldash_run_prefix",
    "sft.special_token_lr_multiplier",
}
PLAN_DIFF_PATHS = {
    "name",
    "storage.output_prefix",
    "execution.coordinator.coordinator_name",
    "rl.name",
    "evaluation.name",
    "archive.hf_upload.repo_prefix",
    "archive.evaldash.run_prefix",
    "sft.special_token_lr_multiplier",
    "resolved_sha256",
}


def _canonical(plan: dict) -> str:
    """Normalize the revision-derived fields for cross-workspace comparisons."""
    normalized = json.loads(json.dumps(plan))
    for key in ("resolved_sha256", "runtime_source", "spec_sha256"):
        normalized.pop(key, None)
    normalized["marin_revision"] = "<revision>"
    return json.dumps(normalized, sort_keys=True, indent=1)


def _leaf_paths(value, path=""):
    if isinstance(value, dict) and value:
        for key, item in value.items():
            yield from _leaf_paths(item, f"{path}.{key}" if path else key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _leaf_paths(item, f"{path}[{index}]")
    else:
        yield path, value


def _structural_diff(left: dict, right: dict, *, bound: int = 140) -> dict[str, str]:
    left_leaves = dict(_leaf_paths(left))
    right_leaves = dict(_leaf_paths(right))
    differences = {}
    for path in sorted(set(left_leaves) | set(right_leaves)):
        old = left_leaves.get(path, "<missing>")
        new = right_leaves.get(path, "<missing>")
        if old != new:
            old_text = str(old)
            new_text = str(new)
            differences[path] = (
                f"{old_text[:bound]}{'…' if len(old_text) > bound else ''} -> "
                f"{new_text[:bound]}{'…' if len(new_text) > bound else ''}"
            )
    return differences


@pytest.fixture
def mirror_storage(monkeypatch, tmp_path):
    """Route the construction's small storage reads to the pinned object-storage mirror."""
    from rigging.filesystem.cluster_config import StoreType
    from rigging.filesystem.s3_compat import configure_coreweave_s3, s3_credentials
    from rigging.filesystem.storage_path import StoragePath

    if s3_credentials(StoreType.COREWEAVE) is None:
        pytest.skip("CoreWeave credentials are required for the round construction proofs")
    configure_coreweave_s3()
    archive_bytes = StoragePath(MIRROR_URI).read_bytes()
    assert len(archive_bytes) == MIRROR_BYTES
    assert hashlib.sha256(archive_bytes).hexdigest() == MIRROR_SHA256
    root = tmp_path / "mirror"
    root.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        archive.extractall(root, filter="data")

    original = StoragePath.read_bytes

    def mirrored(self):
        uri = f"s3://{self.bucket}/{self.key}" if self.bucket else str(self)
        candidate = root / uri.removeprefix("s3://")
        if candidate.exists():
            return candidate.read_bytes()
        return original(self)

    monkeypatch.setattr(StoragePath, "read_bytes", mirrored)
    noop = lambda *args, **kwargs: None  # noqa: E731
    monkeypatch.setattr(exp4, "require_clean_revision", noop)
    monkeypatch.setattr(round_runner, "require_clean_revision", noop)
    # Bundles carry no git repo, so the real manifest walk falls back to rglob over
    # the whole workspace including .venv — minutes per construction. The receipt is
    # workspace-derived metadata that both sides must merely agree on.
    stub_manifest = lambda workspace, revision: {  # noqa: E731
        "schema": 1,
        "marin_revision": revision,
        "files": [{"path": "round-runner-proof", "size": 0, "sha256": "0" * 64}],
    }
    monkeypatch.setattr(exp4, "runtime_source_manifest", stub_manifest)
    monkeypatch.setattr(round_runner, "runtime_source_manifest", stub_manifest)
    captured_launches: list[dict] = []

    def stubbed_validate_launch(document, directory, *, model_config=None):
        captured_launches.append(
            {"document": document, "model_config": None if model_config is None else model_config.decode()}
        )
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "launch.yaml").write_text(document)
        (directory / "receipt.json").write_text(json.dumps({"stubbed": True}, indent=2))
        return {"stubbed": True}

    monkeypatch.setattr(loop_main, "validate_launch", stubbed_validate_launch)
    return captured_launches


def _invoke(main, args):
    result = CliRunner().invoke(main, args, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result


def _exp4_args(validation_output: Path) -> list[str]:
    binding = DATA_RELEASES["v7-option-g"]
    return [
        "--sft-artifact-record",
        binding.record,
        "--sft-artifact-record-sha256",
        binding.record_sha256,
        "--version",
        "2026.11.18.76",
        "--source-revision",
        TEST_REVISION,
        "--validation-output",
        str(validation_output),
    ]


def _round_args(round_yaml: str, version: str, validation_output: Path) -> list[str]:
    return [
        "--round",
        str(ROUND_DIR / round_yaml),
        "--version",
        version,
        "--source-revision",
        TEST_REVISION,
        "--validation-output",
        str(validation_output),
    ]


def _launch_fingerprints(captured: list[dict]) -> list[str]:
    """Hash-first fingerprints: launch documents are large and never diff-rendered."""
    return [
        hashlib.sha256(item["document"].encode()).hexdigest()
        + ("" if item["model_config"] is None else ":" + hashlib.sha256(item["model_config"].encode()).hexdigest())
        for item in captured
    ]


def _normalized_output(output: str) -> str:
    """Plan prints embed live function reprs; their object addresses are not identity."""
    return re.sub(r"0x[0-9a-f]+", "0xADDR", output)


@pytest.mark.integration
@pytest.mark.timeout(900)
def test_round_exp4_reproduces_exp4_construction_byte_for_byte(tmp_path, mirror_storage):
    exp4_output = tmp_path / "exp4"
    runner_output = tmp_path / "runner"
    exp4_result = _invoke(exp4.main, _exp4_args(exp4_output))
    exp4_launches = _launch_fingerprints(mirror_storage)
    runner_result = _invoke(round_runner.main, _round_args("exp4.yaml", "2026.11.18.76", runner_output))
    left = _normalized_output(exp4_result.output)
    right = _normalized_output(runner_result.output)
    if left != right:
        left_lines = left.splitlines()
        right_lines = right.splitlines()
        detail = [f"{x!r} != {y!r}" for x, y in zip(left_lines, right_lines) if x != y][:10]
        pytest.fail(f"printed plan differs ({len(left_lines)} vs {len(right_lines)} lines): {detail}")
    assert _launch_fingerprints(mirror_storage[len(exp4_launches) :]) == exp4_launches
    for name in ("resolved-plan.json", "optimizer-groups.json", "sft-lr-schedule.json", "graph.json"):
        left = (exp4_output / name).read_bytes()
        right = (runner_output / name).read_bytes()
        if hashlib.sha256(left).hexdigest() != hashlib.sha256(right).hexdigest():
            detail = "payloads differ"
            if name.endswith(".json"):
                paths = _structural_diff(json.loads(left), json.loads(right))
                detail = "; ".join(f"{path}: {change}" for path, change in list(paths.items())[:20])
            pytest.fail(f"{name} differs: {detail}")
        assert left == right


@pytest.mark.integration
@pytest.mark.timeout(900)
def test_round_exp5_matches_the_recorded_exp5_dry_run(tmp_path, mirror_storage):
    exp4_output = tmp_path / "exp4"
    exp5_output = tmp_path / "exp5"
    _invoke(exp4.main, _exp4_args(exp4_output))
    result = _invoke(round_runner.main, _round_args("exp5.yaml", "2026.11.19.76", exp5_output))
    assert "long-loop-exp5/round5/sft@2026.11.19.76" in result.output
    assert "users/benfeuer/checkpoints/mochi-exp5-rl@2026.11.19.76" in result.output
    plan = json.loads((exp5_output / "resolved-plan.json").read_bytes())
    assert hashlib.sha256(_canonical(plan).encode()).hexdigest() == EXP5_RECORDED_PLAN_SHA256
    assert plan["sft"]["special_token_lr_multiplier"] == 1.0
    exp4_plan = json.loads((exp4_output / "resolved-plan.json").read_bytes())
    differences = _structural_diff(exp4_plan, plan)
    assert set(differences) == PLAN_DIFF_PATHS


def test_round_yaml_diff_is_exactly_identity_and_multiplier():
    exp4_round = load_round(ROUND_DIR / "exp4.yaml")
    exp5_round = load_round(ROUND_DIR / "exp5.yaml")
    differences = _structural_diff(dataclasses.asdict(exp4_round), dataclasses.asdict(exp5_round))
    assert set(differences) == YAML_DIFF_PATHS
    assert exp4_round.sft.special_token_lr_multiplier == math.sqrt(32)
    assert exp5_round.sft.special_token_lr_multiplier == 1.0


def test_round_schema_rejects_unknown_and_mistyped_fields(tmp_path):
    document = yaml.safe_load((ROUND_DIR / "exp4.yaml").read_text())
    document["unknown_knob"] = True
    path = tmp_path / "unknown.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    with pytest.raises(ValueError, match="unknown fields.*unknown_knob"):
        load_round(path)
    document = yaml.safe_load((ROUND_DIR / "exp4.yaml").read_text())
    document["sft"]["updates"] = "360"
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    with pytest.raises(ValueError, match="sft.updates must be int"):
        load_round(path)


def _round_spec(recipe: dict, **sft_overrides) -> RoundSpec:
    sft = RoundSft(
        updates=360,
        batch_size=recipe["batch_size"],
        min_lr_ratio=0.1,
        learning_rate=5.0e-05,
        adam_learning_rate=5.0e-05 * 3 / 13,
        mesh=recipe["mesh"],
        resources=recipe["resources"],
        special_token_lr_multiplier=math.sqrt(32),
    )
    return RoundSpec(
        schema=1,
        name="long-loop-exp4/round4",
        version="2026.11.18.76",
        slug="mochi-exp4",
        cluster="cw-rno2a",
        storage=RoundStorage(
            output_prefix="s3://marin-us-east-08a/marin/users/benfeuer/long-loop-exp4",
            temporary_prefix="s3://marin-us-east-08a/tmp",
        ),
        archive=RoundArchive(
            hf_upload_repo_prefix="open-athena/Snowball-LongLoop-Round4",
            evaldash_run_prefix="long-loop-round4",
        ),
        data_release="v7-option-g",
        pools="exp4-regional",
        evaluation="frozen-micro210",
        initializer="exp3-staged",
        sft=dataclasses.replace(sft, **sft_overrides),
    )


def _annealed_recipe(recipe: dict, **schedule_overrides) -> dict:
    updates = 360
    recipe["stop_after_steps"] = updates
    recipe["trainer"]["num_train_steps"] = updates
    recipe["optimizer_schedule"] = {
        "horizon": updates,
        "warmup": 0,
        "decay": updates,
        "min_lr_ratio": 0.1,
        "lr_schedule": "linear",
        **schedule_overrides,
    }
    return recipe


def test_round_sft_spec_holds_the_ruled_multiplier_by_construction(production_recipe):
    spec = round_sft_spec(
        _round_spec(production_recipe), "round-test/sft", "2026.11.18.76", _annealed_recipe(production_recipe)
    )
    assert spec.snowball_sft.special_token_lr_multiplier == math.sqrt(32)
    override = round_sft_spec(
        _round_spec(production_recipe, special_token_lr_multiplier=1.0),
        "round-test/sft",
        "2026.11.18.76",
        _annealed_recipe(production_recipe),
    )
    assert override.snowball_sft.special_token_lr_multiplier == 1.0


def test_round_sft_spec_rejects_disagreeing_multipliers(production_recipe):
    recipe = _annealed_recipe(production_recipe)
    recipe["special_token_lr_multiplier"] = 1.0
    with pytest.raises(ValueError, match="two different special-token LR multipliers"):
        round_sft_spec(_round_spec(production_recipe), "round-test/sft", "2026.11.18.76", recipe)


def test_round_sft_spec_applies_the_learning_rate_knobs(production_recipe):
    spec = round_sft_spec(
        _round_spec(production_recipe, learning_rate=6.0e-05, adam_learning_rate=6.0e-05 * 3 / 13),
        "round-test/sft",
        "2026.11.18.76",
        _annealed_recipe(production_recipe),
    )
    assert spec.optimizer.learning_rate == 6.0e-05
    assert spec.optimizer.adam_lr == 6.0e-05 * 3 / 13
    assert spec.snowball_sft.special_token_lr_multiplier == math.sqrt(32)


def test_round_sft_spec_carries_the_min_lr_ratio_knob(production_recipe):
    recipe = _annealed_recipe(production_recipe, min_lr_ratio=0.2)
    spec = round_sft_spec(
        _round_spec(production_recipe, min_lr_ratio=0.2), "round-test/sft", "2026.11.18.76", recipe
    )
    assert spec.optimizer.min_lr_ratio == 0.2
    with pytest.raises(ValueError, match="min_lr_ratio=0.1"):
        round_sft_spec(_round_spec(production_recipe), "round-test/sft", "2026.11.18.76", recipe)


def test_round_sft_spec_rejects_a_horizon_that_differs_from_the_updates_run(production_recipe):
    recipe = _annealed_recipe(production_recipe, horizon=4750, decay=4750)
    with pytest.raises(ValueError, match="horizon must equal num_train_steps"):
        round_sft_spec(_round_spec(production_recipe), "round-test/sft", "2026.11.18.76", recipe)


def test_round_storage_invariant_rejects_scratch_output_prefix():
    recipe = {"batch_size": 256, "mesh": {}, "resources": {}}
    spec = dataclasses.replace(
        _round_spec(recipe),
        storage=RoundStorage(
            output_prefix="s3://marin-us-east-08a/tmp/round", temporary_prefix="s3://marin-us-east-08a/tmp"
        ),
    )
    with pytest.raises(ValueError, match="permanent path"):
        require_permanent_round_storage(spec)


def test_round_retention_invariant_requires_the_sft_selection_binding():
    resolved = {"archive": {"stage_files": {"sft": [{"path": "consumed-config.json", "category": "config"}]}}}
    with pytest.raises(ValueError, match="selected.json"):
        require_archive_binds_final_sft_checkpoint(resolved)
    resolved = {"archive": {"stage_files": {"sft": [{"path": "selected.json", "category": "config"}]}}}
    require_archive_binds_final_sft_checkpoint(resolved)

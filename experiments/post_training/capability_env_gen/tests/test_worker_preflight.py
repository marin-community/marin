import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "worker.sh"
MARIN = ROOT.parents[0] / "marin"


def run_worker(
    tmp_path: Path, stage: str, source: Path
) -> subprocess.CompletedProcess[str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uv = fake_bin / "uv"
    # build-runtime "succeeds" by leaving a stub interpreter at --target, so preflight
    # reaches the checks under test; every other uv call fails as before.
    fake_uv.write_text(
        "#!/bin/sh\n"
        'case "$*" in *build-runtime*)\n'
        '  while [ "$1" != --target ]; do shift; done\n'
        '  mkdir -p "$2/bin" && printf \'#!/bin/sh\\ncase "$*" in *--help*) exit 0;; esac\\nexit 1\\n\' > "$2/bin/python"\n'
        '  chmod +x "$2/bin/python"; exit 0;;\n'
        "esac\nexit 1\n"
    )
    fake_uv.chmod(0o755)
    environment = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "GLM_BASE_URL": "http://example.invalid",
        "GLM_API_TOKEN": "test-token",
        "CW_KEY_ID": "test-id",
        "CW_KEY_SECRET": "test-secret",
        "MARIN_PROJECT": str(MARIN),
    }
    return subprocess.run(
        [
            "bash",
            str(WORKER),
            "--stage",
            stage,
            "--source",
            str(source),
            "--out",
            "runs/test",
            "--concurrency",
            "1",
            "--tier",
            "bulk",
            "--s3-dest",
            "s3://example/run",
            "--run-name",
            "test",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )


def test_propose_seed_does_not_require_accepted_json(tmp_path):
    seed = tmp_path / "seed"
    seed.mkdir()
    for name in ("input_pilot.json", "plans.json", "proposals.json", "report.json"):
        (seed / name).write_text("{}\n")

    result = run_worker(tmp_path, "propose", seed)

    assert result.returncode != 0
    assert "could not restore prior durable results" in result.stdout
    assert "staged accepted proposals missing" not in result.stdout


def test_synthesis_still_requires_accepted_json(tmp_path):
    source = tmp_path / "source"
    source.mkdir()

    result = run_worker(tmp_path, "synthesize", source)

    assert result.returncode == 2
    assert "staged accepted proposals missing" in result.stdout


def test_image_migration_rejects_direct_tree_transport(tmp_path):
    source = tmp_path / "source"
    (source / "restore-seed").mkdir(parents=True)
    (source / "accepted.json").write_text("[]\n")
    (source / "manifest.json").write_text("{}\n")

    result = run_worker(tmp_path, "image-migration-revalidation", source)

    assert result.returncode == 2
    assert "image migration source destination must be new" in result.stdout


def test_checkpoint_revalidation_requires_opaque_transport(tmp_path):
    source = tmp_path / "source"
    source.mkdir()

    result = run_worker(tmp_path, "checkpoint-revalidation", source)

    assert result.returncode == 2
    assert "checkpoint transport receipts are missing" in result.stdout
    assert "staged accepted proposals missing" not in result.stdout


def test_checkpoint_revalidation_worker_binds_restored_input_and_new_controller():
    worker = WORKER.read_text()
    branch = worker[worker.index('elif [ "$PHASE" = checkpoint-revalidation ]; then'):]
    assert 'checkpoint_revalidation_transport.py" download' in worker
    assert 'checkpoint_revalidation_transport.py" restore' in worker
    assert 'running_controller_provenance()' in branch
    assert '"$HERE/scripts/run_checkpoint_revalidation.py"' in branch
    for argument in (
        '--expected-bundle-manifest-sha256 "$SOURCE_SHA"',
        '--expected-request-sha256 "$PLAN_SHA"',
        '--new-controller-provenance "$RESULTS/controller/new-controller-provenance.json"',
    ):
        assert argument in branch


def test_image_migration_worker_uses_script_file_and_long_gate_timeout():
    worker = WORKER.read_text()
    invocation = worker[
        worker.index('elif [ "$PHASE" = image-migration-revalidation ]') :
    ]

    assert '"$HERE/scripts/run_image_migration_revalidation.py"' in invocation
    assert (
        'python "$HERE/scripts/run_image_migration_revalidation.py"' not in invocation
    )
    assert "--validation-timeout 14400" in invocation
    assert "--expected-bundle-manifest-sha256" in invocation
    assert "--expected-candidate-image" in invocation
    assert "--expected-verifier-image" in invocation
    assert "preflight_image_migration_cache.py" not in invocation


def test_worker_finds_marin_project_from_nested_submission_stage(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "lib/rigging").mkdir(parents=True)
    (workspace / "pyproject.toml").write_text(
        "[project]\nname='workspace'\nversion='0'\n"
    )
    stage = workspace / "capability-pipeline-staging/submissions/run-001"
    stage.mkdir(parents=True)
    worker = stage / "worker.sh"
    worker.write_text(WORKER.read_text())
    worker.chmod(0o755)
    marker = (workspace / "lib/rigging").resolve()
    # The shell parser and source layout regression are covered without invoking
    # worker bootstrap or credentials.
    snippet = """
HERE=$(cd "$(dirname "$1")" && pwd)
candidate="$HERE"
while [ "$candidate" != / ]; do
  if [ -f "$candidate/pyproject.toml" ] && [ -d "$candidate/lib/rigging" ]; then printf '%s' "$candidate"; exit 0; fi
  candidate=$(dirname "$candidate")
done
exit 1
"""
    result = subprocess.run(
        ["bash", "-c", snippet, "_", str(worker)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert Path(result.stdout) == workspace
    assert marker.is_dir()


def test_evaluation_worker_requires_only_opaque_frozen_transport(tmp_path):
    source = tmp_path / "source"
    source.mkdir()

    result = run_worker(tmp_path, "evaluate", source)

    assert result.returncode == 2
    assert "evaluation transport archive is missing" in result.stdout
    assert "staged accepted proposals missing" not in result.stdout


def test_evaluation_worker_restores_frozen_transport_once():
    worker = WORKER.read_text()
    assert worker.count('restore_bundle.py" restore-evaluation') == 1


def test_regrade_worker_restores_pinned_transport_without_glm_requirement():
    worker = (Path(__file__).resolve().parents[1] / "scripts/worker.sh").read_text()
    assert worker.count('restore_bundle.py" restore-regrade') == 1
    assert 'if [ "$PHASE" != regrade ] && [ "$PHASE" != reset-probe ]; then\n  [ -n "${GLM_BASE_URL:-}" ]' in worker
    assert "CAPABILITY_REMOTE_REGRADE=1" in worker
    assert "regrade concurrency differs from its frozen plan" in worker
    branch = worker[worker.rindex('elif [ "$PHASE" = evaluate ]'):]
    branch = branch[:branch.index("\nelse\n")]
    assert "uv sync" not in branch
    assert '"${UV[@]}" python3 -m capability_pipeline.cli evaluate' in branch

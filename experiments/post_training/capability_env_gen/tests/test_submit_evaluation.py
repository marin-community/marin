import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("phase", ["generate", "synthesize", "checkpoint-revalidation", "image-migration-revalidation"])
def test_synthesis_entrypoints_stage_image_pipeline_dependencies(tmp_path, phase):
    submit = (ROOT / "scripts/submit.sh").read_text()
    marker = submit.index("    # Trusted image capture stays outside")
    start = submit.rfind("  if ", 0, marker)
    end = submit.index("\n  fi", marker) + len("\n  fi")
    stage = tmp_path / "stage"
    (stage / "scripts").mkdir(parents=True)
    probes = tmp_path / "build-envs/envgen/probes"
    probes.mkdir(parents=True)
    names = ("capture_rootfs.py", "in_sandbox_capture.py", "dtx.py", "cw_presign.py")
    for name in names:
        (probes / name).write_text("# staged controller fixture\n")
    subprocess.run(
        ["bash", "-euc", submit[start:end]],
        env=os.environ | {"PHASE": phase, "STAGE": str(stage),
                          "ROOT": str(ROOT), "BUILD_ENVS": str(tmp_path / "build-envs")},
        check=True, capture_output=True, text=True,
    )
    for name in ("capture_task_images.py", "review_generic_task_image.py",
                 "capture_generic_task_image.py", "publish_generic_task_image.py",
                 "probe_generic_task_image.py", "migrate_generic_task_images.py",
                 "image_publication_handoff.py"):
        assert (stage / "scripts" / name).read_bytes() == (ROOT / "scripts" / name).read_bytes()
    for name in names:
        assert (stage / "daytona-tools/capture-tools" / name).is_file()


def test_submit_validates_the_staged_evaluation_controller_copy():
    submit = (ROOT / "scripts/submit.sh").read_text()
    start = submit.index('staged evaluation controller was not imported')
    snippet = submit[start - 500 : start + 200]
    assert 'sys.path.insert(0, str(stage))' in snippet
    assert 'evaluation.__file__' in snippet
    assert 'map(Path, sys.argv[1:4])' in snippet
    assert 'evaluation.validate_plan(plan, fingerprint.name)' in snippet


def test_regrade_submission_uses_credential_free_model_path_and_pinned_transport():
    submit = (Path(__file__).resolve().parents[1] / "scripts/submit.sh").read_text()
    assert 'if [ "$PHASE" = regrade ] || [ "$PHASE" = reset-probe ]; then\n  CW_ID=' in submit
    assert 'if [ "$PHASE" != regrade ] && [ "$PHASE" != reset-probe ]; then\nBASE_URL="$(resolve_endpoint)"' in submit
    assert 'if [ "$PHASE" != regrade ] && [ "$PHASE" != reset-probe ]; then\n    IRIS_CMD+=(-e GLM_API_TOKEN' in submit
    assert '"pack-$TRANSPORT_KIND"' in submit
    assert '"restore-$TRANSPORT_KIND"' in submit


def test_every_stage_uses_its_own_transport_validator_package():
    submit = (ROOT / "scripts/submit.sh").read_text()
    worker = (ROOT / "scripts/worker.sh").read_text()
    assert 'cp "$HERE/restore_bundle.py" "$STAGE/scripts/restore_bundle.py"' in submit
    assert ': > "$STAGE/scripts/__init__.py"' in submit
    assert submit.index(': > "$STAGE/scripts/__init__.py"') < submit.index('if [ "$PHASE" = evaluate ] || [ "$PHASE" = regrade ]; then')
    assert 'PYTHONSAFEPATH=1 PYTHONPATH="$STAGE${PYTHONPATH:+:$PYTHONPATH}"' in submit
    assert 'export PYTHONSAFEPATH=1' in worker


def test_generate_adoption_uses_opaque_s3_transport_and_fresh_output_only():
    submit = (ROOT / "scripts/submit.sh").read_text()
    worker = (ROOT / "scripts/worker.sh").read_text()
    assert 'proposal adoption requires checkpoint, source archive and launch receipt together' in submit
    assert 'proposal adoption starts a fresh generate output and cannot use --resume' in submit
    assert 'proposal_adoption_transport.py" upload' in submit
    assert 'proposal-adoption.transport.json' in submit
    assert 'proposal_adoption_transport.py" download' in worker
    assert 'validate_checkpoint' in submit and 'validate_checkpoint' in worker
    assert '--adopt-proposal-checkpoint "$ADOPTION_ROOT/checkpoint"' in worker


def test_checkpoint_revalidation_submits_exact_bundle_over_s3_once():
    submit = (ROOT / "scripts/submit.sh").read_text()
    assert 'checkpoint revalidation requires --concurrency 1' in submit
    assert 'checkpoint revalidation requires a fresh output prefix' in submit
    assert 'validate_checkpoint_bundle(Path(sys.argv[1])' in submit
    assert 'checkpoint_revalidation_transport.py" pack' in submit
    assert 'checkpoint_revalidation_transport.py" upload' in submit
    assert '--manifest-sha256 "$SOURCE_SHA" --request-sha256 "$PLAN_SHA"' in submit
    assert '[ -n "$SOURCE" ] && [ "$PHASE" != checkpoint-revalidation ]' in submit
    assert 'cp "$HERE/run_checkpoint_revalidation.py" "$STAGE/scripts/run_checkpoint_revalidation.py"' in submit
    assert 'JOB_CMD+=" --plan-sha256' in submit

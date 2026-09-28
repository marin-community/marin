#!/usr/bin/env bash
set -euo pipefail

release_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(git -C "$release_dir" rev-parse --show-toplevel)
cd "$repo_dir"

uv run python - "$release_dir" "$@" <<'PY'
import argparse
import hashlib
import io
import json
import re
import shlex
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from iris.cluster.client.bundle import create_workspace_zip

from experiments.domain_phase_mix import evaluate_tpp10_target_math_complete as evaluation
from experiments.domain_phase_mix.east5_launch_safety import validate_regional_iris_command
from experiments.domain_phase_mix.starcoder_epoch_matching import file_sha256

parser = argparse.ArgumentParser(description="Validate a target-math launch without submitting jobs or writing GCS")
parser.add_argument("release_directory", type=Path)
parser.add_argument("--command-only", action="store_true", help="Check placement and template before the final spec exists")
args = parser.parse_args()
directory = args.release_directory.resolve()
workspace = Path.cwd().resolve()
submit_path = directory / "submit.sh"
tokens = shlex.split(submit_path.read_text().replace("\\\n", " "))
command = shlex.join(tokens)
safety = validate_regional_iris_command(
    command,
    expected_region="us-central1",
    expected_zone="us-central1-a",
    expected_bucket_prefix="gs://marin-us-central1",
)
if not safety.ok:
    raise ValueError(json.dumps(safety.to_dict()))
required = {
    "--cpu": "1",
    "--memory": "4GB",
    "--max-retries": "0",
    "--max-preemption-retries": "0",
    "--priority": "interactive",
    "--job-name": "tpp10-target-math-completion",
}
for flag, value in required.items():
    if tokens.count(flag) != 1 or tokens[tokens.index(flag) + 1] != value:
        raise ValueError(f"The launch must contain {flag} {value}")
if "--no-preemptible" not in tokens:
    raise ValueError("The CPU parent must use explicit non-preemptible placement")
relative_spec = str((directory / "spec.json").relative_to(workspace))
if tokens[-5:] != ["-m", "experiments.domain_phase_mix.evaluate_tpp10_target_math_complete", "--spec", relative_spec, "--submit"]:
    raise ValueError("The worker command does not select the frozen final evaluation release")
includes = [tokens[index + 1] for index, value in enumerate(tokens) if value == "--bundle-include"]
expected_includes = [
    relative_spec,
    "experiments/domain_phase_mix/tpp10_domain_sweeps_assets/*",
    "experiments/domain_phase_mix/starcoder_tpp10_assets/*",
]
if includes != expected_includes:
    raise ValueError("The three required bundle includes changed")
excludes = [tokens[index + 1] for index, value in enumerate(tokens) if value == "--exclude"]
exclude = re.compile("|".join(f"(?:{pattern})" for pattern in excludes))
if exclude.search(relative_spec):
    raise ValueError("The bundle exclusion removes the final spec")
command_receipt = {
    "checked_at": datetime.now(UTC).isoformat(),
    "status": "command_template_verified",
    "submission_performed": False,
    "command": command,
    "submit_script_sha256": file_sha256(submit_path),
    "region_check": safety.to_dict(),
    "includes": includes,
    "excludes": excludes,
}
(directory / "command_preflight.json").write_text(json.dumps(command_receipt, indent=2) + "\n")
if args.command_only:
    print(json.dumps({"status": "command_template_verified", "bundle_checked": False, "submission_performed": False}))
    raise SystemExit(0)
spec_path = directory / "spec.json"
if not spec_path.exists():
    raise SystemExit("Final spec is absent. Build it after all four final training checkpoints and first-stage receipts verify.")
spec = json.loads(spec_path.read_text())
evaluation.validate_spec(spec)
if len(spec["endpoints"]) != 15 or len(spec["first_stage_result_canonical_sha256"]) != 11 or spec["omitted_requests"]:
    raise ValueError("The release must contain fifteen targets, reusing eleven results and adding four")
for pattern in includes:
    if not any(path.is_file() for path in workspace.glob(pattern)):
        raise ValueError(f"A required include matches no files: {pattern}")
bundle = create_workspace_zip(workspace, exclude=exclude, extra_includes=includes)
with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
    names = archive.namelist()
    if json.loads(archive.read(relative_spec)) != spec:
        raise ValueError("Bundled spec differs from the validated local spec")
    pins = {
        **spec["code_sha256"],
        str(Path(evaluation.__file__).resolve().relative_to(workspace)): spec["wrapper_code_sha256"],
        str(Path(evaluation.first_stage.__file__).resolve().relative_to(workspace)): spec["first_stage_spec"]["wrapper_code_sha256"],
    }
    tokenizer_root = evaluation.scoring.experiment.ASSETS.resolve().relative_to(workspace)
    pins.update({str(tokenizer_root / name): digest for name, digest in spec["tokenizer_pins"]["files_sha256"].items()})
    for name, digest in pins.items():
        if hashlib.sha256(archive.read(name)).hexdigest() != digest:
            raise ValueError(f"Bundled source or tokenizer differs from its frozen pin: {name}")
    expected_agent_paths = {relative_spec}
    if {name for name in names if name.startswith(".agents/")} != expected_agent_paths:
        raise ValueError("Unexpected research artifacts included in the native workspace bundle")
    for name in names:
        if exclude.search(name):
            raise ValueError(f"Native bundle contains an excluded file: {name}")
receipt = {
    "checked_at": datetime.now(UTC).isoformat(),
    "status": "native_bundle_verified",
    "submission_performed": False,
    "spec_sha256": spec["spec_sha256"],
    "first_stage_spec_sha256": spec["first_stage_spec"]["spec_sha256"],
    "bundle_sha256": hashlib.sha256(bundle).hexdigest(),
    "bytes": len(bundle),
    "files": len(names),
    "verified_pinned_files": pins,
    "reused_target_measurements": 11,
    "new_target_evaluations": 4,
    "command_preflight_sha256": file_sha256(directory / "command_preflight.json"),
    "note": "This validates the native bundle without uploading or submitting. Re-run immediately before launch if the checkout changes.",
}
(directory / "bundle_preflight.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({key: receipt[key] for key in ("status", "spec_sha256", "bundle_sha256", "bytes", "files", "submission_performed")}))
PY

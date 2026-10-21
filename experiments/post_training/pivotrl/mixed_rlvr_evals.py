# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Export and evaluate the mixed-RLVR arms' checkpoints (marin #10214) on the October 5 verified paired-eval stack.

    uv run --frozen python -m experiments.post_training.pivotrl.mixed_rlvr_evals export <arm> <step> ... [--submit]
    uv run --frozen python -m experiments.post_training.pivotrl.mixed_rlvr_evals protect <arm> <step> ... [--complete]
    uv run --no-sync python -m experiments.post_training.pivotrl.mixed_rlvr_evals eval <arm> <step> ... [--submit]

``<arm>`` is a receipt label written by :mod:`experiments.post_training.pivotrl.mixed_rlvr_arms` (``a``, ``b-seed43``,
...). ``export`` converts ``global_step_<step>`` to HF safetensors with the training runtime's export job, at the policy
geometry it was saved with; run it from the ``--runtime`` checkout. ``eval`` submits Evalchemy (AIME24, GPQA Diamond,
MMLU-Pro, Math500) and the two Harbor suites (SWE-bench Verified, OT-TBLite) through the October 5 verified policy, one
inference server per benchmark, or one benchmark with ``--group``; run it from the ``--eval-workspace`` checkout. Each
eval job waits for its export job behind :mod:`experiments.post_training.pivotrl.mixed_rlvr_export_barrier`, or for
``<out>/<arm>-step<step>/export-verified.json`` when the export was checked by hand. Without ``--submit`` both run as
dry runs. Receipts land in ``<out>/<arm>-step<step>/``.

The evaluation workspace is marin at ``dml/eval-october17-verified-pivot-mixed-rlvr`` (the
``held/eval-october17-verified`` stack plus the fixes that run exposed) with its own environment; this module shells out
to it and reads only the benchmark configs beside this file, which are the October 5 cohort's verified-policy configs.
"""

import argparse
import contextlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml
from dotenv import dotenv_values

HERE = Path(__file__).resolve().parent
CONFIGS = HERE / "mixed_rlvr_evals"
REFERENCE = json.loads((CONFIGS / "reference.json").read_text())
EVAL_VERSION = "eval-policy-2026-10-05-verified"
EVALCHEMY = ["aime24", "gpqa-diamond", "mmlu-pro", "math500"]
HARBOR = ["swebench-verified", "ot-tblite-recovery"]
TOKENIZER = "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.10.05"
TOKENIZER_REVISION = "8c76b22afc38f5e95360c83ca88008f00b5ae1e0"
DAYTONA_SECRET = "DAYTONA_EVAL_API_KEY"


def training_config(out: Path, arm: str) -> dict:
    path = out / arm / "launch.yaml"
    if not path.exists():
        raise SystemExit(f"no validated launch for arm {arm!r} under {out}")
    return yaml.safe_load(path.read_text())


def receipts(out: Path, arm: str, step: int) -> Path:
    directory = out / f"{arm}-step{step}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def export_target(config: dict, step: int) -> str:
    return config["artifacts"]["export_root"] + f"/global_step_{step}/policy"


def export_job_name(config: dict, step: int) -> str:
    return f"{config['run']['id']}-step{step}-hf-export"


def credentials(repo: Path) -> dict[str, str]:
    values = dotenv_values(repo / ".env")
    env = {key: value for key, value in values.items() if value}
    env.update(
        GOOGLE_APPLICATION_CREDENTIALS=str(Path.home() / ".config/gcloud/application_default_credentials.json"),
        WANDB_ENTITY="marin-community",
        AWS_ACCESS_KEY_ID=values["CW_KEY_ID"],
        AWS_SECRET_ACCESS_KEY=values["CW_KEY_SECRET"],
        AWS_ENDPOINT_URL="https://cwobject.com",
        AWS_DEFAULT_REGION="US-EAST-02A",
        FSSPEC_S3='{"endpoint_url":"https://cwobject.com","config_kwargs":{"s3":{"addressing_style":"virtual"}}}',
    )
    return env


def runtime_imports(runtime: Path, repo: Path) -> None:
    os.environ.update(credentials(repo))
    for part in ("skyrl-train", "skyrl-gym", "."):
        sys.path.insert(0, str(runtime / part))
    # The launcher lives in the runtime checkout, which is not a Marin dependency.
    from rigging.filesystem.s3_compat import configure_coreweave_s3  # noqa: PLC0415  # runtime checkout

    configure_coreweave_s3()
    if Path.cwd() != runtime:
        raise SystemExit(f"run from {runtime}")


def export_spec(out: Path, arm: str, step: int, cluster: str | None, cluster_config: Path):
    from cloud.iris import export_hf_checkpoint as exporter  # noqa: PLC0415  # runtime checkout

    config = training_config(out, arm)
    trainer = yaml.safe_load((out / arm / "resolved.yaml").read_text())["skyrl"]["trainer"]
    placement = trainer["placement"]
    export_args = exporter.argument_parser().parse_args(
        [
            "--launch-config",
            str(out / arm / "launch.yaml"),
            "--ckpt_path",
            config["artifacts"]["checkpoint_root"],
            "--step",
            str(step),
            "--model_path",
            trainer["policy"]["model"]["path"],
            "--num-nodes",
            str(placement["policy_num_nodes"]),
            "--gpus-per-node",
            str(placement["policy_num_gpus_per_node"]),
            "--gpu-variant",
            "H100",
            "--cluster",
            yaml.safe_load(cluster_config.read_text())["name"],
            "--cluster-config",
            str(cluster_config),
            "--parent-cluster-config",
            str(cluster_config),
            "--target-cluster",
            cluster or config["iris"]["target_cluster"],
            "--priority",
            "interactive",
            "--export_path",
            config["artifacts"]["export_root"],
            "--job-name",
            export_job_name(config, step),
            "--no-wait",
        ]
    )
    return exporter.manual_spec(export_args, exporter.argument_parser())


def protect(out: Path, arm: str, step: int, complete: bool, cluster_config: Path) -> None:
    """Record the export request on the checkpoint; trainer cleanup keeps steps whose request is not complete."""
    from skyrl_train.hf_export import write_hf_export_request  # noqa: PLC0415  # runtime checkout
    from skyrl_train.hf_export_schema import HFExportStatus  # noqa: PLC0415  # runtime checkout

    request = export_spec(out, arm, step, None, cluster_config).request
    status = HFExportStatus.COMPLETE if complete else HFExportStatus.IN_PROGRESS
    print("PROTECTION", status.value, write_hf_export_request(request.with_status(status)))


def export(out: Path, arm: str, step: int, submit: bool, cluster: str | None, cluster_config: Path) -> None:
    from cloud.iris.iris_backend import _iris_submission_state, launch  # noqa: PLC0415  # runtime checkout
    from cloud.iris.launch import execute_launch  # noqa: PLC0415  # runtime checkout
    from cloud.iris.launch_config import load_launch_config  # noqa: PLC0415  # runtime checkout
    from omegaconf import OmegaConf  # noqa: PLC0415  # runtime checkout

    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise SystemExit("the runtime checkout has uncommitted changes")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    config = training_config(out, arm)
    directory = receipts(out, arm, step)
    spec = export_spec(out, arm, step, cluster, cluster_config)
    cfg = load_launch_config(Path(spec.launch_config_path))
    cfg.runtime.launcher_commit = commit
    # The export head clears and republishes the Ray rendezvous; keep it away from the live trainer's.
    cfg.ray.rendezvous_dir = f"{cfg.ray.rendezvous_dir}/export-{step}"
    cfg.ray.log_dir = f"{cfg.ray.log_dir}/export-{step}"
    path = directory / "export-launch.yaml"
    OmegaConf.save(cfg, path)
    state = _iris_submission_state(path, cfg)
    state.dry_run = True
    with (
        (directory / "export-dry.log").open("w") as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
    ):
        launch(state, commit)
    print("PREPARED", arm, step, export_target(config, step))
    if not submit:
        return
    receipt = directory / "export-submitted.json"
    if receipt.exists():
        print("ALREADY_SUBMITTED", receipt)
        return
    submit_log = directory / "export.log"
    if submit_log.exists():
        raise SystemExit(f"inspect {submit_log} before retrying an ambiguous export submission")
    with submit_log.open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        result = execute_launch(path)
    if result.state.value != "submitted":
        raise SystemExit(str(result))
    protect(out, arm, step, complete=False, cluster_config=cluster_config)
    receipt.write_text(
        json.dumps(
            {"job_id": result.iris_job_id, "step": step, "target": export_target(config, step), "commit": commit},
            indent=2,
        )
        + "\n"
    )
    print("SUBMITTED", result.iris_job_id)


def eval_commands(out: Path, arm: str, step: int, dependency: str, cluster: str | None, workspace: Path) -> list[dict]:
    config = training_config(out, arm)
    directory = receipts(out, arm, step)
    launcher = directory / "launch_after_export.py"
    launcher.write_text((HERE / "mixed_rlvr_export_barrier.py").read_text().replace("@DEPENDENCY@", dependency))
    commands = []
    # One group per benchmark: each gets its own inference server, so a preemption costs only that benchmark.
    for group in (*EVALCHEMY, *HARBOR):
        model = dict(REFERENCE[group]["model"])
        model.update(
            name=f"{config['run']['id']}-step{step}",
            location=export_target(config, step),
            revision=None,
            tokenizer=TOKENIZER,
            tokenizer_revision=TOKENIZER_REVISION,
        )
        model["resource_hint"] = {**model["resource_hint"], "disk": "500g", "memory": "256g"}
        model_path = directory / f"{group}-model.yaml"
        model_path.write_text(yaml.safe_dump(model, sort_keys=False))
        kind = "evalchemy" if group in EVALCHEMY else "harbor"
        command = [
            "uv",
            "run",
            "--no-sync",
            "python",
            str(launcher),
            "launch",
            "--cluster-config",
            str(workspace / "lib/iris/config/marin.yaml"),
            "--model-config",
            str(model_path),
            "--platform",
            "gpu",
            "--accelerator",
            "H100x8",
            "--federated_cluster",
            cluster or config["iris"]["target_cluster"],
            "--priority",
            "interactive",
            "--records-prefix",
            "s3://marin-us-east-02a/marin/evals",
            "--version",
            EVAL_VERSION,
            "--description",
            f"Mixed-RLVR pivot arm {arm.upper()} step {step} (#10214); October 5 verified policy; thinking enabled.",
            "--no-wait",
            f"--{kind}-config",
            str(CONFIGS / f"{group}.yaml"),
        ]
        commands.append({"group": group, "names": [group], "command": command})
    (directory / "eval-commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    return commands


def daytona_key() -> str:
    """The Daytona credential for the Harbor suites, from the environment or the Gmail gcloud account's secret."""
    if os.environ.get("DAYTONA_API_KEY"):
        return os.environ["DAYTONA_API_KEY"]
    # The secret is readable by the Gmail gcloud credential, not the active openathena.ai account.
    accounts = subprocess.run(
        ["gcloud", "auth", "list", "--filter=account:gmail.com", "--format=value(account)"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    if len(accounts) != 1:
        raise SystemExit("set DAYTONA_API_KEY or keep exactly one Gmail gcloud credential for the Daytona secret")
    secret = subprocess.run(
        ["gcloud", "secrets", "versions", "access", "--secret", DAYTONA_SECRET, "latest", "--account", accounts[0]],
        capture_output=True,
        text=True,
    )
    if secret.returncode:
        raise SystemExit("Daytona credential retrieval failed")
    return secret.stdout.strip()


def evaluate(
    out: Path,
    arm: str,
    step: int,
    submit: bool,
    only_group: str | None,
    cluster: str | None,
    workspace: Path,
    venv: Path,
    repo: Path,
) -> None:
    if Path.cwd() != workspace:
        raise SystemExit(f"run from {workspace}")
    directory = receipts(out, arm, step)
    export_receipt = directory / "export-submitted.json"
    dependency = json.loads(export_receipt.read_text())["job_id"] if export_receipt.exists() else "@UNSUBMITTED@"
    if (directory / "export-verified.json").exists():
        dependency = "@VERIFIED@"
    if submit and dependency == "@UNSUBMITTED@":
        raise SystemExit("submit the export first")
    env = {**os.environ, **credentials(repo)}
    env.update(PYTHONPATH=f"{workspace}:{workspace}/lib/marin/src", UV_PROJECT_ENVIRONMENT=str(venv))
    if submit and (only_group is None or only_group in HARBOR):
        env["DAYTONA_API_KEY"] = daytona_key()
    for item in eval_commands(out, arm, step, dependency, cluster, workspace):
        if only_group and only_group != item["group"]:
            continue
        receipt = directory / f"{item['group']}-eval-submitted.json"
        if submit and receipt.exists():
            print("ALREADY_SUBMITTED", arm, step, item["group"])
            continue
        log = directory / f"{item['group']}-{'submit' if submit else 'dry'}.log"
        if submit and log.exists():
            raise SystemExit(f"inspect {log} before retrying an ambiguous eval submission")
        command = item["command"] + ([] if submit else ["--dry-run"])
        with log.open("w") as output:
            result = subprocess.run(command, cwd=workspace, env=env, stdout=output, stderr=subprocess.STDOUT)
        if result.returncode:
            print(log.read_text()[-4000:])
            raise SystemExit(f"eval CLI failed; inspect {log}")
        if submit:
            match = re.search(r"submitted group (\S+) \((\d+) evals", log.read_text())
            if not match or int(match[2]) != len(item["names"]):
                raise SystemExit(f"unexpected submission output in {log}")
            receipt.write_text(
                json.dumps(
                    {**item, "group_id": match[1], "job_id": "/dml/eval-" + match[1], "dependency": dependency}, indent=2
                )
                + "\n"
            )
        print("SUBMITTED" if submit else "DRY_RUN_OK", arm, step, item["group"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("export", "protect", "eval"))
    parser.add_argument("arm", help="receipt label from mixed_rlvr_arms, such as a or b-seed43")
    parser.add_argument("step", type=int)
    parser.add_argument("--out", type=Path, required=True, help="the receipt directory mixed_rlvr_arms wrote to")
    parser.add_argument("--runtime", type=Path, required=True, help="MarinSkyRL checkout the arms trained with")
    parser.add_argument(
        "--eval-workspace", type=Path, required=True, help="marin checkout of the October 5 paired-eval stack"
    )
    parser.add_argument("--eval-venv", type=Path, required=True, help="that checkout's uv environment")
    parser.add_argument(
        "--cluster-config",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "lib/iris/config/marin.yaml",
        help="Iris hub config whose cached login submits export jobs",
    )
    parser.add_argument("--submit", action="store_true")
    parser.add_argument(
        "--complete", action="store_true", help="protect: release the checkpoint after its export succeeded"
    )
    parser.add_argument("--group", choices=(*EVALCHEMY, *HARBOR))
    # An export needs the saved policy geometry (10 nodes); cw-rno2a usually has it free.
    parser.add_argument("--export-cluster", choices=("cw-us-east-02a", "cw-rno2a"))
    # Overrides the arm's training cluster for an eval whose peer will not accept it.
    parser.add_argument("--eval-cluster", choices=("cw-us-east-02a", "cw-rno2a"))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    cluster_config = args.cluster_config.resolve()
    if args.action == "export":
        runtime_imports(args.runtime, repo)
        export(args.out, args.arm, args.step, args.submit, args.export_cluster, cluster_config)
    elif args.action == "protect":
        runtime_imports(args.runtime, repo)
        protect(args.out, args.arm, args.step, args.complete, cluster_config)
    else:
        evaluate(
            args.out,
            args.arm,
            args.step,
            args.submit,
            args.group,
            args.eval_cluster,
            args.eval_workspace,
            args.eval_venv,
            repo,
        )


if __name__ == "__main__":
    main()

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit a file-backed, shared-serving non-judge timing component from the local checkout."""

import argparse
import hashlib
import json
from pathlib import Path

from iris.cli.connect import open_iris_client
from iris.rpc.proto_display import priority_band_value
from marin.evaluation.hardware import Platform
from marin.evaluation.model_config import load_model_config

from experiments.evaluation.evals import EvalchemyDefinition, HarborDefinition
from experiments.evaluation.launch import (
    EVALUATION_CONTROLLER_CLUSTER,
    LaunchSpec,
    launch_group,
    prepare_evaluation_batch,
)
from experiments.evaluation.policy_finalization import load_policy_finalization

NONJUDGE = (
    "NUPA",
    "aime24",
    "cruxeval",
    "gpqa-diamond",
    "gsm8k-0shot",
    "humanevalplus",
    "ifbench",
    "math500",
    "mbppplus",
    "mmlu-pro",
    "mrcr",
    "triviaqa",
    "truthfulqa",
)
HARBOR_STANDARD = ("bfclparity-pi", "ds-1000-local", "ot-tblite-recovery", "tb2-recovery")
COMPONENTS = {
    "nonjudge": NONJUDGE,
    "harbor-standard": HARBOR_STANDARD,
    "olympiad": ("olympiadbench",),
    "bixbench": ("bixbench-pi",),
    "swebench": ("swebench-verified",),
    "tau3": ("tau3-pi",),
    "excluded-nonjudge": ("piqa", "winogrande", "boolq"),
    "finance": ("financebench",),
    "sotopia": ("sotopia-hard",),
}
PREFIX = "s3://marin-us-east-02a/marin/experiments/mini-eval-policy/runtime-20261030/records"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--configs-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--component", choices=list(COMPONENTS), default="nonjudge")
    parser.add_argument("--judge-model-config", type=Path)
    parser.add_argument("--finalization-config", type=Path)
    args = parser.parse_args()
    if args.component in {"olympiad", "finance", "sotopia"} and args.finalization_config is None:
        raise ValueError("Olympiad timing requires the canonical answer-equivalence finalization config")
    if args.component == "bixbench" and args.judge_model_config is None:
        raise ValueError("BixBench timing requires its canonical co-hosted verifier judge config")
    manifest = json.loads((args.configs_dir / "manifest.json").read_text())
    definitions = []
    harbor_definitions = []
    selected = []
    included = COMPONENTS[args.component]
    ordered = sorted(
        manifest["benchmark_manifests"],
        key=lambda item: included.index(item["benchmark"]) if item["benchmark"] in included else len(included),
    )
    for item in ordered:
        if item["benchmark"] not in included:
            continue
        path = args.configs_dir / (item["benchmark"] + ".yaml")
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["config_sha256"]:
            raise ValueError(f"Timing config changed from frozen manifest: {path}")
        if args.component in {"nonjudge", "olympiad", "finance", "excluded-nonjudge"}:
            definitions.append(EvalchemyDefinition(path.stem, path))
        else:
            harbor_definitions.append(HarborDefinition(path.stem, path))
        selected.append(item)
    grading = Path(__file__).parent / "runtime-configs"
    selected_names = {item["benchmark"] for item in selected}
    if "gsm8k-0shot" in selected_names:
        target = json.loads((grading / "gsm8k-grading-provenance.json").read_text())
        actual = hashlib.sha256(Path("vendor/evalchemy/eval/lm_eval_tasks/gsm8k/utils.py").read_bytes()).hexdigest()
        if actual != target["target_sha256"]:
            raise ValueError("GSM8K grader differs from the frozen target; audit before launching")
    if "NUPA" in selected_names:
        target = json.loads((grading / "nupa-grading-provenance.json").read_text())
        for name, expected in target["files"].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"NUPA grading source changed: {name}; audit before launching")
    if manifest["adaptive_path_replay_only"]:
        raise ValueError("Historical adaptive paths cannot be launched as fresh adaptive experiments")
    spec = LaunchSpec(
        model=load_model_config(args.model_config),
        evals=(),
        evalchemy_definitions=tuple(definitions),
        harbor_definitions=tuple(harbor_definitions),
        platform=Platform.GPU,
        accelerator=None,
        limit=None,
        records_prefix=PREFIX,
        submission_cluster=EVALUATION_CONTROLLER_CLUSTER,
        federated_cluster="cw-rno2a",
        priority_band=priority_band_value("batch"),
        max_concurrent=1,
        judge_model=load_model_config(args.judge_model_config) if args.judge_model_config else None,
        policy_finalization=load_policy_finalization(args.finalization_config) if args.finalization_config else None,
        run_id=args.run_id,
        description=(
            f"Unlabeled mini-policy matched {args.component} timing component; "
            "fixed source selection; no policy attestation"
        ),
    )
    batch = prepare_evaluation_batch(spec)
    with open_iris_client(cluster_name=EVALUATION_CONTROLLER_CLUSTER, workspace=Path.cwd()) as client:
        submitted = launch_group(batch, client)
        receipt = {
            "group": submitted.group_id,
            "job": str(submitted.job.job_id),
            "model": manifest["model"],
            "method": manifest["method"],
            "component": args.component,
            "suite_budget": manifest["budget"],
            "component_task_count": sum(item["task_count"] for item in selected),
            "benchmarks": [item["benchmark"] for item in selected],
            "max_concurrent": 1,
            "evaluations": [
                {"name": evaluation.identity.eval_ref.name, "root": evaluation.identity.output_dir}
                for evaluation in batch.evaluations
            ],
            "manifest_sha256": hashlib.sha256((args.configs_dir / "manifest.json").read_bytes()).hexdigest(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2))
        print(
            json.dumps(
                {
                    key: receipt[key]
                    for key in ["group", "job", "method", "suite_budget", "component_task_count", "benchmarks"]
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()

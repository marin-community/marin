# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch isolated serve/eval jobs for a published policy and its repeats."""

import argparse
from pathlib import Path
from subprocess import run
from tempfile import TemporaryDirectory

import yaml
from marin.evaluation.eval_policy import (
    POLICIES,
    RUNTIME_COMMITS,
    SEPTEMBER_16_VERSION,
    SEPTEMBER_24_VERSION,
)
from marin.evaluation.model_config import load_model_config

ROOT = Path(__file__).resolve().parents[1]
HARBOR_CONFIGS = ROOT / "experiments/evaluation/configs/harbor"
HARBOR_REGISTRY_16 = (
    "https://raw.githubusercontent.com/marin-community/harbor/" "7b18505a56e5624f55887e3b20f4de452f698a7a/registry.json"
)


def _harbor_source(version: str, name: str, artifact_dir: Path | None) -> Path:
    if version == SEPTEMBER_24_VERSION:
        assert artifact_dir is not None
        stem = "sotopia-hard-standalone" if name == "sotopia-hard" else name
        return artifact_dir / "harbor-configs" / f"{stem}.yaml"
    return HARBOR_CONFIGS / f"{name}.yaml"


def _harbor_config(
    version: str, name: str, artifact_dir: Path | None, sotopia_dataset_dir: Path | None, output: Path
) -> Path:
    source = _harbor_source(version, name, artifact_dir)
    if name == "sotopia-hard":
        assert sotopia_dataset_dir is not None
        dataset_link = output.parent / "datasets" / "sotopia-hard"
        dataset_link.parent.mkdir(exist_ok=True)
        dataset_link.symlink_to(sotopia_dataset_dir, target_is_directory=True)
        output.write_text(source.read_text())
        return output
    if version != SEPTEMBER_16_VERSION or name not in {"simpleqa-recovery", "ds-1000-local"}:
        return source
    config = yaml.safe_load(source.read_text())
    dataset = "simpleqa" if name == "simpleqa-recovery" else "ds-1000"
    config["datasets"] = [{"name": dataset, "version": "mini-200", "registry_url": HARBOR_REGISTRY_16}]
    if dataset == "ds-1000":
        config.setdefault("environment", {})["force_build"] = True
    output.write_text(yaml.safe_dump(config, sort_keys=False))
    return output


def launch_policy(
    version: str,
    model_config: Path,
    artifact_dir: Path | None,
    sotopia_dataset_dir: Path | None,
    cluster: str,
    selected_evals: tuple[str, ...] | None = None,
    dry_run: bool = False,
) -> None:
    """Submit the policy with a separate H100 serve for every benchmark launch."""
    unknown = set(selected_evals or ()) - POLICIES[version].keys()
    if unknown:
        raise ValueError(f"evaluations not in {version}: {sorted(unknown)}")
    evaluations = [
        (name, policy) for name, policy in POLICIES[version].items() if selected_evals is None or name in selected_evals
    ]
    if not evaluations:
        raise ValueError("select at least one evaluation")
    if any(policy.mechanism == "evalchemy" for _, policy in evaluations):
        commit = RUNTIME_COMMITS[version]["evalchemy"]
        raise ValueError(
            f"{version} requires historical Evalchemy commit {commit}; "
            "run it from a Marin checkout that predates Evalchemy vendoring"
        )
    needs_artifact = version == SEPTEMBER_24_VERSION
    if needs_artifact and artifact_dir is None:
        raise ValueError("selected September 24 evaluations require the pinned artifact directory from issue #9409")
    if any(name == "sotopia-hard" for name, _ in evaluations) and sotopia_dataset_dir is None:
        raise ValueError("SOTOPIA-hard requires its pinned dataset directory")
    if not model_config.is_file():
        raise ValueError(f"model config does not exist: {model_config}")
    h100_count = load_model_config(model_config).resource_hint.gpu.get("H100")
    if h100_count is None:
        raise ValueError(f"model config must declare an H100 resource hint: {model_config}")
    if sotopia_dataset_dir is not None and not sotopia_dataset_dir.is_dir():
        raise ValueError(f"SOTOPIA-hard dataset does not exist: {sotopia_dataset_dir}")
    if sotopia_dataset_dir is not None and not sotopia_dataset_dir.resolve().is_relative_to(ROOT):
        raise ValueError("SOTOPIA-hard dataset must be inside the Marin workspace for Harbor preflight")
    for name, _ in evaluations:
        source = _harbor_source(version, name, artifact_dir)
        if not source.is_file():
            raise ValueError(f"policy config does not exist: {source}")
    with TemporaryDirectory(prefix="marin-eval-policy-", dir=ROOT / "eval_policy") as temporary:
        directory = Path(temporary)
        for name, _ in evaluations:
            config = directory / f"{name}.yaml"
            config = _harbor_config(version, name, artifact_dir, sotopia_dataset_dir, config)
            command = [
                "uv",
                "run",
                "python",
                "-m",
                "experiments.evaluation.cli",
                "launch",
                "--model-config",
                str(model_config),
                "--harbor-config",
                str(config),
                "--accelerator",
                f"H100x{h100_count}",
                "--federated_cluster",
                cluster,
                "--priority",
                "interactive",
                "--version",
                version,
                "--dry-run" if dry_run else "--no-wait",
            ]
            run(command, check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", choices=(SEPTEMBER_16_VERSION, SEPTEMBER_24_VERSION))
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--sotopia-dataset-dir", type=Path)
    parser.add_argument("--federated-cluster", default="cw-rno2a")
    parser.add_argument("--evals", help="Comma-separated subset of benchmarks; defaults to the full policy")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print every launch without submitting")
    args = parser.parse_args()
    launch_policy(
        args.version,
        args.model_config.resolve(),
        args.artifact_dir.resolve() if args.artifact_dir else None,
        args.sotopia_dataset_dir.resolve() if args.sotopia_dataset_dir else None,
        args.federated_cluster,
        tuple(name.strip() for name in args.evals.split(",") if name.strip()) if args.evals else None,
        args.dry_run,
    )


if __name__ == "__main__":
    main()

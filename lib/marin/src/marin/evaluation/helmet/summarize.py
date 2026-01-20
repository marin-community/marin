# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Literal

import fsspec
import yaml


def _list_files(dir_path: str) -> list[str]:
    fs, root = fsspec.core.url_to_fs(dir_path)
    return [fs.unstrip_protocol(p) for p in fs.find(root)]


def _download_file(remote_path: str, local_path: str) -> None:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with fsspec.open(remote_path, "rb") as src, open(local_path, "wb") as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)


def _csv_values(value: Any, *, n: int, cast: type[int] | type[str]) -> list[Any]:
    if isinstance(value, str):
        parts = [p for p in value.split(",") if p != ""]
        values = [cast(p) for p in parts]
    elif isinstance(value, int):
        values = [cast(value)]
    elif isinstance(value, list):
        values = [cast(v) for v in value]
    else:
        raise TypeError(f"Unsupported csv field type: {type(value)}")

    if len(values) == 1 and n > 1:
        return values * n
    if len(values) != n:
        raise ValueError(f"Expected {n} values, got {len(values)}: {values}")
    return values


def _load_helmet_collect_results_module(repo_dir: str) -> Any:
    # Import HELMET's collect_results module from disk for maximum parity (no sys.path mutations).
    import importlib.util

    collect_results_path = os.path.join(repo_dir, "scripts", "collect_results.py")
    if not os.path.exists(collect_results_path):
        raise FileNotFoundError(f"HELMET collect_results.py not found at {collect_results_path}")

    spec = importlib.util.spec_from_file_location("helmet_collect_results", collect_results_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load HELMET collect_results.py from {collect_results_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dataset_configs_from_helmet(
    *,
    repo_dir: str,
    evals: list[str],
    config_variant: Literal["full", "short"],
    use_chat_template: bool,
    helmet_collect_results: Any,
) -> list[dict[str, Any]]:
    defaults = helmet_collect_results.arguments()
    suffix = "" if config_variant == "full" else "_short"

    dataset_configs: list[dict[str, Any]] = []
    for eval_name in evals:
        cfg_path = os.path.join(repo_dir, "configs", f"{eval_name}{suffix}.yaml")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Missing HELMET config: {cfg_path}")
        cfg = yaml.safe_load(open(cfg_path))

        datasets = str(cfg["datasets"]).split(",")
        test_files = str(cfg["test_files"]).split(",")
        if len(datasets) != len(test_files):
            raise ValueError(f"Mismatched datasets/test_files in {cfg_path}: {len(datasets)} vs {len(test_files)}")

        input_max_lengths = _csv_values(cfg["input_max_length"], n=len(datasets), cast=int)
        generation_max_lengths = _csv_values(cfg["generation_max_length"], n=len(datasets), cast=int)

        max_test_samples = int(cfg.get("max_test_samples", defaults.max_test_samples))
        shots = int(cfg.get("shots", defaults.shots))
        do_sample = bool(cfg.get("do_sample", defaults.do_sample))
        generation_min_length = int(cfg.get("generation_min_length", defaults.generation_min_length))
        temperature = float(cfg.get("temperature", defaults.temperature))
        top_p = float(cfg.get("top_p", defaults.top_p))
        popularity_threshold = int(cfg.get("popularity_threshold", defaults.popularity_threshold))

        if not do_sample and temperature != 0.0:
            temperature = 0.0

        for dataset, test_file, in_len, gen_len in zip(
            datasets, test_files, input_max_lengths, generation_max_lengths, strict=True
        ):
            test_name = os.path.splitext(os.path.basename(test_file))[0]
            dataset_configs.append(
                {
                    "dataset": dataset,
                    "test_name": test_name,
                    "input_max_length": int(in_len),
                    "generation_max_length": int(gen_len),
                    "max_test_samples": max_test_samples,
                    "shots": shots,
                    "do_sample": do_sample,
                    "generation_min_length": generation_min_length,
                    "temperature": temperature,
                    "top_p": top_p,
                    "use_chat_template": bool(use_chat_template),
                    "popularity_threshold": popularity_threshold,
                }
            )

    return dataset_configs


def _read_json(path: str) -> Any:
    with fsspec.open(path) as f:
        return json.load(f)


@dataclass(frozen=True)
class HelmetSummarizeConfig:
    model_name: str
    model_path: str
    # these all exist just to get put into a json report and wandb config
    helmet_repo_url: str
    helmet_repo_sha: str
    helmet_data_sha: str
    use_chat_template: bool
    config_variant: Literal["full", "short"]
    tag: str
    seed: int

    # these are real
    eval_output_paths: list[str]
    output_path: str

    wandb_tags: list[str] | None = None
    allow_partial: bool = False
    """Allow summarizing based on whatever outputs exist (useful for smoke tests)."""


def _report_from_local_outputs(
    *,
    config: HelmetSummarizeConfig,
    evals: list[str],
    local_output_dir: str,
) -> dict[str, Any]:
    outputs: list[dict[str, Any]] = []
    for filename in sorted(os.listdir(local_output_dir)):
        if not filename.endswith(".json"):
            continue
        if filename == "marin_metadata.json":
            continue
        path = os.path.join(local_output_dir, filename)
        with open(path, "r") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            continue
        args = payload.get("args")
        averaged = payload.get("averaged_metrics")
        if not isinstance(args, dict) or not isinstance(averaged, dict):
            continue
        outputs.append(
            {
                "path": path,
                "dataset": args.get("datasets"),
                "test_files": args.get("test_files"),
                "demo_files": args.get("demo_files"),
                "input_max_length": args.get("input_max_length"),
                "generation_max_length": args.get("generation_max_length"),
                "max_test_samples": args.get("max_test_samples"),
                "shots": args.get("shots"),
                "use_chat_template": args.get("use_chat_template"),
                "averaged_metrics": averaged,
            }
        )

    return {
        "model_name": config.model_name,
        "model_path": config.model_path,
        "helmet_repo_url": config.helmet_repo_url,
        "helmet_repo_sha": config.helmet_repo_sha,
        "helmet_data_sha": config.helmet_data_sha,
        "use_chat_template": config.use_chat_template,
        "config_variant": config.config_variant,
        "tag": config.tag,
        "seed": config.seed,
        "evals": evals,
        "outputs": outputs,
        "category_scores": {},
        "collect_results_rows": [],
        "mapping_source": "partial (derived from produced json outputs)",
    }


def summarize_helmet(config: HelmetSummarizeConfig) -> None:
    with tempfile.TemporaryDirectory(prefix="helmet_summarize_") as tmpdir:
        repo_dir = os.path.join(tmpdir, "HELMET")
        subprocess.run(["git", "clone", config.helmet_repo_url, repo_dir], check=True)
        subprocess.run(["git", "checkout", config.helmet_repo_sha], check=True, cwd=repo_dir)

        helmet_collect_results = _load_helmet_collect_results_module(repo_dir)
        custom_avgs: dict[str, list[str]] = helmet_collect_results.custom_avgs

        local_output_dir = os.path.join(tmpdir, "output", config.model_name)
        os.makedirs(local_output_dir, exist_ok=True)

        local_metadata: list[dict[str, Any]] = []
        for root in config.eval_output_paths:
            for remote_path in _list_files(root):
                base = os.path.basename(remote_path)
                if not (base.endswith(".json") or base.endswith(".json.score") or base == "marin_metadata.json"):
                    continue
                if base == "marin_metadata.json":
                    local_metadata.append(_read_json(remote_path))
                    continue

                local_path = os.path.join(local_output_dir, base)
                if os.path.exists(local_path):
                    continue
                _download_file(remote_path, local_path)

        evals: list[str] = sorted({e for m in local_metadata for e in m.get("evals", [])})
        if not evals:
            # Fall back to the common HELMET set when metadata is missing.
            evals = ["recall", "rag", "longqa", "summ", "icl", "rerank", "cite"]

        if config.allow_partial:
            report = _report_from_local_outputs(config=config, evals=evals, local_output_dir=local_output_dir)
            category_scores: dict[str, float] = {}
        else:
            dataset_configs = _dataset_configs_from_helmet(
                repo_dir=repo_dir,
                evals=evals,
                config_variant=config.config_variant,
                use_chat_template=config.use_chat_template,
                helmet_collect_results=helmet_collect_results,
            )

            values_by_len: dict[int, dict[str, list[float]]] = {}
            dataset_rows: list[dict[str, Any]] = []
            failed_paths: list[str] = []

            for ds_cfg in dataset_configs:
                # They named their class lower-case `argument`
                args = helmet_collect_results.arguments()
                args.tag = config.tag
                args.seed = config.seed
                args.output_dir = local_output_dir
                args.update(ds_cfg)

                import contextlib
                import io

                with contextlib.redirect_stdout(io.StringIO()):
                    metric_dict = args.get_averaged_metric()
                if metric_dict is None:
                    failed_paths.append(args.get_path())
                    continue

                match = args.get_metric_name()
                if match is None:
                    raise RuntimeError(f"Could not infer metric name for dataset: {args.dataset}")
                dataset_simple, _ = match

                input_len = int(args.input_max_length)
                len_bucket = values_by_len.setdefault(input_len, {})
                for metric_name, value in metric_dict.items():
                    col = f"{dataset_simple} {metric_name}"
                    len_bucket.setdefault(col, []).append(float(value))

                dataset_rows.append(
                    {
                        "dataset": args.dataset,
                        "test_name": args.test_name,
                        "input_max_length": input_len,
                        "path": args.get_path(),
                        "dataset_simple": dataset_simple,
                        "metrics": {k: float(v) for k, v in metric_dict.items()},
                    }
                )

            if failed_paths:
                raise RuntimeError(
                    "Missing expected HELMET outputs; refusing to aggregate partial results. "
                    f"Missing paths (first 20): {failed_paths[:20]}"
                )

            collect_results_rows: list[dict[str, Any]] = []
            for input_len in sorted(values_by_len):
                row: dict[str, Any] = {"input_max_length": input_len}
                for col, vals in values_by_len[input_len].items():
                    row[col] = sum(vals) / len(vals)

                # Match HELMET's `custom_avgs` behavior (computed in insertion order).
                for category, deps in custom_avgs.items():
                    dep_vals: list[float] = []
                    for dep in deps:
                        if dep not in row:
                            raise RuntimeError(f"Missing column '{dep}' needed for category '{category}' at {input_len}")
                        dep_vals.append(float(row[dep]))
                    row[category] = sum(dep_vals) / len(dep_vals) if dep_vals else None

                collect_results_rows.append(row)

            canonical_row = max(collect_results_rows, key=lambda r: int(r["input_max_length"]))
            category_scores = {
                k: float(canonical_row[k]) for k in custom_avgs if k in canonical_row and canonical_row[k] is not None
            }

            report = {
                "model_name": config.model_name,
                "model_path": config.model_path,
                "helmet_repo_url": config.helmet_repo_url,
                "helmet_repo_sha": config.helmet_repo_sha,
                "helmet_data_sha": config.helmet_data_sha,
                "use_chat_template": config.use_chat_template,
                "config_variant": config.config_variant,
                "tag": config.tag,
                "seed": config.seed,
                "evals": evals,
                "outputs": dataset_rows,
                "category_scores": category_scores,
                "collect_results_rows": collect_results_rows,
                "mapping_source": "HELMET scripts/collect_results.py (arguments + custom_avgs)",
            }

    fs, out_root = fsspec.core.url_to_fs(config.output_path)
    fs.makedirs(out_root, exist_ok=True)
    remote_report_path = os.path.join(out_root, "helmet_report.json")

    with tempfile.TemporaryDirectory(prefix="helmet_report_") as tmpdir:
        local_report_path = os.path.join(tmpdir, "helmet_report.json")
        with open(local_report_path, "w") as f:
            json.dump(report, f, indent=2)

        fs.put(local_report_path, remote_report_path)

        import wandb

        run = wandb.init(
            project="marin",
            job_type="eval",
            name=f"{config.model_name}-helmet",
            tags=config.wandb_tags,
            config={
                "model_name": config.model_name,
                "model_path": config.model_path,
                "helmet_repo_sha": config.helmet_repo_sha,
                "helmet_data_sha": config.helmet_data_sha,
                "use_chat_template": config.use_chat_template,
                "config_variant": config.config_variant,
                "tag": config.tag,
                "seed": config.seed,
            },
        )
        log_payload: dict[str, float] = {f"helmet/category/{k}": v for k, v in category_scores.items()}
        wandb.log(log_payload)

        artifact = wandb.Artifact(name=f"{config.model_name}-helmet-report", type="report")
        artifact.add_file(local_report_path)
        run.log_artifact(artifact)
        run.finish()

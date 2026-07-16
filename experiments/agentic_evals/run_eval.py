#!/usr/bin/env python3
"""Worker CLI: run an agentic eval inside the cluster pod.

Adapted from OT-Agent ``eval/local/run_eval.py``. Creates an ``EvalRunner``
that subclasses ``LocalHarborRunner``, wires the configured ``ResultSink``
into ``post_harbor_hook``, and runs the full Ray+vLLM+Harbor lifecycle.

Usage:
    python -m agentic_evals.run_eval \\
        --harbor_config harbor.yaml \\
        --model Qwen/Qwen3-32B \\
        --dataset_path ./tasks \\
        --agent terminus-2
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

from .runtime.runner import LocalHarborRunner
from .runtime.args import add_harbor_env_arg, add_hf_upload_args, add_database_upload_args
from .harness.config import get_harbor_env_from_config
from .results import ResultSink, NoOpResultSink
from .results.local import LocalResultSink
from .results.hf_upload import HFResultSink, derive_benchmark_repo


class EvalRunner(LocalHarborRunner):
    """Local Harbor runner for evaluation."""

    JOB_PREFIX = "eval"
    DEFAULT_EXPERIMENTS_SUBDIR = "eval_runs"
    DEFAULT_N_CONCURRENT = 16
    TPU_SERVE_DEFAULT_CLI_ARGS = ["--enable-prefix-caching"]

    @classmethod
    def create_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Run Harbor evals against a local Ray/vLLM server."
        )
        cls.add_common_arguments(parser)

        parser.add_argument(
            "--dataset",
            help="Harbor dataset slug (e.g., terminal-bench@2.0).",
        )
        parser.add_argument(
            "--dataset_path",
            help="Path to a Harbor task directory.",
        )
        parser.add_argument("--dataset-path", dest="dataset_path", help=argparse.SUPPRESS)

        add_harbor_env_arg(parser, default=None, legacy_names=["--eval-env", "--eval_env"])

        parser.add_argument("--datagen_config", help="Optional datagen YAML.")
        parser.add_argument("--datagen-config", dest="datagen_config", help=argparse.SUPPRESS)

        parser.add_argument("--vllm_model_uri", help="Object-store URI for vLLM weights.")
        parser.add_argument("--vllm-model-uri", dest="vllm_model_uri", help=argparse.SUPPRESS)

        parser.add_argument(
            "--experiments_dir",
            help="Directory for logs + endpoint JSON.",
        )
        parser.add_argument("--experiments-dir", dest="experiments_dir", help=argparse.SUPPRESS)

        parser.add_argument(
            "--refire_filter_error_type",
            "--refire-filter-error-type",
            dest="refire_filter_error_types",
            action="append",
            default=None,
            help="Exception type to delete-and-re-run on a warm-dir re-fire (repeatable).",
        )

        add_hf_upload_args(parser)
        add_database_upload_args(parser)

        return parser

    def get_env_type(self) -> str:
        if self.args.harbor_env:
            return self.args.harbor_env
        return get_harbor_env_from_config(self.args.harbor_config)

    def get_dataset_label(self) -> str:
        return self.args.dataset or self.args.dataset_path or "dataset"

    def get_dataset_for_harbor(self) -> Tuple[Optional[str], Optional[str]]:
        return (self.args.dataset, self.args.dataset_path)

    def validate_args(self) -> None:
        if self.args.dataset and self.args.dataset_path:
            raise ValueError("Specify either --dataset or --dataset-path (not both).")
        if not self.args.dataset and not self.args.dataset_path:
            raise ValueError("Must provide --dataset or --dataset-path.")

    def print_banner(self) -> None:
        args = self.args
        dataset_label = self.get_dataset_label()
        print("=== Local Eval Runner ===")
        print(f"  Model: {args.model}")
        print(f"  Dataset: {dataset_label}")
        print(f"  TP/PP/DP: {args.tensor_parallel_size}/{args.pipeline_parallel_size}/{args.data_parallel_size}")
        print(f"  GPUs: {args.gpus}")
        print("=========================")

    def _build_result_sink(self) -> ResultSink:
        """Build the configured ResultSink from args."""
        args = self.args
        hf_repo = getattr(args, "upload_hf_repo", None)

        if hf_repo:
            return HFResultSink(
                hf_repo_id=hf_repo,
                hf_token=getattr(args, "upload_hf_token", None),
                hf_private=getattr(args, "upload_hf_private", False),
                hf_episodes=getattr(args, "upload_hf_episodes", "last"),
            )

        # Default: write a local result summary
        experiments_dir = self.get_experiments_dir()
        return LocalResultSink(experiments_dir)

    def post_harbor_hook(self) -> None:
        """Publish results via the configured ResultSink."""
        args = self.args

        if args.dry_run:
            print("[upload] Skipping upload because --dry-run was set.")
            return

        job_name = self._harbor_job_name
        jobs_dir_path = getattr(args, "_jobs_dir_path", None)
        if not job_name or jobs_dir_path is None:
            print("[upload] Unable to determine job directory; upload skipped.")
            return

        run_dir = Path(jobs_dir_path) / job_name
        if not run_dir.exists():
            print(f"[upload] Expected Harbor job directory {run_dir} does not exist; upload skipped.")
            return

        sink = self._build_result_sink()
        benchmark_name = derive_benchmark_repo(
            harbor_dataset=args.dataset,
            dataset_path=args.dataset_path,
        )

        try:
            result = sink.publish(
                job_dir=run_dir,
                job_name=job_name,
                model_name=args.model,
                benchmark_name=benchmark_name,
                metadata={"agent": args.agent},
            )
            if result:
                print(f"[upload] Result published: {result}")
        except Exception as e:
            print(f"[upload] Error during result publishing: {e}")


def main() -> None:
    parser = EvalRunner.create_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    runner = EvalRunner(args, repo_root)
    runner.setup()
    runner.run()


if __name__ == "__main__":
    main()

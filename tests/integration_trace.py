# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration test expressed using StepSpec and StepRunner directly.

This is equivalent to integration_test.py but avoids the executor "magic".
"""

import logging
import os
import sys
from collections.abc import Callable
from typing import Any, Generic, TypeVar
from uuid import uuid4

import click
from fray.v1.cluster import create_cluster, set_current_cluster
from fray.v2 import ResourceConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.gpt2 import Gpt2Config
from levanter.trainer import TrainerConfig
from marin.execution.artifact import Artifact
from marin.execution.step_model import StepSpec
from marin.execution.step_runner import StepRunner
from marin.processing.classification.consolidate import FilterConfig, FilterType, consolidate_fn
from marin.processing.classification.dataset_utils import DatasetConfig
from marin.processing.classification.deduplication.dedup_commons import DedupMetadata, DedupMode, deduplicate_fn
from marin.processing.classification.fasttext.train_fasttext import train
from marin.processing.tokenize import lm_data_config
from marin.processing.tokenize.tokenize import TokenizedMetadata, tokenize_fn
from marin.schemas.web.convert import ResiliparseConfig
from marin.training.training import run_levanter_train_lm_fn
from marin.transform.simple_html_to_md.process import html_to_md

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Deferred(Generic[T]):
    def __init__(self, fn: Callable[[], T] | None = None, id: str | None  = None, value: Any = None):
        self.fn = fn
        # TODO(make a separate type, for handling value = None)
        self.value = value
        self.id: str = id or uuid4()

    @staticmethod
    def from_value(v: Any):
      return Deferred(value=v)



class StepContext:
    """Implicit context for a currently running step."""

    def deps(self) -> dict[str, Artifact]: ...

    def output_path(self) -> str: ...

    @classmethod
    def current(cls) -> "StepContext":
      return DeferredStepContext()

    def resolve_id(self, dep_id: str) -> Any:
      return Artifact.load(self.deps()[dep_id])

    def resolve(self, dep: Deferred[T]) -> T:
      """Typed dependency resolution"""
      return self.resolve_id(dep.id)


# make a decoractor for traced functions
def deferred(fn: Callable):
  def wrapper(*args, **kwargs):
    return Deferred(fn=fn, args=args, kwargs=kwargs)
  return wrapper


@deferred
def _html_to_md(input_path: Deferred[str]) -> Deferred[str]:
    ctx = StepContext.current()
    return html_to_md(
        input_path=ctx.resolve(input_path),
        output_path=ctx.output_path(),
        extract_method="resiliparse",
        config=ResiliparseConfig(),
    )

@deferred
def _train_quality(inputs: dict[str, Deferred[str]]):
    fasttext_args = {"lr": 0.001, "minCount": 1, "epoch": 25, "wordNgrams": 2, "dim": 50, "thread": 1}
    ctx = StepContext.current()
    return train(
        datasets=[
            DatasetConfig(input_doc_path=ctx.resolve(inputs["hq"]), label="hq", sampling_rate=1.0),
            DatasetConfig(input_doc_path=ctx.resolve(inputs["lq"]), label="lq", sampling_rate=1.0),
        ],
        output_path=ctx.output_path(),
        fasttext_args=fasttext_args,
    )

@deferred
def _dedup_exact(input_path: Deferred[str]):
    ctx = StepContext.current()
    return deduplicate_fn(
        input_paths=ctx.resolve(input_path),
        output_path=ctx.output_path(),
        mode=DedupMode.EXACT_PARAGRAPH,
        ray_memory=1 * 1024**3,  # 1GB
        ray_num_cpus=1,
    )

@deferred
def _dedup_fuzzy(input_path: Deferred[str]):
    ctx = StepContext.current()
    return deduplicate_fn(
        input_paths=ctx.resolve(input_path),
        output_path=ctx.output_path(),
        mode=DedupMode.FUZZY_DOCUMENT,
        ray_memory=1 * 1024**3,  # 1GB
        ray_num_cpus=1,
    )

@deferred
def _consolidate(inputs: dict[str, Deferred[str]]):
    ctx = StepContext.current()
    return consolidate_fn(
        input_path=ctx.resolve(inputs["hq"]),
        output_path=ctx.output_path(),
        filters=[
            FilterConfig(
                type=FilterType.REMOVE_SPANS,
                attribute_path=Artifact.load(inputs["dedup_exact"], DedupMetadata).data_path,
                name=DedupMode.EXACT_PARAGRAPH,
            ),
            FilterConfig(
                type=FilterType.REMOVE_DOC,
                attribute_path=Artifact.load(inputs["dedup_fuzzy"], DedupMetadata).data_path,
                name=DedupMode.FUZZY_DOCUMENT,
            ),
        ],
    )

@deferred
def _tokenize(inputs: dict[str, Deferred[str]]):
    ctx = StepContext.current()
    return tokenize_fn(
        train_paths=[ctx.resolve(inputs["consolidate"])],
        validation_paths=[],
        cache_path=ctx.output_path(),
        tokenizer="gpt2",
        zephyr_num_cpus=1,
        zephyr_memory=1 * 1024**2,  # 1MB
    )

@deferred
def _train_lm(inputs: dict[str, Deferred[str]]):
    ctx = StepContext.current()
    train_env_vars = {
        "WANDB_API_KEY": "",
        "WANDB_MODE": "disabled",
        "JAX_TRACEBACK_FILTERING": "off",
    }

    pod_config = ResourceConfig.with_cpu()
    return run_levanter_train_lm_fn(
        output_path=ctx.output_path(),
        resources=pod_config,
        env_vars=train_env_vars,
        train_config=TrainLmConfig(
            data=lm_data_config(("training_data", Artifact.load(ctx.resolve(inputs["tokenize"]), TokenizedMetadata))),
            hf_save_steps=1,
            model=Gpt2Config(
                num_layers=2,
                num_heads=2,
                max_seq_len=64,
                hidden_dim=32,
            ),
            trainer=TrainerConfig(
                train_batch_size=8, num_train_steps=2, max_eval_batches=1, require_accelerator=False
            ),
        ),
    )

def create_steps(*, prefix: str, synth_data: str) -> Deferred[str]:
    # ############################################################
    # Transform HTML to text

    hq_step = _html_to_md(
      input_path=Deferred.from_value(os.path.join(synth_data, "pos"))
    )

    lq_step = _html_to_md(
        input_path=Deferred.from_value(os.path.join(synth_data, "neg"))
    )

    quality = _train_quality({"hq": hq_step, "lq": lq_step})
    dedup_exact_step = _dedup_exact(hq_step)
    dedup_fuzzy_step = _dedup_fuzzy(hq_step)
    consolidate_step = _consolidate({"hq": hq_step, "dedup_exact": dedup_exact_step, "dedup_fuzzy": dedup_fuzzy_step})
    tokenize_step = _tokenize({"consolidate": consolidate_step})
    train_step = _train_lm({"tokenize": tokenize_step})
    return train_step


def deferred_to_steps(root: Deferred) -> list[StepSpec]:
    """Trace out deferred execution into a list of explicit steps."""
    # StepSpec(name="", output_path_prefix="", deps=[], hash_attrs={}, override_output_path={}, fn=Callable())
    pending = [root]
    steps = dict[str, StepSpec]
    while pending:
        curr = pending.pop(0)
        if curr.id in steps:
            continue
        steps[curr.id] = StepSpec(
            name=curr.name,
            deps=[d.id for d in curr.deps],
            hash_attrs=curr.hash_attrs,
            override_output_path=curr.override_output_path,
            fn=curr.fn,
        )
        pending.extend(curr.deps)
    return list(steps.values())



@click.command()
@click.option("--prefix", default=None, help="Output path prefix")
@click.option("--dry-run", is_flag=True, default=False, help="Dry run mode")
def main(prefix: str | None, dry_run: bool):
    try:
        bucket_prefix = prefix or "/tmp"

        if "uv run" in " ".join(sys.argv):
            raise RuntimeError("integration_nomagic_test.py must not be launched via `uv run`. Please run it directly.")

        import ray

        ray.init(
            resources={"head_node": 1},
            runtime_env={"working_dir": None},
            num_cpus=os.cpu_count(),
            _memory=1024 * 1024 * 1024 * 1024,  # 1TB
        )
        set_current_cluster(create_cluster("ray"))

        synth_data = "./tests/quickstart-data"

        experiment_prefix = f"{bucket_prefix}/quickstart-tests-nomagic"
        if os.path.exists(experiment_prefix):
            os.system(f"rm -rf {experiment_prefix}")

        os.environ["MARIN_PREFIX"] = experiment_prefix
        deferred = create_steps(prefix=experiment_prefix, synth_data=synth_data)
        steps = deferred_to_steps(deferred)

        StepRunner().run(steps, dry_run=dry_run)

        logger.info(f"Execution completed successfully. All outputs are in {experiment_prefix}")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()

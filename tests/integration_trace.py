# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Self-contained integration test using a traced Deferred DAG with its own minimal runner.

Builds a DAG of Deferred nodes, topologically sorts them, and executes each
node sequentially with a ContextVar-based StepContext.
"""

import hashlib
import json
import logging
import os
import sys
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Generic, TypeVar

import click
from fray.v1.cluster import create_cluster, set_current_cluster
from fray.v2 import ResourceConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.gpt2 import Gpt2Config
from levanter.trainer import TrainerConfig
from marin.execution.artifact import Artifact
from marin.execution.step_model import StepSpec
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
    def __init__(
        self,
        fn: Callable[..., T] | None = None,
        id: str | None = None,  # noqa: A002
        value: Any = None,
        args: tuple = (),
        kwargs: dict | None = None,
        name: str | None = None,
    ):
        self.fn = fn
        self.value = value
        self.args = args
        self.kwargs = kwargs or {}
        self.name = name or (fn.__name__.lstrip("_") if fn else "value")
        self.id: str = id or self._make_id()

    def _make_id(self) -> str:
        """Deterministic ID from name + value + dep IDs."""
        content: dict[str, Any] = {"name": self.name}
        if self.value is not None:
            content["value"] = repr(self.value)
        dep_ids = []
        for arg in self.args:
            _collect_ids(arg, dep_ids)
        for v in self.kwargs.values():
            _collect_ids(v, dep_ids)
        content["deps"] = sorted(dep_ids)
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:12]

    @staticmethod
    def from_value(v: Any) -> "Deferred":
        return Deferred(value=v)

    @property
    def deps(self) -> "list[Deferred]":
        """Extract all Deferred dependencies from args and kwargs."""
        found: list[Deferred] = []
        for arg in self.args:
            _collect_deferred(arg, found)
        for v in self.kwargs.values():
            _collect_deferred(v, found)
        return found

    @cached_property
    def output_path(self) -> str:
        """Deterministic path: hash over name + sorted dep output paths + value."""
        content: dict[str, Any] = {"name": self.name}
        if self.value is not None:
            content["value"] = repr(self.value)
        content["deps"] = sorted(d.output_path for d in self.deps)
        hash_id = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:8]
        prefix = os.environ.get("MARIN_PREFIX", "/tmp/marin")
        return f"{prefix}/{self.name}_{hash_id}"


def _collect_deferred(obj: Any, out: list["Deferred"]):
    """Recursively collect Deferred instances from nested args (handles dicts)."""
    if isinstance(obj, Deferred):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_deferred(v, out)


def _collect_ids(obj: Any, out: list[str]):
    """Recursively collect Deferred IDs from nested args (handles dicts)."""
    if isinstance(obj, Deferred):
        out.append(obj.id)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_ids(v, out)


# ---------------------------------------------------------------------------
# StepContext — ContextVar-based, fully self-contained
# ---------------------------------------------------------------------------

_current_step_context: ContextVar["StepContext | None"] = ContextVar("step_context", default=None)


@dataclass
class StepContext:
    _output_path: str
    dep_id_to_path: dict[str, str]

    def output_path(self) -> str:
        return self._output_path

    @classmethod
    def current(cls) -> "StepContext":
        ctx = _current_step_context.get()
        if ctx is None:
            raise RuntimeError("No active StepContext")
        return ctx

    def resolve(self, dep: "Deferred[T]") -> Any:
        """Load the artifact at the dependency's output path."""
        path = self.dep_id_to_path[dep.id]
        return Artifact.load(path)


@contextmanager
def active_step_context(ctx: StepContext):
    token = _current_step_context.set(ctx)
    try:
        yield ctx
    finally:
        _current_step_context.reset(token)


# ---------------------------------------------------------------------------
# @deferred decorator
# ---------------------------------------------------------------------------


def deferred(fn: Callable):
    def wrapper(*args, **kwargs):
        return Deferred(fn=fn, args=args, kwargs=kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Deferred -> StepSpec conversion
# ---------------------------------------------------------------------------


def deferred_to_steps(root: Deferred) -> list[StepSpec]:
    """Trace the Deferred DAG into a topologically-sorted list of StepSpecs.

    DFS post-order gives leaves first so each dep's output_path is known
    when we build the StepSpec for its dependents. Each StepSpec.fn closes
    over the original Deferred, setting up a StepContext before invoking it.
    """
    all_nodes: dict[str, Deferred] = {}
    pending: list[Deferred] = [root]
    while pending:
        curr = pending.pop()
        if curr.id in all_nodes:
            continue
        all_nodes[curr.id] = curr
        pending.extend(curr.deps)

    topo_order: list[str] = []
    visited: set[str] = set()

    def dfs(node_id: str):
        if node_id in visited:
            return
        visited.add(node_id)
        for dep in all_nodes[node_id].deps:
            dfs(dep.id)
        topo_order.append(node_id)

    dfs(root.id)

    deferred_id_to_output_path: dict[str, str] = {}
    steps: list[StepSpec] = []
    for deferred_id in topo_order:
        d = all_nodes[deferred_id]
        dep_output_paths = [deferred_id_to_output_path[dep.id] for dep in d.deps]

        step_fn = _make_step_fn(d) if d.fn is not None else _make_value_fn(d)

        spec = StepSpec(
            name=d.name,
            deps=dep_output_paths,
            fn=step_fn,
        )
        deferred_id_to_output_path[d.id] = spec.output_path
        steps.append(spec)

    return steps


def _make_step_fn(d: Deferred) -> Callable[[str], Any]:
    """Build a StepSpec.fn that sets up StepContext and invokes the Deferred."""

    def step_fn(output_path: str) -> Any:
        dep_map = {dep.id: dep.output_path for dep in d.deps}
        ctx = StepContext(_output_path=output_path, dep_id_to_path=dep_map)
        with active_step_context(ctx):
            return d.fn(*d.args, **d.kwargs)

    return step_fn


def _make_value_fn(d: Deferred) -> Callable[[str], Any]:
    """Build a StepSpec.fn for a constant-value Deferred."""

    def value_fn(output_path: str) -> Any:
        return d.value

    return value_fn


# ---------------------------------------------------------------------------
# Minimal sequential runner (replaces StepRunner)
# ---------------------------------------------------------------------------


def run_steps(steps: list[StepSpec], *, dry_run: bool = False):
    """Execute a list of StepSpecs sequentially, saving results as Artifacts."""
    for step in steps:
        if dry_run:
            logger.info(f"[DRY RUN] Would run {step.name} -> {step.output_path}")
            continue

        result = step.fn(step.output_path)
        if result is not None:
            Artifact.save(result, step.output_path)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


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
            trainer=TrainerConfig(train_batch_size=8, num_train_steps=2, max_eval_batches=1, require_accelerator=False),
        ),
    )


def create_steps(*, prefix: str, synth_data: str) -> Deferred[str]:
    # ############################################################
    # Transform HTML to text

    hq_step = _html_to_md(input_path=Deferred.from_value(os.path.join(synth_data, "pos")))

    lq_step = _html_to_md(input_path=Deferred.from_value(os.path.join(synth_data, "neg")))

    _quality = _train_quality({"hq": hq_step, "lq": lq_step})
    dedup_exact_step = _dedup_exact(hq_step)
    dedup_fuzzy_step = _dedup_fuzzy(hq_step)
    consolidate_step = _consolidate({"hq": hq_step, "dedup_exact": dedup_exact_step, "dedup_fuzzy": dedup_fuzzy_step})
    tokenize_step = _tokenize({"consolidate": consolidate_step})
    train_step = _train_lm({"tokenize": tokenize_step})
    return train_step


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
        root = create_steps(prefix=experiment_prefix, synth_data=synth_data)
        steps = deferred_to_steps(root)
        run_steps(steps, dry_run=dry_run)

        logger.info(f"Execution completed successfully. All outputs are in {experiment_prefix}")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()

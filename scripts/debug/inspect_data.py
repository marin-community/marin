#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect training data at a given step.

Given an experiment file and a step number, dump the decoded training examples
for that step's batch to JSONL. Requires a Ray cluster (--cluster).

Usage:
  uv run scripts/debug/inspect_data.py experiments/references/canary_train.py \
      --step 100 --cluster us-central2
  uv run scripts/debug/inspect_data.py experiments/references/canary_train.py \
      --step 100 --cluster us-central2 --output step_100.jsonl
  uv run scripts/debug/inspect_data.py experiments/references/canary_train.py \
      --step 100 --cluster us-central2 --var training_step
  uv run scripts/debug/inspect_data.py experiments/references/canary_train.py \
      --step 100 --cluster us-central2 --summary
  uv run scripts/debug/inspect_data.py experiments/references/canary_train.py \
      --steps 0,100,500 --cluster us-central2 --summary
  uv run scripts/debug/inspect_data.py experiments/references/canary_train.py \
      --step 100 --cluster us-central2 --truncate 200
  uv run scripts/debug/inspect_data.py experiments/references/canary_train.py \
      --step 100 --cluster us-central2 --tui
"""

import importlib.util
import json
import os
import shlex
import sys
from collections import Counter

import click
import haliax as hax
import jax.random as jrandom

from levanter.data.mixture import MixtureDataset, rescale_mixture_schedule_for_batch_schedule
from levanter.utils.thread_utils import blocking_wait
from marin.execution.executor import Executor, ExecutorStep
from marin.training.training import TrainLmOnPodConfig


def _load_module(path: str):
    spec = importlib.util.spec_from_file_location("experiment", path)
    if spec is None or spec.loader is None:
        raise click.ClickException(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _find_training_step(mod, var: str | None) -> ExecutorStep:
    """Find an ExecutorStep whose config is a TrainLmOnPodConfig."""
    candidates = {}
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, ExecutorStep) and isinstance(obj.config, TrainLmOnPodConfig):
            candidates[name] = obj

    if var is not None:
        if var not in candidates:
            available = ", ".join(candidates) or "(none)"
            raise click.ClickException(f"Variable '{var}' is not a training ExecutorStep. Available: {available}")
        return candidates[var]

    if len(candidates) == 0:
        raise click.ClickException("No ExecutorStep with TrainLmOnPodConfig found in module.")
    if len(candidates) > 1:
        names = ", ".join(candidates)
        raise click.ClickException(f"Multiple training steps found: {names}. Use --var to specify one.")
    return next(iter(candidates.values()))


def _submit_to_cluster(
    cluster: str,
    experiment: str,
    step: int | None,
    var: str | None,
    output: str | None,
    prefix: str | None = None,
    steps: str | None = None,
    truncate: int | None = None,
    summary: bool = False,
):
    import asyncio

    from ray.job_submission import JobSubmissionClient

    from fray.v1.cluster.ray import DashboardConfig, ray_dashboard

    config = _resolve_cluster_config(cluster)

    parts = ["python", "scripts/debug/inspect_data.py", experiment]
    if steps:
        parts.extend(["--steps", steps])
    else:
        parts.extend(["--step", str(step)])
    if var:
        parts.extend(["--var", var])
    if output:
        parts.extend(["--output", output])
    if prefix:
        parts.extend(["--prefix", prefix])
    if truncate is not None:
        parts.extend(["--truncate", str(truncate)])
    if summary:
        parts.append("--summary")
    entrypoint = shlex.join(parts)

    runtime_env = _cluster_runtime_env()

    async def _run():
        client = JobSubmissionClient("http://127.0.0.1:8265")
        submission_id = client.submit_job(entrypoint=entrypoint, runtime_env=runtime_env)
        click.echo(f"Submitted Ray job: {submission_id}", err=True)
        async for lines in client.tail_job_logs(submission_id):
            print(lines, end="")

    with ray_dashboard(DashboardConfig.from_cluster(config)):
        asyncio.run(_run())


def _resolve_config(executor_step: ExecutorStep, prefix: str) -> TrainLmOnPodConfig:
    """Resolve InputName references in the config using the Executor."""
    executor = Executor(prefix=prefix, executor_info_base_path=os.path.join(prefix, "experiments"))
    executor.compute_version(executor_step, is_pseudo_dep=False)
    return executor.configs[executor_step]


def _build_dataset(train_config):
    """Build the MixtureDataset from a resolved train config."""
    seed = train_config.trainer.seed
    # Mirrors key splitting in train_lm.py: data_key, loader_key, model_key, training_key
    data_key, _, _, _ = jrandom.split(jrandom.PRNGKey(seed), 4)
    if train_config.data_seed is not None:
        data_key = jrandom.PRNGKey(train_config.data_seed)

    train_length = train_config.train_seq_len
    if train_length is None:
        train_length = train_config.model.max_seq_len
    Pos = hax.Axis("position", train_length)

    mix_key, shuffle_key = jrandom.split(data_key)
    batch_schedule = train_config.trainer.batch_schedule
    initial_batch_size = batch_schedule.batch_size_at_step(0)

    datasets = train_config.data.train_sets(Pos, key=shuffle_key, initial_batch_size=initial_batch_size)

    weights = train_config.data.train_weights
    if isinstance(weights, list):
        weights = rescale_mixture_schedule_for_batch_schedule(weights, batch_schedule)

    dataset = MixtureDataset(
        datasets=datasets,
        weights=weights,
        stop_strategy=train_config.data.stop_strategy,
        key=mix_key,
        block_size=train_config.data.mixture_block_size,
    )
    return dataset, batch_schedule


def _get_source_names(dataset: MixtureDataset, indices: list[int]) -> list[str]:
    """Map global indices to their source dataset names."""
    names = []
    for idx in indices:
        block_id = idx // dataset.block_size
        index_within_block = idx % dataset.block_size
        block = dataset._get_block(block_id)
        packed_id = block[index_within_block]
        dataset_id = packed_id >> 16
        names.append(dataset.dataset_index[dataset_id])
    return names


def _get_step_weights(dataset: MixtureDataset, batch_schedule, step: int) -> dict[str, float]:
    """Get the mixture weights active at a given step."""
    offset = batch_schedule.global_data_offset_by_step(step)
    block_id = offset // dataset.block_size
    stage_idx = dataset._get_stage_for_block(block_id)
    return dataset.weight_stages[stage_idx][1]


def _fetch_step_examples(dataset, batch_schedule, tokenizer, step: int) -> list[dict]:
    """Fetch and decode all examples for a given step, with metadata."""
    offset = batch_schedule.global_data_offset_by_step(step)
    bs = batch_schedule.batch_size_at_step(step)
    indices = list(range(offset, offset + bs))
    examples = blocking_wait(dataset.get_batch(indices))
    sources = _get_source_names(dataset, indices)
    results = []
    for i, (ex, src) in enumerate(zip(examples, sources, strict=True)):
        tokens = ex.tokens.tolist()
        lw = ex.loss_weight
        pct_masked = float((lw == 0).sum()) / len(lw) * 100
        results.append(
            {
                "index": i,
                "text": tokenizer.decode(tokens),
                "source": src,
                "num_tokens": len(tokens),
                "pct_masked": round(pct_masked, 1),
            }
        )
    return results


def _build_summary(dataset, batch_schedule, examples: list[dict], step: int) -> dict:
    """Build a summary dict for a step's batch."""
    weights = _get_step_weights(dataset, batch_schedule, step)
    source_counts = dict(Counter(ex["source"] for ex in examples))
    token_counts = [ex["num_tokens"] for ex in examples]
    mask_pcts = [ex["pct_masked"] for ex in examples]
    return {
        "step": step,
        "batch_size": len(examples),
        "weights": weights,
        "source_counts": source_counts,
        "token_stats": {
            "min": min(token_counts),
            "max": max(token_counts),
            "mean": round(sum(token_counts) / len(token_counts), 1),
        },
        "mask_stats": {
            "min_pct": min(mask_pcts),
            "max_pct": max(mask_pcts),
            "mean_pct": round(sum(mask_pcts) / len(mask_pcts), 1),
        },
    }


def _parse_steps(steps_str: str) -> list[int]:
    """Parse a steps string like '0,100,500' or '0:1000:100'."""
    if ":" in steps_str:
        parts = steps_str.split(":")
        if len(parts) == 2:
            return list(range(int(parts[0]), int(parts[1])))
        elif len(parts) == 3:
            return list(range(int(parts[0]), int(parts[1]), int(parts[2])))
        else:
            raise click.ClickException(f"Invalid step range: {steps_str}")
    return [int(s.strip()) for s in steps_str.split(",")]


def _run_tui(fetch_fn, start_step: int):
    """Interactive TUI: left/right for docs, up/down for steps, g for goto.

    Args:
        fetch_fn: callable(step: int) -> list[dict], returns example dicts for a step.
    """
    import curses
    import textwrap

    step = start_step
    doc_idx = 0
    examples: list[dict] = []
    scroll_offset = 0
    status_msg = ""
    show_stats = False
    goto_mode = False
    goto_buf = ""

    def load_step(scr):
        nonlocal examples, doc_idx, scroll_offset, status_msg
        h, w = scr.getmaxyx()
        scr.clear()
        scr.addnstr(h // 2, 0, f"Loading step {step}...", w - 1)
        scr.refresh()
        try:
            examples = fetch_fn(step)
            doc_idx = min(doc_idx, max(0, len(examples) - 1))
            scroll_offset = 0
            status_msg = ""
        except Exception as e:
            examples = []
            status_msg = str(e)

    def _source_distribution():
        counts = Counter(ex["source"] for ex in examples)
        total = len(examples)
        parts = []
        for name, count in sorted(counts.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            parts.append(f"{name}: {count} ({pct:.1f}%)")
        return " | ".join(parts)

    def draw(scr):
        scr.clear()
        h, w = scr.getmaxyx()
        if h < 4 or w < 20:
            return

        # Header with metadata
        if examples and 0 <= doc_idx < len(examples):
            ex = examples[doc_idx]
            header = (
                f" Step {step} | Doc {doc_idx + 1}/{len(examples)}"
                f" [{ex['source']}] {ex['num_tokens']}tok {ex['pct_masked']}%masked"
                f" | q:quit </>:docs ^/v:steps g:goto s:stats"
            )
        else:
            header = f" Step {step} | Doc {doc_idx + 1}/{len(examples)} | q:quit </>:docs ^/v:steps g:goto"
        scr.attron(curses.A_REVERSE)
        scr.addnstr(0, 0, header.ljust(w), w - 1)
        scr.attroff(curses.A_REVERSE)

        if status_msg:
            scr.addnstr(2, 0, f"Error: {status_msg}", w - 1)
            scr.refresh()
            return

        if not examples:
            scr.addnstr(2, 0, "No examples.", w - 1)
            scr.refresh()
            return

        # Determine footer area height
        footer_lines = 1
        stats_line = ""
        if show_stats:
            stats_line = _source_distribution()
            footer_lines = 2

        # Wrap and display text with scrolling
        text = examples[doc_idx]["text"]
        wrapped = []
        for line in text.split("\n"):
            if line == "":
                wrapped.append("")
            else:
                wrapped.extend(textwrap.wrap(line, w - 1) or [""])

        body_h = h - 1 - footer_lines
        visible = wrapped[scroll_offset : scroll_offset + body_h]
        for row, line in enumerate(visible):
            scr.addnstr(row + 1, 0, line, w - 1)

        # Stats bar
        if stats_line:
            scr.attron(curses.A_REVERSE)
            scr.addnstr(h - 2, 0, (" " + stats_line).ljust(w), w - 1)
            scr.attroff(curses.A_REVERSE)

        # Footer / goto prompt
        if goto_mode:
            footer = f" Go to step: {goto_buf}_ (Enter to confirm, Esc to cancel)"
        else:
            total_lines = len(wrapped)
            if total_lines > body_h:
                pct = min(100, int((scroll_offset + body_h) / total_lines * 100))
                footer = f" {pct}%  PgUp/PgDn to scroll "
            else:
                footer = ""
        scr.attron(curses.A_REVERSE)
        scr.addnstr(h - 1, 0, footer.ljust(w), w - 1)
        scr.attroff(curses.A_REVERSE)

        scr.refresh()

    def main_loop(scr):
        nonlocal step, doc_idx, scroll_offset, show_stats, goto_mode, goto_buf
        curses.curs_set(0)
        scr.timeout(-1)
        load_step(scr)
        draw(scr)

        while True:
            key = scr.getch()

            if goto_mode:
                if key == 27:  # Escape
                    goto_mode = False
                    goto_buf = ""
                elif key in (curses.KEY_ENTER, 10, 13):
                    if goto_buf:
                        try:
                            step = int(goto_buf)
                        except ValueError:
                            pass
                        load_step(scr)
                    goto_mode = False
                    goto_buf = ""
                elif key == curses.KEY_BACKSPACE or key == 127:
                    goto_buf = goto_buf[:-1]
                elif 48 <= key <= 57:  # digits 0-9
                    goto_buf += chr(key)
                draw(scr)
                continue

            if key == ord("q") or key == ord("Q"):
                break
            elif key == ord("g") or key == ord("G"):
                goto_mode = True
                goto_buf = ""
            elif key == ord("s") or key == ord("S"):
                show_stats = not show_stats
            elif key == curses.KEY_RIGHT:
                if examples and doc_idx < len(examples) - 1:
                    doc_idx += 1
                    scroll_offset = 0
            elif key == curses.KEY_LEFT:
                if doc_idx > 0:
                    doc_idx -= 1
                    scroll_offset = 0
            elif key == curses.KEY_DOWN:
                step += 1
                load_step(scr)
            elif key == curses.KEY_UP:
                if step > 0:
                    step -= 1
                    load_step(scr)
            elif key == curses.KEY_NPAGE:  # Page Down
                h, w = scr.getmaxyx()
                body_h = h - 1 - (2 if show_stats else 1)
                if examples:
                    text = examples[doc_idx]["text"]
                    total = 0
                    for ln in text.split("\n"):
                        total += len(textwrap.wrap(ln, w - 1)) if ln else 1
                    scroll_offset = min(scroll_offset + body_h, max(0, total - body_h))
            elif key == curses.KEY_PPAGE:  # Page Up
                h = scr.getmaxyx()[0]
                body_h = h - 1 - (2 if show_stats else 1)
                scroll_offset = max(0, scroll_offset - body_h)
            elif key == curses.KEY_RESIZE:
                pass
            else:
                continue
            draw(scr)

    curses.wrapper(main_loop)


def _resolve_cluster_config(cluster: str) -> str:
    from marin.cluster.config import find_config_by_region

    if cluster.endswith(".yaml") or os.path.exists(cluster):
        return cluster
    return find_config_by_region(cluster)


def _cluster_runtime_env() -> dict:
    from fray.v1.cluster.ray.deps import build_python_path

    return {
        "working_dir": os.getcwd(),
        "excludes": [".git", "docs/", "**/*.pack", "lib/levanter/docs"],
        "env_vars": {"PYTHONPATH": ":".join(build_python_path())},
    }


def _run_tui_on_cluster(cluster: str, experiment: str, step: int, var: str | None, prefix: str | None):
    """Run the TUI locally with data fetched from a Ray cluster actor."""
    import ray

    from fray.v1.cluster.ray import DashboardConfig, ray_dashboard

    config = _resolve_cluster_config(cluster)
    runtime_env = _cluster_runtime_env()

    # ray_init=False: we call ray.init() ourselves so we can pass our runtime_env
    with ray_dashboard(DashboardConfig.from_cluster(config)) as conn:
        cluster_name = next(iter(conn.port_mappings))
        api_port = conn.port_mappings[cluster_name].api_port
        ray.init(address=f"ray://localhost:{api_port}", runtime_env=runtime_env)

        @ray.remote
        class DataInspector:
            def __init__(self, experiment_path, prefix, var):
                # On the cluster, MARIN_PREFIX is set in the node environment.
                # Our runtime_env env_vars don't include it, so fall back to os.environ.
                prefix = prefix or os.environ.get("MARIN_PREFIX")
                mod = _load_module(experiment_path)
                executor_step = _find_training_step(mod, var)
                resolved = _resolve_config(executor_step, prefix)
                train_config = resolved.train_config
                self.tokenizer = train_config.data.the_tokenizer
                self.dataset, self.batch_schedule = _build_dataset(train_config)

            def fetch_step(self, step):
                return _fetch_step_examples(self.dataset, self.batch_schedule, self.tokenizer, step)

        click.echo("Creating data inspector on cluster...", err=True)
        inspector = DataInspector.remote(experiment, prefix, var)
        # Warm up: ensure actor is ready before entering curses
        ray.get(inspector.fetch_step.remote(step))
        click.echo("Ready.", err=True)

        def fetch_fn(s):
            return ray.get(inspector.fetch_step.remote(s))

        _run_tui(fetch_fn, step)


@click.command()
@click.argument("experiment", type=click.Path(exists=True))
@click.option("--step", default=None, type=int, help="Training step to inspect.")
@click.option("--steps", default=None, type=str, help="Comma-separated steps or start:end:stride range.")
@click.option("--var", default=None, help="Name of the ExecutorStep variable (auto-detected if unambiguous).")
@click.option("--output", "-o", default=None, type=click.Path(), help="Output JSONL path (default: stdout).")
@click.option("--cluster", default=None, help="Ray cluster name or config path. Submits as a Ray job when set.")
@click.option("--prefix", default=None, help="Marin prefix (default: $MARIN_PREFIX).")
@click.option("--tui", is_flag=True, default=False, help="Interactive viewer.")
@click.option("--truncate", default=None, type=int, help="Max characters per text in output.")
@click.option("--summary", is_flag=True, default=False, help="Print per-step stats instead of full text.")
def main(
    experiment: str,
    step: int | None,
    steps: str | None,
    var: str | None,
    output: str | None,
    cluster: str | None,
    prefix: str | None,
    tui: bool,
    truncate: int | None,
    summary: bool,
):
    """Dump decoded training examples for a given step's batch to JSONL."""
    if step is None and steps is None:
        if tui:
            step = 0
        else:
            raise click.ClickException("Must specify --step or --steps.")
    if step is not None and steps is not None:
        raise click.ClickException("Cannot use both --step and --steps.")
    if tui and steps is not None:
        raise click.ClickException("--steps cannot be used with --tui. Use --step instead.")

    step_list = _parse_steps(steps) if steps else [step]

    # When submitted as a Ray job, the script runs on the cluster without --cluster.
    # Detect this via RAY_JOB_ID which Ray sets automatically for submitted jobs.
    on_cluster_node = os.environ.get("RAY_JOB_ID") is not None

    if not cluster and not on_cluster_node:
        raise click.ClickException("Must specify --cluster.")

    if cluster and tui:
        _run_tui_on_cluster(cluster, experiment, step_list[0], var, prefix)
        return

    if cluster:
        _submit_to_cluster(
            cluster,
            experiment,
            step,
            var,
            output,
            prefix,
            steps=steps,
            truncate=truncate,
            summary=summary,
        )
        return

    # Running on a cluster node as a submitted Ray job.
    prefix = prefix or os.environ.get("MARIN_PREFIX")
    if not prefix:
        raise click.ClickException("MARIN_PREFIX not set on cluster node.")

    mod = _load_module(experiment)
    executor_step = _find_training_step(mod, var)
    resolved_config = _resolve_config(executor_step, prefix)
    train_config = resolved_config.train_config
    tokenizer = train_config.data.the_tokenizer

    dataset, batch_schedule = _build_dataset(train_config)

    out = open(output, "w") if output else sys.stdout
    total_examples = 0
    try:
        for s in step_list:
            examples = _fetch_step_examples(dataset, batch_schedule, tokenizer, s)
            if summary:
                json.dump(_build_summary(dataset, batch_schedule, examples, s), out)
                out.write("\n")
            else:
                for ex in examples:
                    record = dict(ex, step=s)
                    if truncate is not None and len(record["text"]) > truncate:
                        record["text"] = record["text"][:truncate]
                        record["truncated"] = True
                    json.dump(record, out)
                    out.write("\n")
            total_examples += len(examples)
    finally:
        if output:
            out.close()

    if output:
        click.echo(f"Wrote {total_examples} examples to {output}")


if __name__ == "__main__":
    main()

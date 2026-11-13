# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rerun the milestone-16 identity gate's five-pass scoring on a batch-invariant vLLM serve.

Retest for https://github.com/marin-community/marin/issues/10699. The gate's scorer and gate
module are loaded unchanged from their pinned commits (``git show``), with their recorded
hashes checked. Each arm is served with the gate's original arguments plus
``VLLM_BATCH_INVARIANT=1``, in the interactive band, and scored with the gate's ``score_arm``:
the 64-case grid, the three stability rows, then the 24 primary rows four more times.

Differences from the original run, all recorded in the receipt:

- ``VLLM_BATCH_INVARIANT=1`` is set on the serving job, and the band is interactive;
- the H100 peer is chosen with ``--target-cluster``; an interactive serve on a peer that also runs
  production work can be preempted mid-scoring, which loses the arm;
- ``controls.jsonl`` is not archived with the experiment, so an empty file stands in. ``check``
  shows the reconstructed grid equals the original receipt's 64 cases in order;
- model provenance is copied from the original receipt, not re-hashed.

Run from the repository root::

    uv run python experiments/post_training/identity_gate_batch_invariant.py check ...
    uv run python experiments/post_training/identity_gate_batch_invariant.py repeat ...
    uv run python experiments/post_training/identity_gate_batch_invariant.py run --arm baseline ...

``repeat`` is the smaller check: the baseline arm only, the 24 primary rows scored twice and
compared exactly. ``run`` and ``decide`` are the full two-arm, five-pass gate.
    uv run python experiments/post_training/identity_gate_batch_invariant.py decide ...
"""

import contextlib
import hashlib
import importlib
import json
import logging
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import click
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.types import JobName
from marin.inference.iris_cli import _wait_for_endpoint
from rigging.timing import Duration
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

SCORER_REVISION = "2a1de41d86b5d6469a49ed35da505abcb202f189"
GATE_REVISION = "9e3857f86b168605b063e4b8f5aedb34ad10f01c"
GATE_PACKAGE = "experiments/interp/snowball_identity"
BATCH_INVARIANT_ENV = "VLLM_BATCH_INVARIANT=1"
# The original gate's SERVE_ARGS without its production band, plus batch invariance.
SERVE_ARGS = [
    "--cluster", "marin", "--backend", "vllm", "--gpu", "H100x8", "--target-cluster", "cw-rno2a",
    "--vllm-source", "marin-fork", "--dtype", "bfloat16", "--tensor-parallel-size", "1",
    "--vllm-arg=--data-parallel-size=8", "--vllm-arg=--enable-expert-parallel",
    "--vllm-arg=--reasoning-parser=grug", "--vllm-arg=--max-num-seqs=256",
    "--max-model-len", "32768", "--max-num-batched-tokens", "2048",
    "--cpu", "32", "--memory", "512g", "--disk", "512g", "--timeout-hours", "1",
    "--max-retries-preemption", "0", "--no-cache", "--no-wait", "--env", BATCH_INVARIANT_ENV,
]  # fmt: skip
ENDPOINT_WAIT_SECONDS = 2700
IRIS = [sys.executable, "-c", "import sys; from iris.cli import main; sys.argv[0] = 'iris'; main()", "--cluster=marin"]
GRID_FIELDS = ("order", "row_id", "category", "split", "condition", "prefix", "candidates", "correct_indices")


def _git_show(revision: str, path: str) -> bytes:
    return subprocess.run(["git", "show", f"{revision}:{path}"], check=True, capture_output=True).stdout


def load_gate_code(workdir: Path) -> tuple[ModuleType, ModuleType]:
    """Materialize the pinned scorer and gate module as an importable package and import them."""
    package = workdir / GATE_PACKAGE
    package.mkdir(parents=True)
    for directory in (workdir / "experiments", workdir / "experiments/interp", package):
        (directory / "__init__.py").write_bytes(b"")
    (package / "identity_eval.py").write_bytes(_git_show(SCORER_REVISION, f"{GATE_PACKAGE}/identity_eval.py"))
    (package / "m16_gate.py").write_bytes(_git_show(GATE_REVISION, f"{GATE_PACKAGE}/m16_gate.py"))
    sys.path.insert(0, str(workdir))
    shared = importlib.import_module("experiments.interp.snowball_identity.identity_eval")
    gate_module = importlib.import_module("experiments.interp.snowball_identity.m16_gate")
    assert Path(shared.__file__).is_relative_to(workdir), shared.__file__
    return shared, gate_module


def prepare_prompt_dir(workdir: Path, identity_path: Path) -> Path:
    prompt_dir = workdir / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "identity.jsonl").write_bytes(identity_path.read_bytes())
    (prompt_dir / "controls.jsonl").write_bytes(b"")
    return prompt_dir


def check_inputs(shared: ModuleType, gate_module: ModuleType, prompt_dir: Path, tokenizer) -> None:
    """The gate's frozen-input checks, except the unavailable controls file."""
    if gate_module.sha256_file(Path(shared.__file__)) != gate_module.IDENTITY_EVAL_SHA256:
        raise click.ClickException("identity_eval.py differs from the gate's pinned scorer")
    if gate_module.sha256_file(prompt_dir / "identity.jsonl") != gate_module.PROMPT_SHA256["identity.jsonl"]:
        raise click.ClickException("identity.jsonl differs from the gate's frozen prompts")
    if hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != gate_module.TEMPLATE_SHA256:
        raise click.ClickException("Tokenizer chat template differs from the gate's template")


def check_grid_matches(gate_module: ModuleType, prompt_dir: Path, gate: dict, original: dict, tokenizer) -> int:
    """Require the reconstructed grid and candidate spans to equal the original receipt's cases."""
    jobs = gate_module.grid_jobs(prompt_dir, gate)
    cases = original["contrastive_grid"]["cases_detail"]
    assert len(jobs) == len(cases), (len(jobs), len(cases))
    for order, ((row, condition), case) in enumerate(zip(jobs, cases, strict=True)):
        spec = row.contrastive
        rebuilt = (order, row.id, row.category, row.split, condition.value, spec["prefix"], spec["candidates"])
        assert rebuilt == tuple(case[field] for field in GRID_FIELDS[:-1]), (order, row.id)
        assert spec["correct_indices"] == case["correct_indices"], row.id
        spans = gate_module.span_lengths(tokenizer, spec["prefix"], spec["candidates"])
        assert spans == case["candidate_token_span"], (row.id, spans)
    return len(jobs)


def serve_args_for(target_cluster: str) -> list[str]:
    """The serve arguments with the original gate's cw-rno2a placement replaced."""
    args = list(SERVE_ARGS)
    args[args.index("--target-cluster") + 1] = target_cluster
    return args


def submit_serve(model: str, serve_args: list[str], name: str, endpoint_name: str) -> str:
    """Submit the serve and return its job id as the CLI printed it."""
    argv = [sys.executable, "-m", "marin.inference.iris_cli", model, *serve_args]
    argv += ["--name", name, "--endpoint-name", endpoint_name]
    output = subprocess.run(argv, check=True, capture_output=True, text=True).stdout
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] == "job":
            return fields[1]
    raise click.ClickException("The serve CLI did not report a job id")


@contextlib.contextmanager
def served_endpoint(
    shared: ModuleType,
    model: str,
    serve_args: list[str],
    tag: str,
    wait_seconds: float = ENDPOINT_WAIT_SECONDS,
) -> Iterator[tuple[object, dict]]:
    """Serve ``model``, yield the gate's ``Endpoint`` and the serve record, and cancel the serve on exit.

    ``wait_seconds`` covers queueing for capacity as well as model load.
    """
    name = f"m16gate-bi-{tag}-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
    endpoint_name = f"/serve/{name}"
    job_id = submit_serve(model, serve_args, name, endpoint_name)
    logger.info("Submitted %s", job_id)
    record = {"job": job_id, "submitted_utc": time.time()}
    try:
        with connect_controller(cluster_name="marin") as info:
            with IrisClient.remote(info.url, workspace=Path.cwd(), credentials=info.credentials) as client:
                job = client.job(JobName.from_wire(job_id))
                _wait_for_endpoint(client, job, endpoint_name, wait_seconds)
                token = client.mint_endpoint_token(endpoint_name, ttl=Duration.from_hours(1))
        record["ready_utc"] = time.time()
        logger.info("Endpoint ready; scoring")
        # The capability URL stays in memory and is never logged.
        yield shared.Endpoint(token.capability_url.rstrip("/")), record
    finally:
        cancel_rc = subprocess.run([*IRIS, "job", "cancel", job_id], capture_output=True).returncode
        logger.info("Serve cancel returned %d", cancel_rc)
        record["cancelled_after_scoring"] = cancel_rc == 0


@click.group()
def cli():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def common(fn):
    fn = click.option("--gate", "gate_path", required=True, type=click.Path(path_type=Path, exists=True))(fn)
    fn = click.option("--identity", "identity_path", required=True, type=click.Path(path_type=Path, exists=True))(fn)
    fn = click.option("--tokenizer", "tokenizer_path", required=True, type=click.Path(path_type=Path, exists=True))(fn)
    return click.option("--original", required=True, type=click.Path(path_type=Path, exists=True))(fn)


@cli.command()
@common
def check(gate_path: Path, identity_path: Path, tokenizer_path: Path, original: Path):
    """No GPU: verify the pinned code, the inputs, and the reconstructed grid."""
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        shared, gate_module = load_gate_code(workdir)
        gate = gate_module.load_gate(gate_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        prompt_dir = prepare_prompt_dir(workdir, identity_path)
        check_inputs(shared, gate_module, prompt_dir, tokenizer)
        cases = check_grid_matches(gate_module, prompt_dir, gate, json.loads(original.read_text()), tokenizer)
    print(json.dumps({"status": "ok", "grid_cases_matching_original": cases}))


@cli.command()
@common
@click.option("--arm", type=click.Choice(["baseline", "candidate"]), required=True)
@click.option("--target-cluster", required=True, help="H100 peer to serve on; the original gate used cw-rno2a.")
@click.option("--out", required=True, type=click.Path(path_type=Path))
def run(
    gate_path: Path,
    identity_path: Path,
    tokenizer_path: Path,
    original: Path,
    arm: str,
    target_cluster: str,
    out: Path,
):
    """Serve one arm with batch invariance, score it with the gate's score_arm, release the serve."""
    serve_args = serve_args_for(target_cluster)
    original_receipt = json.loads(original.read_text())
    assert original_receipt["arm"] == arm, original_receipt["arm"]
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        shared, gate_module = load_gate_code(workdir)
        gate = gate_module.load_gate(gate_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        prompt_dir = prepare_prompt_dir(workdir, identity_path)
        check_inputs(shared, gate_module, prompt_dir, tokenizer)
        check_grid_matches(gate_module, prompt_dir, gate, original_receipt, tokenizer)

        provenance = {
            key: value
            for key, value in original_receipt["provenance"].items()
            if key not in ("serve", "serve_args", "engine_log_markers")
        }
        provenance["serve_args"] = serve_args
        provenance["rerun"] = {
            "target_cluster": target_cluster,
            "issue": "https://github.com/marin-community/marin/issues/10699",
            "model_provenance": "copied from the original receipt; model files were not re-hashed",
            "controls_jsonl": "unavailable; empty stand-in, grid verified against the original receipt",
            "serve_environment": [BATCH_INVARIANT_ENV],
            "priority_band": "interactive",
            "gate_module_revision": GATE_REVISION,
        }
        if arm == "candidate":
            provenance["candidate_kind"] = "sft"

        with served_endpoint(shared, provenance["model"], serve_args, arm) as (endpoint, serve):
            provenance["serve"] = serve
            receipt = gate_module.score_arm(endpoint, tokenizer, prompt_dir, gate, arm, provenance)
        gate_module.write_json(out, receipt)
    print(
        json.dumps(
            {
                "arm": arm,
                "status": "scored",
                "stability_max_abs_diff": receipt["logprob_stability_max_abs_diff"]["max"],
                "family_mean_spread": receipt["primary_passes"]["family_mean_spread_NLL_per_token"],
            }
        )
    )


@cli.command()
@common
@click.option("--target-cluster", required=True, help="H100 peer to serve on; the original gate used cw-rno2a.")
@click.option("--passes", type=click.IntRange(min=2), default=2, help="Passes over the 24 primary rows.")
@click.option(
    "--endpoint-wait-hours",
    type=float,
    default=ENDPOINT_WAIT_SECONDS / 3600,
    show_default=True,
    help="How long to wait for the serve, including time queued for capacity.",
)
@click.option("--out", required=True, type=click.Path(path_type=Path))
def repeat(
    gate_path: Path,
    identity_path: Path,
    tokenizer_path: Path,
    original: Path,
    target_cluster: str,
    passes: int,
    endpoint_wait_hours: float,
    out: Path,
):
    """Smaller check: serve the baseline arm and score the 24 primary rows ``passes`` times.

    Exactly equal passes mean the gate's five-pass spread is zero on this serve. Results are
    written after every pass, so a preempted serve keeps what it finished.
    """
    serve_args = serve_args_for(target_cluster)
    original_receipt = json.loads(original.read_text())
    assert original_receipt["arm"] == "baseline", original_receipt["arm"]
    model = original_receipt["provenance"]["model"]
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        shared, gate_module = load_gate_code(workdir)
        gate = gate_module.load_gate(gate_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        prompt_dir = prepare_prompt_dir(workdir, identity_path)
        check_inputs(shared, gate_module, prompt_dir, tokenizer)
        check_grid_matches(gate_module, prompt_dir, gate, original_receipt, tokenizer)
        rows = {row.id: row for row, _ in gate_module.grid_jobs(prompt_dir, gate)}
        primary_rows = gate["primary_rows"]
        report = {
            "issue": "https://github.com/marin-community/marin/issues/10699",
            "model": model,
            "serve_args": serve_args,
            "serve_environment": [BATCH_INVARIANT_ENV],
            "scorer_revision": SCORER_REVISION,
            "gate_module_revision": GATE_REVISION,
            "sampling": gate_module.SAMPLING,
            "request_order": "the 24 primary rows (gate order, no_system), one request per candidate, per pass",
            "original_stability_max_abs_diff": original_receipt["logprob_stability_max_abs_diff"],
            "original_family_mean_spread": original_receipt["primary_passes"]["family_mean_spread_NLL_per_token"],
            "passes": [],
        }
        wait_seconds = endpoint_wait_hours * 3600
        with served_endpoint(shared, model, serve_args, "repeat", wait_seconds) as (endpoint, serve):
            report["serve"] = serve
            for index in range(passes):
                logprobs, records = {}, {}
                for row_id, spec in primary_rows.items():
                    row = rows[row_id]
                    result = shared.score_contrastive_row(endpoint, tokenizer, row, shared.Condition.NO_SYSTEM)
                    spans = gate_module.span_lengths(tokenizer, result["prefix"], result["candidates"])
                    case = gate_module.case_record(-1, result, spans, row.contrastive["correct_indices"])
                    logprobs[row_id] = result["logprobs"]
                    records[row_id] = gate_module.primary_record(case, spec)
                report["passes"].append(
                    {
                        "pass": index,
                        "candidate_summed_logprobs": logprobs,
                        "family_mean_NLL_per_token": gate_module.family_means(records),
                    }
                )
                gate_module.write_json(out, report)
                logger.info("Finished pass %d", index)
        first = report["passes"][0]["candidate_summed_logprobs"]
        later = [p["candidate_summed_logprobs"] for p in report["passes"][1:]]
        diffs = [
            abs(a - b) for other in later for row_id in first for a, b in zip(first[row_id], other[row_id], strict=True)
        ]
        means = [p["family_mean_NLL_per_token"] for p in report["passes"]]
        report["summary"] = {
            "passes": passes,
            "rows": len(first),
            "rows_identical_in_every_pass": sum(
                all(other[row_id] == first[row_id] for other in later) for row_id in first
            ),
            "candidate_scores_compared": len(diffs),
            "candidate_scores_identical": sum(diff == 0.0 for diff in diffs),
            "max_abs_diff_summed_logprob": max(diffs),
            "family_mean_spread": {
                family: max(m[family] for m in means) - min(m[family] for m in means) for family in means[0]
            },
        }
        gate_module.write_json(out, report)
    print(json.dumps(report["summary"]))


@cli.command()
@click.option("--gate", "gate_path", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--baseline", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--candidate", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--out", required=True, type=click.Path(path_type=Path))
def decide(gate_path: Path, baseline: Path, candidate: Path, out: Path):
    """Apply the gate's decision and amendment A1 to two receipts.

    The byte-unchanged clause is false, as the original decision recorded for this candidate.
    """
    with tempfile.TemporaryDirectory() as tmp:
        _, gate_module = load_gate_code(Path(tmp))
        gate = gate_module.load_gate(gate_path)
        base, cand = json.loads(baseline.read_text()), json.loads(candidate.read_text())
        cand["provenance"].setdefault("candidate_kind", "sft")
        decision = gate_module.decide(gate, base, cand, byte_unchanged=False)
        gate_module.write_json(out, decision)
    print(json.dumps({"label": decision["label"], "amendment_A1": decision["amendment_A1"]}, indent=1))


if __name__ == "__main__":
    cli()

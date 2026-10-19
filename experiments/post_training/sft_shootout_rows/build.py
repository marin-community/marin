# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build one arm's Levanter SFT rows from canonical generation traces.

Inputs: one or more rebuild_traces.py outputs (one row per conversation of a Harbor trial attempt), the pool task
table, the LR-sweep replay slice (sweep_data), and for RS-multi, relay and recovery the plain arm's built prefix (it
fixes the task slots, B and, for recovery, the trajectories).

Rows follow the LR-sweep set: input_ids padded to 32,768 with pad id 128001, loss_weights shifted so weight i scores
token i + 1, zero-loss padding rows to a multiple of 16, nothing truncated. Replay is a prefix of the replay slice in
a seeded, fixed order, cut where its supervised tokens reach R = round(B / 4) with B from plain; every arm gets the same
rows (checked by hash against plain's manifest).

Training order: the trainer reads rows in file order (mixture block 16, one block per batch) and concatenates the
epoch files in order without shuffling. So each epoch file is its own seeded permutation of that epoch's rows (seed
'<SEED>:<arm>:epoch-<e>:<pool manifest sha256>', recorded per split), with zero-loss padding rows at the end to a
multiple of 16. Plain, relay and recovery repeat the same rows in each epoch; RS-multi picks new attempts per epoch.

Outputs under --output: train-epoch-<e>-prebuilt.jsonl.gz for e in 0..2 (plain, RS-multi, relay, recovery) or
validation-<family>-prebuilt.jsonl.gz (validation: the shared set for all arms, GLM's first passing trace per
held-out task, no replay), train-index.jsonl (with epoch) or validation-index.jsonl (row -> source),
selection.jsonl (task or slot -> chosen trial),
dropped.jsonl, and manifest.json with per-family supervised and context tokens and file hashes.

Relay per family: each family's relay episodes are taken whole, in their own seeded order, until they reach plain's B
for that family. --relay-families builds only those families, with no replay, as a component marked
COMPONENT_NOT_AN_ARM, so a family can be built and verified before the others finish. The full arm built later from the
same inputs holds the same rows for that family. Fallback (off unless triggered): if a family's relay pool ends below
plain's B for it, the relay arm takes every episode of that family, and --arm plain --match-relay <relay prefix> builds
a plain control whose rows of that family are cut, in a seeded task order, to relay's total.

Usage:
    python -m experiments.post_training.sft_shootout_rows.build --arm plain --traces <rebuilt.jsonl> ... \\
        --pool <tasks.jsonl> --replay <replay slice prefix> --output <prefix>
"""

import argparse
import base64
import collections
import dataclasses
import gzip
import hashlib
import json
import random
import re
import tempfile
from pathlib import Path
from typing import Any, Literal, NamedTuple, get_args

from rigging.filesystem.storage_path import StoragePath
from transformers import AutoTokenizer

from experiments.post_training.sft_shootout_rows import recovery
from experiments.post_training.sft_shootout_rows.arms import (
    SEED,
    Attempt,
    fewest_reasoning,
    first_passing,
    match_plain_to_relay,
    relay_subsample,
    rs_multi_epochs,
)
from experiments.post_training.sft_shootout_rows.traces import HISTORY, TEACHER, RenderedTrace, render_trace
from experiments.post_training.sweep_data.prepare_agentic import (
    BATCH,
    PAD_TOKEN,
    STUDENT_MODEL,
    STUDENT_REVISION,
    TOKENIZER_FILES,
    audit_hashes,
    load_tokenizer,
    output_hashes,
    read_component,
    write_prebuilt,
)
from experiments.post_training.sweep_data.prepare_replay import REPLAY_FRACTION_OF_B
from experiments.post_training.sweep_data.render import training_row
from experiments.post_training.sweep_data.select import CONTEXT

# Pool v5, the frozen pool.
POOL_MANIFEST_SHA256 = "7f3d6f7dbef1ed80d302f251540c72a5c548edb871fd0afed729653d7b6def68"
EPOCHS = 3
# Replay for every arm: R = round(B / 4) supervised tokens (REPLAY_FRACTION_OF_B), so replay is about 20% of a split
# (decided 2026-10-17).
REPLAY_ORDER_SEED = f"{SEED}:replay-order"
# The arms' replay artifact (decided 2026-10-17): continuation v2 (sweep_data.continue_replay), which supersedes the
# frozen slice for the arms because the projected B reached 99.2% of the frozen slice's ceiling. Its first 1,361 rows
# are the frozen slice in frozen order, so every cut below 29,626,824 is identical. The LR sweep keeps its own frozen
# slice.
REPLAY_URI = "s3://marin-us-east-02a/marin/teacher-traces/mopd2/task217-replay-continuation-v2-20261017"
REPLAY_MANIFEST_SHA256 = "32e6e20b30498eb6ae01a167d890e6ab84f573a0e7a6668c8f30c268e9d3fff4"
# Decided before any arm launches (2026-10-17), recorded in every manifest.
REPLAY_CONTINGENCY = (
    "if plain's B exceeds 4x the slice's supervised tokens, extend the slice with a seeded continuation of the same "
    "#202 sampler (same partitions, same quotas, next rows in order) as a new artifact version; keep R = round(B / 4), "
    "never relax it. Applied ahead of the flag (queen, 2026-10-17): the arms use continuation v2 (task #217, "
    "REPLAY_URI, 10,355,114 supervised tokens, B up to 41.4M); its frozen rows come first, so cuts below the old total "
    "are unchanged"
)
LAUNCH_RULE = (
    "plain's manifest (B) and the replay cut are frozen before any arm launches; any later change to either is a new "
    "artifact version with a new name, never an edit"
)
LIMITATION = "results are for trajectories that fit 32,768 tokens in v0's chat template; longer passes are dropped"
# Decided 2026-10-17: keep one multi-turn row per trace, as the community recipes under test do.
FAMILY_NOTES = {
    "tmax": (
        "Terminus-2 rows carry earlier turns' reasoning in context, which the harness does not resend at inference; "
        "v0/GLM length ratio p50 1.70, p99 3.77 (pilot, length_ratio.py)"
    ),
    "swesmith": (
        "mini-swe-agent resends earlier reasoning, so rows match the harness; v0/GLM length ratio p50 1.03, p99 1.07"
    ),
}
SWE_SUBMIT = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
TASK_COMPLETE = re.compile(r'"task_complete"\s*:\s*true')
START_THINK, END_THINK = 128002, 128003
# Written into a relay component prefix (some families, no replay); verify_arms and trainers refuse it as an arm.
COMPONENT_MARKER = "COMPONENT_NOT_AN_ARM"
# Changes to row semantics or manifest accounting, newest last; recorded in every manifest.
CHANGELOG = (
    "2705769260 plain, RS-multi, relay arms from #207 canonical traces; v0 turns masked",
    "1811670ebb shared validation arm; per-family drop counts",
    "e9704e380b drop traces touching mini-swe-agent's tool env",
    "21a1da4d16 drop traces reading a library copy of the task's repo",
    "37801fc8c8 pool v4 pin; drop passes after a harness exception",
    "2b656bcd99 per-family eligibility",
    "c8e908696e drop traces holding a live proxy capability token (decoded JWT)",
    "095f31382d eligibility counts only the arm's own run type (the relay denominator had included plain runs)",
    "45b4478e04 pool v5 pin (frozen pool; drops 11 tasks with broken or hanging verifiers)",
    "d43ab0af02 replay cut at R = round(B / 4) with plain's B; same rows in every arm (hash-checked)",
    "65b0b9d36b continuation slices (base_rows): frozen rows keep their order, continuation rows follow",
    "c1b656a907 relay manifest records the full usable relay pool per family",
    "8af202e2d3 every training arm writes 3 epoch files, each a seeded permutation (seed recorded per split)",
    "e39c3ead56 arms' replay pinned to continuation v2 (frozen #202 slice superseded for the arms)",
    "PRODUCTION production fields: rows_sha256 per split, builder commit and source hashes, trace file identities, "
    "token gate over the inputs, held-out ids, _SUCCESS marker (task #234)",
    "RECOVERY recovery arm: plain's rows with masked failed attempts before teacher turns of 20% of rows; a row "
    "over the context takes a smaller injection (task #225)",
    "RELAYFAMILY relay subsample per family: each family's episodes to its share of plain's B, own seeded order; "
    "--relay-families builds a component (no replay); --match-relay cuts plain to a short relay family (fallback)",
    "CLEAN --require-clean-completion drops passes that did not end on the agent's own submit / task_complete; "
    "--plain-selection fewest-reasoning picks each task's pass with the fewest trained reasoning tokens",
    "RECOVERYVARIANTS recovery TMax failures: seeded variants within each style (v0 TBLite-100 parser errors) and "
    "v0's measured repeats within a point (same style 125/221, identical text 0/221)",
)
# The generation proxy's capability URL carries a secret JWT (three base64url segments starting 'eyJ', payload
# aud == 'iris-proxy'); a trace that printed it (e.g. GLM running 'env') must never become a training row. Tokens are
# decoded, not pattern-matched, because task content holds legitimate JWT fixtures (oauthlib tests); line breaks and
# JSON escapes inside a printed token are tolerated. The scrubbed form ('<capability-token>' in the URL) is harmless and
# only counted.
JWT = re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")
LINE_BREAKS = re.compile(r"(?:\\+[nrt]|\s)+")
SCRUBBED_PROXY_URL = re.compile(r"iris(?:-cw-us-east-02a)?\.oa\.dev/proxy/t/")


def has_capability_token(text: str) -> bool:
    """True when `text` holds a JWT whose payload's audience is the generation proxy."""
    for token in JWT.findall(LINE_BREAKS.sub("", text)):
        payload = token.split(".")[1]
        try:
            claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        except ValueError:
            continue
        audience = claims.get("aud") if isinstance(claims, dict) else None
        if audience == "iris-proxy" or (isinstance(audience, list) and "iris-proxy" in audience):
            return True
    return False


# Defense in depth for the same leak class: a library location (site-packages, dist-packages, a vendored copy, conda's
# package cache, the Go module cache) followed by a path component named after the task repo's import name, e.g.
# .../site-packages/setuptools/_vendor/typeguard/_checkers.py. Pool rows carry the repo's import names.
LIBRARY_LOCATION = r"(?:site-packages|dist-packages|_vendor|/opt/miniconda3/pkgs|/go/pkg/mod)/(?:[^\s\"'<>]*/)?"


def library_copy_pattern(import_names: list[str]) -> re.Pattern | None:
    if not import_names:
        return None
    names = "|".join(re.escape(name) for name in import_names)
    return re.compile(LIBRARY_LOCATION + rf"(?:{names})(?:/|\.py\b|-[0-9]|@)")


# mini-swe-agent's own tool environment, installed in the sandbox at agent setup, holds clean copies of some repos
# (pool v3 removes those repos); a trace that reads anything there is dropped. Pattern from the generation pilot: path
# reads, find/grep listings, 'uv tool dir|list|run' and the tool's own bin/lib all leave one of these strings.
AGENT_TOOL_ENV = re.compile(r"(?:\.local/share/uv/tools|uv\s+tool\s+(?:dir|list|run)|mini-swe-agent/(?:bin|lib)/)")


Arm = Literal["plain", "rs-multi", "relay", "recovery", "validation"]


def load_pool(uri: str) -> dict[str, dict]:
    rows = [json.loads(line) for line in StoragePath(uri).read_text().splitlines()]
    return {row["task_id"]: row for row in rows}


# Runs that must never feed training rows, with the reason.
DENIED_RUNS = {
    "generation/relay/swe-relay1": (
        "ran before the relay fix: v0 answered mini-swe 'Tool call error' turns that GLM should repair"
    ),
}


def load_traces(uris: list[str], files: list[dict] | None = None) -> list[dict]:
    for uri in uris:
        for run, reason in DENIED_RUNS.items():
            if f"/{run}/" in uri:
                raise ValueError(f"{uri} is a denied run: {reason}")
    traces = []
    for uri in uris:
        data = StoragePath(uri).read_bytes()
        rows = [json.loads(line) | {"source": uri} for line in data.decode().splitlines()]
        traces += rows
        if files is not None:
            files.append(trace_file_info(uri, data, rows))
    keys = [(trace["trial_id"], trace.get("conversation", 0)) for trace in traces]
    if len(set(keys)) != len(keys):
        raise ValueError("A (trial id, conversation) pair appears twice across the trace files")
    return traces


def trace_file_info(uri: str, data: bytes, rows: list[dict]) -> dict:
    """A trace file's identity: content hash, row count, and the generation run's Harbor commit and policy."""
    info = {
        "uri": uri,
        "sha256": hashlib.sha256(data).hexdigest(),
        "rows": len(rows),
        "token_hits": sum(has_capability_token(line) for line in data.decode().splitlines()),
    }
    run = StoragePath(uri.rsplit("/", 1)[0])
    for name, keys in (("lock.json", ("harbor", "git_commit_hash")), ("driver_config.json", ("policy",))):
        try:
            value = json.loads((run / name).read_text())
        except FileNotFoundError:
            continue
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        info["harbor_commit" if name == "lock.json" else "policy"] = value
    return info


class Candidate(NamedTuple):
    """One attempt the arm may use: its selection record, its rendering and its source trace."""

    attempt: Attempt
    rendered: RenderedTrace
    trace: dict


def clean_completion(trace: dict) -> bool:
    """The trial ended on the agent's own completion: its last assistant turn issues mini-swe-agent's submit command
    (SWE-smith) or sets Terminus-2's task_complete to true (TMax), rather than being cut at the context limit."""
    assistant = [m for m in trace["messages"] if m["role"] == "assistant"]
    if not assistant:
        return False
    last = (assistant[-1].get("content") or "") + json.dumps(assistant[-1].get("tool_calls") or [])
    if trace["family"] == "swesmith":
        return SWE_SUBMIT in last
    return bool(TASK_COMPLETE.search(last))


def think_tokens(rendered: RenderedTrace) -> int:
    """Trained tokens strictly inside <|start_think|> ... <|end_think|>."""
    count, inside = 0, False
    for token, trained in zip(rendered.ids, rendered.mask, strict=True):
        if token == START_THINK:
            inside = True
        elif token == END_THINK:
            inside = False
        elif inside and trained:
            count += 1
    return count


def candidates(
    arm: Arm,
    traces: list[dict],
    pool: dict[str, dict],
    tokenizer,
    allow_summarized: bool = False,
    require_clean_completion: bool = False,
) -> tuple[dict[str, Candidate], list[dict]]:
    """Render every attempt the arm may use; return ({trial_id: Candidate}, dropped records).

    allow_summarized (recovery handoff rows only) keeps the conversations of a summarized trial, whose echoed history
    turns are tagged HISTORY by recovery.handoff_traces. require_clean_completion also drops passing trials that did
    not end on the agent's own completion (clean_completion).
    """
    kept, dropped = {}, []
    for trace in traces:
        reason = None
        task = pool.get(trace.get("task_id"))
        if task is None:
            reason = "task_not_in_pool"
        elif task["heldout"] != (arm == "validation"):
            reason = "heldout_task" if task["heldout"] else "not_heldout"
        elif task["family"] != trace["family"]:
            reason = "family_mismatch"
        elif trace["relay"] != (arm == "relay"):  # validation, like plain, uses plain GLM runs
            reason = "relay_run" if trace["relay"] else "plain_run"
        elif trace.get("n_conversations", 1) > 1 and not allow_summarized:
            # Terminus-2 summarized this trial's history: the trial does not fit one 32k row.
            reason = "summarized_trial"
        elif trace["problems"]:
            reason = "trace_problems"
        elif not trace["passed"]:
            reason = "not_passed"
        elif trace.get("exception"):
            # A trial that still passed after a harness exception (e.g. AgentTimeoutError) stops mid-task.
            reason = "harness_exception"
        elif require_clean_completion and not clean_completion(trace):
            # Passed because the verifier scored the working tree after a context-limit cutoff, not a completion.
            reason = "no_clean_completion"
        elif has_capability_token(json.dumps([trace["messages"], trace.get("tools")])):
            reason = "contains_proxy_capability_token"
        elif AGENT_TOOL_ENV.search(json.dumps(trace["messages"])):
            reason = "touches_agent_tool_env"
        elif (pattern := library_copy_pattern(task.get("import_names") or [])) and pattern.search(
            json.dumps(trace["messages"])
        ):
            reason = "touches_library_copy_of_repo"
        if reason is None:
            authors = [m.get("author") for m in trace["messages"] if m["role"] == "assistant"]
            if TEACHER not in authors:
                reason = "no_teacher_turn"
            elif arm != "relay" and set(authors) - ({HISTORY} if allow_summarized else set()) != {TEACHER}:
                reason = "student_turn_in_plain_run"
        if reason is None:
            try:
                rendered = render_trace(tokenizer, trace["messages"], trace.get("tools"))
            except ValueError as error:
                reason = f"render_error: {error}"
            else:
                if len(rendered.ids) > CONTEXT:
                    reason = "over_context"
        if reason:
            dropped.append(
                {
                    "trial_id": trace["trial_id"],
                    "conversation": trace.get("conversation", 0),
                    "task_id": trace.get("task_id"),
                    "family": trace.get("family"),
                    "reason": reason,
                }
            )
            continue
        attempt = Attempt(
            task_id=trace["task_id"],
            family=trace["family"],
            trial_id=trace["trial_id"],
            sample=trace["sample"],
            retry=trace["retry"],
            supervised=rendered.supervised,
            context=len(rendered.ids),
            authors=rendered.authors,
            reasoning=think_tokens(rendered),
        )
        kept[trace["trial_id"]] = Candidate(attempt, rendered, trace)
    return kept, dropped


def replay_rows(prefix: str) -> tuple[list[str], list[dict], dict]:
    manifest, manifest_bytes, files = read_component(
        StoragePath(prefix), ("replay-prebuilt.jsonl.gz", "replay-rows.jsonl")
    )
    lines = gzip.decompress(files["replay-prebuilt.jsonl.gz"]).decode().splitlines()[: manifest["rows"]]
    index = []
    for line in files["replay-rows.jsonl"].decode().splitlines():
        entry = json.loads(line)
        if "row" in entry:
            entry["row_in_group"] = entry.pop("row")
        index.append({"family": "replay"} | entry)
    info = {
        "uri": prefix,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "supervised_tokens": manifest["supervised_tokens"],
        # A continuation slice (sweep_data/continue_replay.py) holds the frozen rows first; older slices are all base.
        "base_rows": manifest.get("base_rows", len(lines)),
    }
    return lines, index, info


def replay_order(rows: int, base_rows: int) -> list[int]:
    """The frozen rows in seeded order, then any continuation rows in their own seeded order.

    For a continuation slice the first part equals the frozen slice's order, so every cut the frozen slice reaches is
    unchanged.
    """
    base = list(range(base_rows))
    random.Random(REPLAY_ORDER_SEED).shuffle(base)
    extra = list(range(base_rows, rows))
    random.Random(f"{REPLAY_ORDER_SEED}:continuation").shuffle(extra)
    return base + extra


def cut_replay(
    entries: list[tuple[str, dict]], b_tokens: int, base_rows: int | None = None
) -> tuple[list[tuple[str, dict]], dict]:
    """Take the replay slice in replay_order up to the first prefix whose supervised tokens reach round(B / 4).

    Whole rows only: the row that crosses R is kept in full (the overshoot is retained, no boundary-row masking).
    """
    target = round(REPLAY_FRACTION_OF_B * b_tokens)
    order = replay_order(len(entries), len(entries) if base_rows is None else base_rows)
    cut, supervised = [], 0
    for position in order:
        if supervised >= target:
            break
        cut.append(entries[position])
        supervised += entries[position][1]["supervised_tokens"]
    if supervised < target:
        raise ValueError(
            f"Replay slice holds {supervised} supervised tokens, short of R = {target}; {REPLAY_CONTINGENCY}"
        )
    if len(cut) > len(entries):
        raise ValueError("Replay cut holds more rows than the slice")
    return cut, {
        "rule": "whole rows, first seeded prefix reaching R; overshoot retained, no boundary-row masking",
        "contingency": REPLAY_CONTINGENCY,
        "launch_rule": LAUNCH_RULE,
        "order_seed": REPLAY_ORDER_SEED,
        "target_supervised_tokens": target,
        "supervised_tokens": supervised,
        "rows": len(cut),
        "slice_rows": len(entries),
        "rows_sha256": hashlib.sha256("".join(line + "\n" for line, _ in cut).encode()).hexdigest(),
    }


def epoch_seed(arm: str, epoch: int) -> str:
    return f"{SEED}:{arm}:epoch-{epoch}:{POOL_MANIFEST_SHA256}"


def write_epochs(out: Path, arm: str, rows_for_epoch, splits: dict) -> list[dict]:
    """Write train-epoch-<e> files, each a seeded permutation of its epoch's rows; return the index with epochs."""
    index = []
    for epoch in range(EPOCHS):
        epoch_index, splits[f"epoch-{epoch}"] = write_split(
            out / f"train-epoch-{epoch}-prebuilt.jsonl.gz", rows_for_epoch(epoch), epoch_seed(arm, epoch)
        )
        index += [{"epoch": epoch} | item for item in epoch_index]
    return index


def write_split(path: Path, entries: list[tuple[str, dict]], seed: str) -> tuple[list[dict], dict]:
    """Shuffle (row json, index entry) pairs with `seed`, write rows plus zero-loss padding, re-read and check."""
    order = list(range(len(entries)))
    random.Random(seed).shuffle(order)
    index = [{"row": row_number} | entries[position][1] for row_number, position in enumerate(order)]
    _, padding = write_prebuilt(path, (entries[position][0] for position in order))
    supervised = collections.Counter()
    context = collections.Counter()
    digest = hashlib.sha256()
    with gzip.open(path, "rt") as stream:
        for row_number, line in enumerate(stream):
            digest.update(line.encode())
            row = json.loads(line)
            if len(row["input_ids"]) != CONTEXT or len(row["loss_weights"]) != CONTEXT:
                raise ValueError(f"Bad row shape at {row_number}")
            if row_number >= len(index):
                if any(row["loss_weights"]):
                    raise ValueError("Padding row carries loss")
                continue
            entry = index[row_number]
            if sum(row["loss_weights"]) != entry["supervised_tokens"] or any(
                row["loss_weights"][entry["context_tokens"] - 1 :]
            ):
                raise ValueError(f"Row {row_number} differs from its index entry")
            supervised[entry["family"]] += entry["supervised_tokens"]
            context[entry["family"]] += entry["context_tokens"]
    return index, {
        "rows": len(index),
        "padding_rows": padding,
        # Rows the trainer reads per epoch of this file (a multiple of 16); the consumer sets its block size from it.
        "padded_rows": len(index) + padding,
        "permutation_seed": seed,
        "rows_sha256": digest.hexdigest(),
        "steps": (len(index) + padding) // BATCH,
        "supervised_tokens": dict(supervised),
        "context_tokens": dict(context),
        "supervised_total": sum(supervised.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=get_args(Arm), required=True)
    parser.add_argument("--traces", nargs="+", required=True)
    parser.add_argument("--pool", required=True, help="pool tasks.jsonl")
    parser.add_argument(
        "--pool-manifest", required=True, help="pool manifest.json (hash-checked against POOL_MANIFEST_SHA256)"
    )
    parser.add_argument("--replay", required=True, help="replay slice prefix (sweep_data)")
    parser.add_argument("--plain", help="built plain arm prefix (required for rs-multi, relay and recovery)")
    parser.add_argument(
        "--recovery-swe-mode",
        choices=(recovery.HARNESS_FAITHFUL, recovery.WITH_FAILED_REPLY),
        default=recovery.HARNESS_FAITHFUL,
        help="SWE-smith injections: error messages only, as mini-swe-agent sends them (default), or with failed replies",
    )
    parser.add_argument(
        "--recovery-handoff",
        action="store_true",
        help="also add context-handoff rows from summarized TMax trials (off by default; see recovery.py)",
    )
    parser.add_argument(
        "--relay-families",
        nargs="+",
        choices=("swesmith", "tmax"),
        help="relay only: build just these families as a component (no replay, not a trainable arm); default all",
    )
    parser.add_argument(
        "--match-relay",
        help=(
            "plain only, fallback mode, off unless triggered: a built relay arm prefix; plain's rows of a family whose "
            "relay total ended below plain's B for it are cut to relay's total (needs --plain)"
        ),
    )
    parser.add_argument(
        "--require-clean-completion",
        action="store_true",
        help="drop passing trials that did not end on the agent's own submit / task_complete (see clean_completion)",
    )
    parser.add_argument(
        "--plain-selection",
        choices=("first", "fewest-reasoning"),
        default="first",
        help="plain only: per task, the first passing attempt (default) or the one with the fewest reasoning tokens",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer-dir", help="local v0 tokenizer files (hash-checked); default: the regional copy")
    parser.add_argument(
        "--test-inputs", action="store_true", help="skip the pool and replay pins (local validation only)"
    )
    parser.add_argument("--builder-commit", help="git commit of this builder (required for a production build)")
    args = parser.parse_args()
    if not args.test_inputs and not args.builder_commit:
        raise ValueError("--builder-commit is required unless --test-inputs")
    if (args.relay_families and args.arm != "relay") or (args.match_relay and args.arm != "plain"):
        raise ValueError("--relay-families is for the relay arm and --match-relay for plain")
    return args


def check_pins(args: argparse.Namespace) -> None:
    """Production builds only: the replay artifact, the pool manifest and tasks.jsonl must match their pins."""
    replay_manifest = (StoragePath(args.replay) / "manifest.json").read_bytes()
    if hashlib.sha256(replay_manifest).hexdigest() != REPLAY_MANIFEST_SHA256:
        raise ValueError(f"Replay is not the arms' replay artifact; use {REPLAY_URI}")
    pool_manifest = StoragePath(args.pool_manifest).read_bytes()
    if hashlib.sha256(pool_manifest).hexdigest() != POOL_MANIFEST_SHA256:
        raise ValueError("Pool manifest is not pool v5")
    pool_hash = hashlib.sha256(StoragePath(args.pool).read_bytes()).hexdigest()
    if pool_hash != json.loads(pool_manifest)["file_sha256"]["tasks.jsonl"]:
        raise ValueError("tasks.jsonl does not match the pool manifest")


def v0_tokenizer(tokenizer_dir: str | None, directory: Path):
    """v0's tokenizer from local files or the regional copy, each file checked against the campaign-3 audit."""
    expected = audit_hashes()
    if tokenizer_dir:
        for name in TOKENIZER_FILES:
            if hashlib.sha256((Path(tokenizer_dir) / name).read_bytes()).hexdigest() != expected[name]:
                raise ValueError(f"Local tokenizer file differs from the audit: {name}")
        return AutoTokenizer.from_pretrained(tokenizer_dir)
    (directory / "tokenizer").mkdir()
    return load_tokenizer(directory / "tokenizer", expected)


def token_gate(trace_files: list[dict]) -> dict:
    """The publication gate's criterion over the exact input files; refuse to build on any hit."""
    gate = {
        "criterion": "check_tokens.scan: has_capability_token (decoded JWT, aud == iris-proxy) on every raw line",
        "files": len(trace_files),
        "lines": sum(info["rows"] for info in trace_files),
        "hits": sum(info["token_hits"] for info in trace_files),
    }
    if gate["hits"]:
        raise ValueError(f"Capability tokens in the input traces: {gate}")
    return gate


def entry(candidate: Candidate) -> tuple[str, dict]:
    """A candidate's row line and its index entry."""
    attempt, rendered, trace = candidate
    line = json.dumps(training_row(rendered.ids, rendered.mask, CONTEXT, PAD_TOKEN), separators=(",", ":"))
    return line, {
        "family": attempt.family,
        "task_id": attempt.task_id,
        "trial_id": attempt.trial_id,
        "trial_name": trace.get("trial_name"),
        "sample": attempt.sample,
        "retry": attempt.retry,
        "run": trace.get("run"),
        "turns": dict(collections.Counter(attempt.authors)),
        "supervised_tokens": attempt.supervised,
        "context_tokens": attempt.context,
    }


class Replay:
    """The replay slice's rows; `cut` takes the arm's R = round(B / 4) prefix and records it in `info`."""

    def __init__(self, prefix: str, plain: dict | None):
        lines, index, self.info = replay_rows(prefix)
        self.entries = list(zip(lines, index, strict=True))
        self.plain = plain

    def cut(self, b_tokens: int) -> list[tuple[str, dict]]:
        cut, cut_info = cut_replay(self.entries, b_tokens, self.info["base_rows"])
        if self.plain is not None and cut_info["rows_sha256"] != self.plain["replay"].get("rows_sha256"):
            raise ValueError("Replay rows differ from plain's; rebuild plain with this builder")
        self.info.update({"slice_supervised_tokens": self.info["supervised_tokens"]} | cut_info)
        return cut


class ArmRows(NamedTuple):
    selection: list[dict]
    index: list[dict]
    splits: dict
    b_tokens: int | None
    # Arm-specific manifest fields (relay family budgets, a relay component, the plain fallback match).
    extra: dict


def build_rows(
    arm: Arm,
    kept: dict[str, Candidate],
    attempts: list[Attempt],
    plain: dict | None,
    replay: Replay,
    out: Path,
    relay_families: list[str] | None = None,
    match_relay: dict | None = None,
    plain_selection: str = "first",
) -> ArmRows:
    """Select the arm's attempts and write its split files into `out`.

    relay_families (relay only): build just these families as a component, without replay. match_relay (plain only):
    the built relay arm's manifest; plain's rows of a family whose relay total is below plain's B for it are cut to
    relay's total, and replay stays plain's cut."""
    splits: dict = {}
    if arm == "plain":
        chosen = (fewest_reasoning if plain_selection == "fewest-reasoning" else first_passing)(attempts)
        extra = {"plain_selection": plain_selection}
        replay_b = b_tokens = sum(a.supervised for a in chosen.values())
        if match_relay is not None:
            if plain is None or sorted(chosen) != plain["task_slots"]:
                raise ValueError("--match-relay needs --plain, and this build's tasks must be plain's task slots")
            plain_b, relay_totals = plain_family_b(plain), family_supervised(match_relay)
            chosen = match_plain_to_relay(chosen, plain_b, relay_totals)
            b_tokens, replay_b = sum(a.supervised for a in chosen.values()), plain["B_supervised_tokens"]
            extra = {
                "matched_to_relay": {
                    "relay_builder_commit": match_relay["builder_commit"],
                    "relay_output_hashes": match_relay["output_hashes"],
                    "plain_family_b": plain_b,
                    "relay_family_totals": relay_totals,
                    "matched_family_totals": {
                        f: sum(a.supervised for a in chosen.values() if a.family == f) for f in plain_b
                    },
                    "note": "fallback: plain cut to relay's total in each family where relay ended below plain's B",
                }
            }
        selection = [{"task_id": task, "trial_id": a.trial_id, "sample": a.sample} for task, a in sorted(chosen.items())]
        rows = [entry(kept[a.trial_id]) for _, a in sorted(chosen.items())] + replay.cut(replay_b)
        index = write_epochs(out, "plain", lambda epoch: rows, splits)
        return ArmRows(selection, index, splits, b_tokens, extra)
    if arm == "validation":
        # One fixed set for every arm: GLM's first passing trace per held-out task, per family, no replay.
        chosen = first_passing(attempts)
        selection = [{"task_id": task, "trial_id": a.trial_id, "sample": a.sample} for task, a in sorted(chosen.items())]
        index = []
        for family in sorted({a.family for a in chosen.values()}):
            rows = [entry(kept[a.trial_id]) for _, a in sorted(chosen.items()) if a.family == family]
            family_index, splits[f"validation-{family}"] = write_split(
                out / f"validation-{family}-prebuilt.jsonl.gz", rows, f"{SEED}:validation:{family}"
            )
            index += [{"split": f"validation-{family}"} | item for item in family_index]
        return ArmRows(selection, index, splits, None, {})
    if plain is None:
        raise ValueError("--plain is required for rs-multi and relay")
    b_tokens = plain["B_supervised_tokens"]
    if arm == "rs-multi":
        tasks = sorted(plain["task_slots"])
        missing = [task for task in tasks if not any(a.task_id == task for a in attempts)]
        if missing:
            raise ValueError(f"Plain task slots without RS candidates: {missing[:5]}")
        plan = rs_multi_epochs([a for a in attempts if a.task_id in set(tasks)], tasks, EPOCHS)
        replay_cut = replay.cut(b_tokens)
        selection = [
            {"epoch": epoch, "task_id": task, "trial_id": a.trial_id, "sample": a.sample}
            for epoch, chosen in enumerate(plan)
            for task, a in sorted(chosen.items())
        ]
        index = write_epochs(
            out,
            "rs-multi",
            lambda epoch: [entry(kept[a.trial_id]) for _, a in sorted(plan[epoch].items())] + replay_cut,
            splits,
        )
        return ArmRows(selection, index, splits, b_tokens, {})
    budgets = plain_family_b(plain)
    families = sorted(relay_families or budgets)
    component = families != sorted(budgets)
    chosen_episodes, totals = relay_subsample(attempts, {family: budgets[family] for family in families})
    selection = [{"trial_id": a.trial_id, "task_id": a.task_id, "sample": a.sample} for a in chosen_episodes]
    rows = [entry(kept[a.trial_id]) for a in chosen_episodes]
    if not component:
        rows += replay.cut(b_tokens)
    index = write_epochs(out, "relay", lambda epoch: rows, splits)
    extra = {
        "relay_families": {
            family: {"plain_b": budgets[family], "achieved": totals[family], "delta": totals[family] - budgets[family]}
            for family in families
        },
        "relay_component": component,
    }
    return ArmRows(selection, index, splits, sum(budgets[f] for f in families) if component else b_tokens, extra)


def family_supervised(manifest: dict) -> dict[str, int]:
    """A built arm's agentic supervised tokens per family (epoch 0; plain and relay repeat their rows every epoch)."""
    return {f: v for f, v in manifest["splits"]["epoch-0"]["supervised_tokens"].items() if f != "replay"}


def plain_family_b(plain: dict) -> dict[str, int]:
    """Plain's B split by agentic family."""
    budgets = family_supervised(plain)
    if sum(budgets.values()) != plain["B_supervised_tokens"]:
        raise ValueError("Plain's per-family supervised tokens do not sum to its B")
    return budgets


def handoff_rows(
    traces: list[dict],
    pool: dict[str, dict],
    tokenizer,
    kept: dict[str, Candidate],
    chosen: dict[str, Attempt],
    recovered: dict[str, RenderedTrace],
    plain: dict,
    seed: str,
) -> tuple[list[Attempt], set[str], dict]:
    """Context-handoff rows for the recovery arm, and the clean TMax rows dropped to hold B; adds them to `kept`.

    Returns (handoff attempts, dropped trial ids, manifest stats).
    """
    handoff_kept, handoff_dropped = candidates(
        "recovery", recovery.handoff_traces(traces), pool, tokenizer, allow_summarized=True
    )
    kept.update(handoff_kept)
    tmax_rows = sum(a.family == "tmax" for a in chosen.values())
    handoff = recovery.plan_handoff([c.attempt for c in handoff_kept.values()], tmax_rows, seed)
    added = sum(a.supervised for a in handoff)
    clean = [a for a in chosen.values() if a.family == "tmax" and a.trial_id not in recovered]
    dropped_clean = recovery.hold_budget(clean, added, seed)
    removed = sum(a.supervised for a in dropped_clean)
    stats = {
        "rule": (
            "conversations 1+ of passing summarized TMax plain trials (any pool task), seeded order, at most "
            f"{recovery.HANDOFF_SHARE:.0%} of plain's TMax rows; clean TMax rows dropped in seeded order until "
            "they cover the handoff rows' supervised tokens"
        ),
        "candidates": len(handoff_kept),
        "candidate_drops": dict(collections.Counter(d["reason"].split(":")[0] for d in handoff_dropped)),
        "rows": len(handoff),
        "kinds": dict(collections.Counter(recovery.handoff_kind(kept[a.trial_id].trace["messages"]) for a in handoff)),
        "supervised_tokens_added": added,
        "clean_tmax_rows_dropped": len(dropped_clean),
        "supervised_tokens_dropped": removed,
        "B_minus_plain_B": added - removed,
        "tasks_outside_plain_slots": len({a.task_id for a in handoff} - set(plain["task_slots"])),
    }
    return handoff, {a.trial_id for a in dropped_clean}, stats


def build_recovery_rows(
    args: argparse.Namespace,
    traces: list[dict],
    pool: dict[str, dict],
    tokenizer,
    kept: dict[str, Candidate],
    attempts: list[Attempt],
    plain: dict,
    replay: Replay,
    out: Path,
) -> tuple[ArmRows, dict]:
    """Plain's rows with masked failed attempts injected (recovery.py); optionally with context-handoff rows."""
    chosen = first_passing(attempts)
    plain_selection = [
        json.loads(line) for line in (StoragePath(args.plain) / "selection.jsonl").read_text().splitlines()
    ]
    if {(s["task_id"], s["trial_id"]) for s in plain_selection} != {(t, a.trial_id) for t, a in chosen.items()}:
        raise ValueError("Recovery's trajectories differ from plain's selection; build both from the same traces")
    b_tokens = sum(a.supervised for a in chosen.values())
    if b_tokens != plain["B_supervised_tokens"]:
        raise ValueError("Recovery B differs from plain's")
    selection = [{"task_id": task, "trial_id": a.trial_id, "sample": a.sample} for task, a in sorted(chosen.items())]
    seed = f"{SEED}:recovery:{POOL_MANIFEST_SHA256}"
    recovered: dict[str, RenderedTrace] = {}
    stats = recovery.build_recovery(chosen, kept, tokenizer, args.recovery_swe_mode, seed, recovered)
    handoff, removed = [], set()
    if args.recovery_handoff:
        handoff, removed, stats["handoff"] = handoff_rows(traces, pool, tokenizer, kept, chosen, recovered, plain, seed)
        b_tokens += stats["handoff"]["B_minus_plain_B"]
        selection += [{"task_id": a.task_id, "trial_id": a.trial_id, "handoff": True} for a in handoff]
        selection = [item for item in selection if item["trial_id"] not in removed]

    def recovery_entry(attempt: Attempt, extra: dict) -> tuple[str, dict]:
        candidate = kept[attempt.trial_id]
        if attempt.trial_id in recovered:
            rendered = recovered[attempt.trial_id]
            candidate = Candidate(dataclasses.replace(attempt, context=len(rendered.ids)), rendered, candidate.trace)
            extra = extra | {"plain_context_tokens": attempt.context}
        line, item = entry(candidate)
        return line, item | extra

    rows = [
        recovery_entry(a, {"injected": a.trial_id in recovered})
        for _, a in sorted(chosen.items())
        if a.trial_id not in removed
    ]
    rows += [recovery_entry(a, {"handoff": recovery.handoff_kind(kept[a.trial_id].trace["messages"])}) for a in handoff]
    # Replay is cut at plain's B so its rows hash-match plain's, with or without handoff rows.
    rows += replay.cut(plain["B_supervised_tokens"])
    splits: dict = {}
    index = write_epochs(out, "recovery", lambda epoch: rows, splits)
    return ArmRows(selection, index, splits, b_tokens, {}), stats


def eligibility(arm: Arm, traces: list[dict], pool: dict[str, dict], attempts: list[Attempt]) -> dict:
    """Per family: tasks the arm's own run type attempted, and how many have a usable attempt."""
    attempted: dict[str, set[str]] = collections.defaultdict(set)
    for trace in traces:
        task = pool.get(trace.get("task_id"))
        if (
            task is not None
            and task["heldout"] == (arm == "validation")
            and task["family"] == trace["family"]
            and trace["relay"] == (arm == "relay")
        ):
            attempted[trace["family"]].add(trace["task_id"])
    eligible: dict[str, set[str]] = collections.defaultdict(set)
    for attempt in attempts:
        eligible[attempt.family].add(attempt.task_id)
    return {
        family: {
            "tasks_attempted": len(tasks),
            "tasks_eligible": len(eligible[family] & tasks),
            "rate": len(eligible[family] & tasks) / len(tasks) if tasks else None,
            "note": "eligible = at least one passing attempt that fits 32,768 tokens and survives every drop rule",
        }
        for family, tasks in sorted(attempted.items())
    }


def heldout_rows(arm: Arm, index: list[dict], pool: dict[str, dict]) -> int:
    """Count index rows from held-out tasks; validation must be all held out, training none."""
    heldout = sum(1 for item in index if item.get("task_id") and pool[item["task_id"]]["heldout"])
    agentic = sum(1 for item in index if item.get("task_id"))
    if arm == "validation" and heldout != agentic:
        raise ValueError("Validation rows must all come from held-out tasks")
    if arm != "validation" and heldout:
        raise ValueError(f"{heldout} rows come from held-out tasks")
    return heldout


def candidate_counts(kept: dict[str, Candidate], dropped: list[dict]) -> dict:
    return {
        "kept": len(kept),
        "dropped": dict(collections.Counter(d["reason"].split(":")[0] for d in dropped)),
        "dropped_by_family": {
            family: dict(collections.Counter(d["reason"].split(":")[0] for d in dropped if d["family"] == family))
            for family in sorted({d["family"] for d in dropped if d["family"]})
        },
        # One count per summarized trial (each contributes one dropped row per conversation).
        "kept_with_scrubbed_proxy_url": sum(
            1 for candidate in kept.values() if SCRUBBED_PROXY_URL.search(json.dumps(candidate.trace["messages"]))
        ),
        "library_copy_drops_by_task": dict(
            collections.Counter(d["task_id"] for d in dropped if d["reason"] == "touches_library_copy_of_repo")
        ),
        "agent_tool_env_drops_by_task": dict(
            collections.Counter(d["task_id"] for d in dropped if d["reason"] == "touches_agent_tool_env")
        ),
        "summarized_trials_by_family": dict(
            collections.Counter(
                d["family"] for d in dropped if d["reason"] == "summarized_trial" and d["conversation"] == 0
            )
        ),
    }


def relay_pool(attempts: list[Attempt]) -> dict:
    """Every usable relay episode before the subsample to B, so parity with plain can be read off directly."""
    pool_by_family: dict[str, dict] = {}
    for attempt in attempts:
        family = pool_by_family.setdefault(attempt.family, {"episodes": 0, "supervised_tokens": 0, "tasks": set()})
        family["episodes"] += 1
        family["supervised_tokens"] += attempt.supervised
        family["tasks"].add(attempt.task_id)
    return {
        family: {"episodes": v["episodes"], "supervised_tokens": v["supervised_tokens"], "tasks": len(v["tasks"])}
        for family, v in sorted(pool_by_family.items())
    }


def build_manifest(
    args: argparse.Namespace,
    pool: dict[str, dict],
    kept: dict[str, Candidate],
    dropped: list[dict],
    attempts: list[Attempt],
    trace_files: list[dict],
    gate: dict,
    eligible: dict,
    rows: ArmRows,
    replay: Replay,
    out: Path,
) -> dict[str, Any]:
    splits, b_tokens = rows.splits, rows.b_tokens
    manifest = {
        "task": 211,
        "arm": args.arm,
        "student_model": STUDENT_MODEL,
        "student_revision": STUDENT_REVISION,
        "pad_token_id": PAD_TOKEN,
        "max_sequence_tokens": CONTEXT,
        "batch_size": BATCH,
        "seed": SEED,
        "pool_manifest_sha256": None if args.test_inputs else POOL_MANIFEST_SHA256,
        "test_inputs": args.test_inputs,
        "traces": args.traces,
        "builder_commit": args.builder_commit,
        "builder_sources_sha256": {
            f"{path.parent.name}/{path.name}": hashlib.sha256(path.read_bytes()).hexdigest()
            for directory in (Path(__file__).parent, Path(training_row.__code__.co_filename).parent)
            for path in sorted(directory.glob("*.py"))
        },
        "trace_files": trace_files,
        "token_gate": gate,
        "heldout_task_ids": {
            family: sorted(task for task, row in pool.items() if row["heldout"] and row["family"] == family)
            for family in sorted({row["family"] for row in pool.values()})
        },
        "rows_sha256_definition": (
            "splits[<split>].rows_sha256 = sha256 of the decompressed split file's bytes: every line including "
            "the zero-loss padding rows, each followed by a newline; replay.rows_sha256 hashes the replay cut in "
            "replay order (before the epoch permutation)"
        ),
        "candidates": candidate_counts(kept, dropped),
        "eligibility": eligible,
        "limitation": LIMITATION,
        "family_notes": {family: FAMILY_NOTES[family] for family in sorted(eligible) if family in FAMILY_NOTES},
        "over_context_by_family": {
            family: reasons.get("over_context", 0)
            for family, reasons in sorted(
                {
                    family: dict(collections.Counter(d["reason"] for d in dropped if d["family"] == family))
                    for family in {d["family"] for d in dropped if d["family"]}
                }.items()
            )
        },
        "builder_changelog": list(CHANGELOG),
        "B_supervised_tokens": b_tokens,
        "task_slots": sorted({item["task_id"] for item in rows.selection}) if args.arm == "plain" else None,
        "splits": splits,
        "replay": (
            None
            if args.arm == "validation" or rows.extra.get("relay_component")
            else (
                replay.info
                | {
                    "note": (
                        "seeded prefix of the replay slice cut at R = round(B / 4) with plain's B, same rows in "
                        "every arm; share_of_supervised is the achieved share of each split's supervised tokens"
                    ),
                    "share_of_supervised": {
                        name: s["supervised_tokens"].get("replay", 0) / s["supervised_total"]
                        for name, s in splits.items()
                    },
                    "ratio_to_B": replay.info["supervised_tokens"] / b_tokens if b_tokens else None,
                }
            )
        ),
        "selection_rule": {
            "plain": "first passing attempt per eligible task by sample index",
            "rs-multi": (
                f"plain's task slots; per epoch (seed '{SEED}:rs-multi:<epoch>') one passing attempt per slot, "
                "without replacement until a task's passes are used, then refilled"
            ),
            "relay": (
                f"passing relay episodes with >=1 GLM turn; per family a seeded order '{SEED}:relay:<family>', "
                "whole episodes to that family's share of plain's B"
            ),
            "recovery": (
                "plain's trajectories (same selection, same B); a seeded 20% of each family's rows get 1-2 "
                "injection points with 1-3 masked failed attempts each (recovery.py)"
            ),
            "validation": "first passing GLM attempt per held-out pool task, one file per family, no replay",
        }[args.arm],
        "loss_mask": (
            "assistant turns authored by glm; v0 turns, injected failed attempts, harness error messages, prompts and "
            "tool output are context"
        ),
        "heldout_rows": heldout_rows(args.arm, rows.index, pool),
        "output_hashes": output_hashes(out),
    } | rows.extra
    if args.arm == "relay":
        manifest["relay_supervised_tokens"] = sum(family_supervised(manifest).values())
        manifest["relay_pool"] = relay_pool(attempts)
    return manifest


def upload(out: Path, output: StoragePath, manifest: dict, production: bool) -> None:
    """Write manifest.json into `out`, copy every file (manifest last) re-hashing each, then _SUCCESS if production."""
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    output.mkdirs()
    for path in sorted(out.iterdir(), key=lambda path: path.name == "manifest.json"):
        (output / path.name).write_bytes(path.read_bytes())
        if path.name != "manifest.json":
            if hashlib.sha256((output / path.name).read_bytes()).hexdigest() != manifest["output_hashes"][path.name]:
                raise ValueError(f"Upload hash mismatch: {path.name}")
    if production:
        # Written last: a prefix without _SUCCESS is incomplete and must not be read.
        (output / "_SUCCESS").write_text("")


def main() -> None:
    args = parse_args()
    output = StoragePath(args.output)
    if output.exists():
        raise ValueError("Use a fresh output prefix")
    pool = load_pool(args.pool)
    if not args.test_inputs:
        check_pins(args)
    plain = json.loads((StoragePath(args.plain) / "manifest.json").read_text()) if args.plain else None
    if args.arm in ("rs-multi", "relay", "recovery") and plain is None:
        raise ValueError("--plain is required for rs-multi, relay and recovery")

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        tokenizer = v0_tokenizer(args.tokenizer_dir, directory)
        trace_files: list[dict] = []
        traces = load_traces(args.traces, trace_files)
        gate = token_gate(trace_files)
        kept, dropped = candidates(
            args.arm, traces, pool, tokenizer, require_clean_completion=args.require_clean_completion
        )
        attempts = [candidate.attempt for candidate in kept.values()]
        replay = Replay(args.replay, plain)
        out = directory / "out"
        out.mkdir()
        recovery_stats = None
        if args.arm == "recovery":
            rows, recovery_stats = build_recovery_rows(args, traces, pool, tokenizer, kept, attempts, plain, replay, out)
        else:
            match_relay = (
                json.loads((StoragePath(args.match_relay) / "manifest.json").read_text()) if args.match_relay else None
            )
            rows = build_rows(
                args.arm, kept, attempts, plain, replay, out, args.relay_families, match_relay, args.plain_selection
            )
        index_name = "validation-index.jsonl" if args.arm == "validation" else "train-index.jsonl"
        (out / index_name).write_text("".join(json.dumps(item) + "\n" for item in rows.index))
        (out / "selection.jsonl").write_text("".join(json.dumps(item) + "\n" for item in rows.selection))
        (out / "dropped.jsonl").write_text("".join(json.dumps(item) + "\n" for item in dropped))
        if rows.extra.get("relay_component"):
            (out / COMPONENT_MARKER).write_text("relay component: some families only, no replay; not a trainable arm\n")
        eligible = eligibility(args.arm, traces, pool, attempts)
        manifest = build_manifest(args, pool, kept, dropped, attempts, trace_files, gate, eligible, rows, replay, out)
        if recovery_stats is not None:
            manifest["recovery"] = recovery_stats
        manifest["require_clean_completion"] = args.require_clean_completion
        upload(out, output, manifest, production=not args.test_inputs)
        print(
            "ARM_BUILT " + json.dumps({k: v for k, v in manifest.items() if k not in ("task_slots", "traces")}),
            flush=True,
        )


if __name__ == "__main__":
    main()

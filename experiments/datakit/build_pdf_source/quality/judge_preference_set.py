# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adjudicate pdf-inspector against the VLM on rendered pages, and write the router's label table.

:mod:`~experiments.datakit.build_pdf_source.quality.build_preference_set` writes the packets; this
module buys the verdicts and turns them into the one column the router trains on:
:data:`ESCALATE_COLUMN`, true where a judge looking at the rendered page preferred the VLM's
transcription to pdf-inspector's.

**Rank only. The model's margin is recorded and not used.** A 45-document human-judged subset of the
three-route adjudication set calibrated this judge, and the two halves of that calibration point in
opposite directions. Pairwise agreement with the human is 0.756 overall and tracks the human's own
confidence exactly as a real signal should -- 1.000 where the human called the gap large (n=6),
0.760 small (n=25), 0.643 where the human called the two equivalent (n=14), which is chance where
humans genuinely differ. Aggregate rates match closely and the human ranked pdf-inspector *worse*
than the model does, so the judge is not biased toward the cheap route.

The margin does not survive the same test. The human called 6 of 45 gaps large and 14 equivalent;
the model called 36 large and **zero** equivalent, for a margin agreement of 0.22. A graded label
built on that number would be a graded label of the model's verbosity. So the target is the pairwise
call, and :data:`SECOND_JUDGE` supplies the graded variant instead: on
:data:`~...build_preference_set.SECOND_JUDGE_SIZE` documents a second vendor's model judges the same
packet, and :func:`graded_target` reads agreement between the two as confidence. That measures
something external to either judge, which self-reported margin does not.

**Spend is projected before it is committed.** :func:`pilot` buys a small batch, measures the
realised cost per verdict against this packet shape, and multiplies out. The run refuses to continue
past :data:`MAX_SPEND` or past what the key has left, because the failure mode of a judging pass is
not an error -- it is a bill.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --job-name pdf-preference-judge \\
        --cpu 8 --memory 16GB --disk 16GB --enable-extra-resources \\
        -e OR_KEY_SCALE_UP "$OR_KEY_SCALE_UP" \\
        -- python -m experiments.datakit.build_pdf_source.quality.judge_preference_set
"""

import asyncio
import json
import logging
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

import httpx
import numpy as np
import polars as pl
from rigging.log_setup import configure_logging

from experiments.datakit.build_pdf_source.quality.build_preference_set import (
    BLIND_LABELS,
    KEY_PATH,
    OUTPUT_PREFIX,
    PACKETS_PREFIX,
    Outcome,
)
from experiments.datakit.build_pdf_source.quality.build_route_study import storage

logger = logging.getLogger(__name__)

VERDICT_PREFIX = f"{OUTPUT_PREFIX}/verdicts"
LABELS_PATH = f"{OUTPUT_PREFIX}/labels.parquet"
REPORT_PATH = f"{OUTPUT_PREFIX}/judging_report.json"

JUDGE_MODEL = "openai/gpt-5.6-luna"
# A different vendor's model, so the consistency signal is not two samples of one model's taste.
SECOND_JUDGE = "google/gemini-3.7-flash"
JUDGE_KEY_VAR = "OR_KEY_SCALE_UP"
REASONING_EFFORT = "medium"
BASE_URL = "https://openrouter.ai/api/v1"

# In-flight requests. Latency per verdict is ~40 s and dominated by the judge's own reasoning, so
# throughput is concurrency divided by latency and the ceiling is the key's rate limit rather than
# anything local; this key declares none. Measured at 128 the pass runs ~1.8 hours, which is most of
# a working session for a job whose output everything else waits on.
JUDGE_CONCURRENCY = 256
JUDGE_MAX_ATTEMPTS = 6
JUDGE_TIMEOUT = 300.0
# Reasoning tokens are charged against this ceiling, and a verdict that runs out of budget comes
# back as unparseable JSON rather than as an error. The three-route pass used 4,000 for a
# three-way ranking; a two-way one needs less, but not so much less that the margin is worth
# trimming to nothing.
MAX_COMPLETION_TOKENS = 3000

# Verdicts bought before the spend is projected, and the ceiling the projection is checked against.
# The ceiling is a guard rather than a budget: at the measured $0.0037 per three-route verdict a
# 20,000-document two-route pass lands near $60, and anything far above that means the packet shape
# changed rather than that the pass got bigger.
PILOT_SIZE = 150
MAX_SPEND = 140.0
# Share of the key's remaining credit this pass is allowed to reach for. Below 1.0 so an
# under-estimate does not exhaust a shared key.
MAX_KEY_SHARE = 0.8

# Packets above this are never sent. A packet is the page images inline as base64, and a
# large-format scan can reach 100 MB; measured, the requests that fail deterministically with a 502
# are 51.6 MB and 68.6 MB while p99 of the distribution is 12.6 MB. Spending six attempts and an
# upload apiece on a request that cannot succeed costs minutes and blocks the pilot's projection
# behind it. Skipped packets are simply never judged, which the label table already records as
# `unjudged` rather than treating as an escalation either way.
MAX_PACKET_BYTES = 30_000_000
# Verdicts between progress lines. Small enough that a stalled run is visible within a minute.
PROGRESS_EVERY = 250

# httpx defaults to 100 pooled connections and 20 keep-alive. Left alone, everything above that
# queues *inside* the client waiting for a connection, and the wait counts against the request
# timeout -- so raising JUDGE_CONCURRENCY past 100 buys queueing rather than throughput, and looks
# from the outside like the judge got slower. This repository has hit the same ceiling before, in
# the inference worker's forwarding client. The pool is sized to the concurrency that governs it.
CONNECTION_LIMITS = httpx.Limits(
    max_connections=JUDGE_CONCURRENCY,
    max_keepalive_connections=JUDGE_CONCURRENCY,
)

ESCALATE_COLUMN = "escalate"
CONFIDENCE_COLUMN = "escalate_confidence"

SYSTEM_PROMPT = """You adjudicate PDF text extraction. You are shown one or more rendered pages of a \
PDF and, for each page, two extractions of that same page produced by two different systems, \
labelled Extraction A and Extraction B.

The rendered page image is the ground truth. Judge only how faithfully each extraction reproduces \
what is actually on the page:

- Content completeness: text present on the page that an extraction dropped, and text an extraction \
produced that is not on the page. An extraction that stops partway through the page has lost \
everything after that point.
- Reading order: whether multi-column, sidebar and caption text is serialized in the order a reader \
would follow. Scrambled order is a severe failure even when every word is present.
- Character fidelity: garbled or wrongly-mapped characters, broken diacritics, wrong script, \
mojibake, missing ligatures.
- Structure: whether tables keep their rows and cells, whether headings and lists survive as \
distinct from body text, whether equations are readable.
- Repetition: text repeated far beyond what the page shows is a generation failure, not content.

Explicitly ignore, because they are presentation conventions and not quality:
- Which markup dialect an extraction uses, and whether it uses any.
- Whether tables are drawn with pipes, tags, or plain lines.
- Whitespace, padding, line-wrapping and blank-line differences.
- The markers [table], [formula] and [figure], which are inserted by the packaging and are not \
something a system produced.
- Whether an extraction transcribes text inside charts and diagrams. Systems are under different \
instructions about figures, so figure text is not evidence either way.

You must choose which of the two reproduces the pages better even when they are close, and then say \
separately whether the difference is real. Reply with JSON only, no prose outside it:

{"ranking": ["A","B"], "margin": "large" | "small" | "none", "reason": "<one or two sentences>"}

"ranking" orders the two extractions over the whole document, better first. "margin" is "large" if \
one is clearly better, "small" if the difference is real but minor, and "none" if the two are \
equivalent and the ordering is arbitrary."""


# ---------------------------------------------------------------------------
# One judging task
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Task:
    """One verdict to buy: a packet and the model that judges it."""

    packet_id: str
    model: str

    @property
    def path(self) -> str:
        return f"{VERDICT_PREFIX}/{self.model.replace('/', '_')}/{self.packet_id}.json"


def build_request(fs, packet_id: str) -> list[dict]:
    """The multimodal message for one packet: the page images, then the packet document.

    Images lead so the judge reads the page before it reads anyone's transcription of it, and they
    are already base64 in the packet object, so this is a rearrangement rather than a re-encode.
    """
    with fs.open(f"{PACKETS_PREFIX}/{packet_id}.json", "r") as stream:
        packet = json.load(stream)
    content: list[dict] = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}} for encoded in packet["images"]
    ]
    content.append({"type": "text", "text": packet["markdown"]})
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


def parse_verdict(text: str) -> dict:
    """Read the judge's JSON, tolerating a fenced block around it."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("```")[1]
        stripped = stripped[4:] if stripped.startswith("json") else stripped
    verdict = json.loads(stripped)
    if sorted(verdict["ranking"]) != sorted(BLIND_LABELS):
        raise ValueError(f"ranking is not a permutation of {BLIND_LABELS}: {verdict['ranking']}")
    if verdict["margin"] not in ("large", "small", "none"):
        raise ValueError(f"unknown margin {verdict['margin']}")
    return verdict


def request(fs, task: Task) -> dict:
    """The chat-completions body for one verdict."""
    return {
        "model": task.model,
        "messages": build_request(fs, task.packet_id),
        "max_tokens": MAX_COMPLETION_TOKENS,
        "reasoning": {"effort": REASONING_EFFORT},
    }


async def judge_one(client: httpx.AsyncClient, fs, task: Task, gate: asyncio.Semaphore) -> float | None:
    """Buy one verdict, write it, and return what it cost, or ``None`` if it never came back.

    The verdict is written here rather than collected and written by the caller. Twenty thousand
    verdicts is hours of wall time, and a pass that only persists at batch boundaries loses
    everything in flight when it is interrupted -- and shows no progress in between, which makes a
    stalled run indistinguishable from a slow one.
    """
    payload = None
    for attempt in range(JUDGE_MAX_ATTEMPTS):
        async with gate:
            # Built inside the gate, and off the event loop. A packet is ~1.5 MB of base64 page
            # images; read directly from the coroutine that sends the request, the blocking fetch
            # stalls every other in-flight request for its duration, and the loop then spends
            # longer moving bytes than waiting on the judge. Kept across retries so a 502 does not
            # buy the same megabyte and a half again.
            if payload is None:
                payload = await asyncio.to_thread(request, fs, task)
            try:
                response = await client.post("/chat/completions", json=payload, timeout=JUDGE_TIMEOUT)
                response.raise_for_status()
                body = response.json()
                verdict = parse_verdict(body["choices"][0]["message"]["content"])
            except Exception as error:
                failed = error
            else:
                failed = None
        if failed is not None:
            # Backed off *outside* the gate. Holding a concurrency slot while sleeping turns a burst
            # of upstream 502s into a throughput collapse: the slots fill with coroutines that are
            # waiting rather than working, and nothing else can start.
            # `repr`, not `str`: httpx's timeout and protocol errors carry no message, and an
            # empty one in the log makes a real failure mode look like a blank line.
            logger.info("%s attempt %d: %r", task.packet_id, attempt + 1, failed)
            await asyncio.sleep(min(2**attempt, 30))
            continue
        result = {
            "packet_id": task.packet_id,
            "model": task.model,
            "verdict": verdict,
            "cost": body.get("usage", {}).get("cost"),
        }
        await asyncio.to_thread(write_verdict, fs, task, result)
        return result["cost"] or 0.0
    logger.warning("%s: no verdict after %d attempts", task.packet_id, JUDGE_MAX_ATTEMPTS)
    return None


def bought_packets(fs, model: str) -> set[str]:
    """Packet ids this model already has a verdict for, from one prefix listing."""
    slot = f"{VERDICT_PREFIX}/{model.replace('/', '_')}"
    return {path.rsplit("/", 1)[-1].removesuffix(".json") for path in fs.glob(f"{slot}/*.json")}


def write_verdict(fs, task: Task, result: dict) -> None:
    with fs.open(task.path, "w") as stream:
        json.dump(result, stream)


async def run_batch(client: httpx.AsyncClient, fs, tasks: list[Task]) -> float:
    """Buy every verdict in *tasks*, bounded by the semaphore, reporting as they land.

    One gather over the whole list rather than a loop over batches: the semaphore already bounds
    what is in flight, so batching adds nothing but a barrier that idles the tail of every batch
    while its slowest request finishes.
    """
    gate = asyncio.Semaphore(JUDGE_CONCURRENCY)
    spent = 0.0
    done = 0
    started = time.monotonic()
    pending = [asyncio.create_task(judge_one(client, fs, task, gate)) for task in tasks]
    for completed in asyncio.as_completed(pending):
        cost = await completed
        done += 1
        spent += cost or 0.0
        if done % PROGRESS_EVERY == 0 or done == len(tasks):
            rate = done / max(time.monotonic() - started, 1e-9)
            logger.info(
                "judged %d/%d, spent $%.4f, %.1f verdicts/s, eta %.0f min",
                done,
                len(tasks),
                spent,
                rate,
                (len(tasks) - done) / max(rate, 1e-9) / 60,
            )
    return spent


async def key_credit(client: httpx.AsyncClient) -> float:
    """What the key has left to spend, so a projection can be checked against it."""
    response = await client.get("/key", timeout=60.0)
    response.raise_for_status()
    data = response.json()["data"]
    remaining = data.get("limit_remaining")
    return float("inf") if remaining is None else float(remaining)


async def buy(fs, tasks: list[Task]) -> dict:
    """Buy every verdict not already on storage, after projecting what the rest will cost.

    The pilot is not a dry run -- its verdicts are kept. It exists so the projection is made against
    this packet shape and this judge rather than against a remembered number from a different pass.
    """
    # One listing per model rather than a HEAD per task. At twenty thousand tasks the sequential
    # existence check is minutes of the run on its own, and it has to stay per model: the same
    # packet is judged by two of them and their verdicts live under different prefixes.
    bought = {model: bought_packets(fs, model) for model in {task.model for task in tasks}}
    pending = [task for task in tasks if task.packet_id not in bought[task.model]]
    oversized = {
        entry["name"].rsplit("/", 1)[-1].removesuffix(".json")
        for entry in fs.ls(PACKETS_PREFIX, detail=True)
        if entry["size"] > MAX_PACKET_BYTES
    }
    skipped = [task for task in pending if task.packet_id in oversized]
    pending = [task for task in pending if task.packet_id not in oversized]
    logger.info(
        "judging: %d tasks, %d already bought, %d skipped as larger than %.0f MB",
        len(tasks),
        len(tasks) - len(pending) - len(skipped),
        len(skipped),
        MAX_PACKET_BYTES / 1e6,
    )
    if not pending:
        return {"pilot_verdicts": 0, "projected": 0.0, "spent": 0.0, "oversized_skipped": len(skipped)}

    headers = {"Authorization": f"Bearer {os.environ[JUDGE_KEY_VAR]}"}
    async with httpx.AsyncClient(base_url=BASE_URL, headers=headers, limits=CONNECTION_LIMITS) as client:
        credit = await key_credit(client)
        logger.info("key credit remaining: $%.2f", credit)

        pilot_tasks = pending[:PILOT_SIZE]
        pilot_spend = await run_batch(client, fs, pilot_tasks)
        per_verdict = pilot_spend / max(len(pilot_tasks), 1)
        projected = per_verdict * len(pending)
        logger.info(
            "pilot: %d verdicts for $%.4f ($%.5f each); projected total $%.2f for %d verdicts",
            len(pilot_tasks),
            pilot_spend,
            per_verdict,
            projected,
            len(pending),
        )
        allowed = min(MAX_SPEND, credit * MAX_KEY_SHARE)
        if projected > allowed:
            raise RuntimeError(
                f"projected ${projected:.2f} exceeds the ${allowed:.2f} this pass may spend "
                f"(cap ${MAX_SPEND:.2f}, key credit ${credit:.2f}); nothing past the pilot was bought"
            )
        rest = await run_batch(client, fs, pending[PILOT_SIZE:])

    return {
        "pilot_verdicts": len(pilot_tasks),
        "cost_per_verdict": per_verdict,
        "projected": projected,
        "spent": pilot_spend + rest,
        "oversized_skipped": len(skipped),
        "max_packet_bytes": MAX_PACKET_BYTES,
    }


# ---------------------------------------------------------------------------
# Reading the verdicts out
# ---------------------------------------------------------------------------


def prefers_vlm(entry: dict, verdict: dict) -> bool:
    """Whether the judge put the VLM's transcription ahead of pdf-inspector's."""
    ranking = [entry["labels"][blind] for blind in verdict["ranking"]]
    return ranking.index("vlm") < ranking.index("inspector")


def graded_target(primary: bool | None, second: bool | None) -> float | None:
    """Confidence that escalating is right, from agreement between two judges rather than a margin.

    Two models from different vendors seeing the same packet and reaching the same call is external
    evidence; one model saying "large" is not, and the human calibration measured that self-reported
    margin agrees with a person 0.22 of the time. A document only one judge saw keeps its binary
    call at full weight, because a single competent verdict is still the best estimate available --
    it just carries no consistency evidence either way.
    """
    if primary is None:
        return None
    if second is None:
        return float(primary)
    if primary == second:
        return float(primary)
    return 0.5


def load_verdicts(fs, model: str) -> dict[str, dict]:
    paths = fs.glob(f"{VERDICT_PREFIX}/{model.replace('/', '_')}/*.json")
    verdicts = {}
    for path in paths:
        with fs.open(path, "r") as stream:
            result = json.load(stream)
        verdicts[result["packet_id"]] = result
    return verdicts


def label_table(entries: list[dict], primary: dict[str, dict], second: dict[str, dict]) -> pl.DataFrame:
    """One row per drawn document: the escalation label and everything needed to split on it.

    Documents decided without a judge are rows here too. A route that produced nothing has lost the
    document, and that is a routing decision with a known right answer -- pdf-inspector failing
    means escalate, the VLM failing means do not. Leaving them out would train a router that has
    never seen either failure.
    """
    rows = []
    for entry in entries:
        packet_id = entry["packet_id"]
        outcome = entry["outcome"]
        if outcome == str(Outcome.INSPECTOR_FAILED):
            escalate, source = True, "inspector_failed"
        elif outcome in (str(Outcome.VLM_FAILED), str(Outcome.NO_PAGES)):
            escalate, source = False, outcome
        elif packet_id in primary:
            escalate, source = prefers_vlm(entry, primary[packet_id]["verdict"]), "judged"
        else:
            escalate, source = None, "unjudged"

        second_call = (
            prefers_vlm(entry, second[packet_id]["verdict"]) if packet_id in second and source == "judged" else None
        )
        rows.append(
            {
                "source_id": entry["source_id"],
                "packet_id": packet_id,
                "domain": entry["domain"],
                "stratum": entry["stratum"],
                "trustworthy": entry["trustworthy"],
                "label_source": source,
                ESCALATE_COLUMN: escalate,
                CONFIDENCE_COLUMN: graded_target(escalate, second_call),
                "second_judge_escalate": second_call,
                "margin": primary[packet_id]["verdict"]["margin"] if packet_id in primary else None,
            }
        )
    return pl.DataFrame(rows, schema_overrides={ESCALATE_COLUMN: pl.Boolean, "second_judge_escalate": pl.Boolean})


# Words that mark a verdict decided on something the prompt asks the judge to ignore. Figure and
# chart text is the one that matters: the two systems are under different instructions about what to
# do with axis labels, so a verdict resting on them is measuring a policy difference rather than a
# quality difference. Spot checks found the judge citing tick labels despite the instruction, so the
# rate is measured instead of assumed away.
IGNORED_EVIDENCE = ("chart", "axis", "axes", "tick label", "figure text", "diagram")


def prompt_compliance(verdicts: dict[str, dict]) -> dict:
    """How often a verdict's stated reason rests on evidence the prompt excluded.

    A rate rather than a filter: these verdicts are kept, because the reason field is a summary and
    not the whole basis of the call, and dropping them would select on the judge's explanation
    style. What the number does is bound how much of the label could be a figure-policy artifact.
    """
    reasons = [(result["verdict"].get("reason") or "").lower() for result in verdicts.values()]
    flagged = sum(any(word in reason for word in IGNORED_EVIDENCE) for reason in reasons)
    return {"verdicts": len(reasons), "cite_ignored_evidence": flagged, "rate": flagged / max(len(reasons), 1)}


def _wilson(successes: int, total: int) -> tuple[float, float]:
    """A 95% Wilson interval, which is what a per-stratum ``n`` needs instead of a bare proportion."""
    if total == 0:
        return (float("nan"), float("nan"))
    z, phat = 1.96, successes / total
    denominator = 1 + z**2 / total
    centre = (phat + z**2 / (2 * total)) / denominator
    spread = z * np.sqrt(phat * (1 - phat) / total + z**2 / (4 * total**2)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def rates_by(labels: pl.DataFrame, key: str) -> dict[str, dict]:
    """Escalation rate per group, with its domain count beside its document count.

    Near-duplicates cluster by publisher, so domains are the independent unit and a confidence
    interval computed on the document count is the optimistic bound.
    """
    known = labels.filter(pl.col(ESCALATE_COLUMN).is_not_null())
    output = {}
    for (name,), group in known.group_by(key, maintain_order=False):
        successes = int(group[ESCALATE_COLUMN].sum())
        low, high = _wilson(successes, group.height)
        output[str(name)] = {
            "documents": group.height,
            "domains": group["domain"].n_unique(),
            "escalate_rate": successes / group.height,
            "escalate_rate_ci95": [low, high],
        }
    return dict(sorted(output.items()))


def judge_agreement(labels: pl.DataFrame) -> dict:
    """How often the two judges made the same call on the documents both saw."""
    both = labels.filter(pl.col("second_judge_escalate").is_not_null())
    if both.height == 0:
        return {"documents": 0}
    same = int((both[ESCALATE_COLUMN] == both["second_judge_escalate"]).sum())
    return {
        "documents": both.height,
        "domains": both["domain"].n_unique(),
        "agreement": same / both.height,
        # Chance agreement on a binary call is 1/2.
        "kappa": (same / both.height - 0.5) / 0.5,
        "primary_escalate_rate": float(both[ESCALATE_COLUMN].mean()),
        "second_escalate_rate": float(both["second_judge_escalate"].mean()),
    }


def margin_distribution(labels: pl.DataFrame) -> dict:
    """What the model claims about its own confidence, recorded so the miscalibration stays visible.

    The human calibration put margin agreement at 0.22, so this is reported and never trained on.
    """
    counts = Counter(labels.filter(pl.col("margin").is_not_null())["margin"].to_list())
    total = max(sum(counts.values()), 1)
    return {name: {"documents": count, "share": count / total} for name, count in sorted(counts.items())}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    configure_logging(logging.INFO)
    if JUDGE_KEY_VAR not in os.environ:
        raise RuntimeError(f"{JUDGE_KEY_VAR} is not set; the judge cannot run without it")
    fs = storage()

    with fs.open(KEY_PATH, "r") as stream:
        entries = json.load(stream)
    judged = [entry for entry in entries if entry["outcome"] == str(Outcome.JUDGED)]
    logger.info(
        "key: %d drawn, %d packets to judge, %d decided without a judge",
        len(entries),
        len(judged),
        len(entries) - len(judged),
    )

    tasks = [Task(entry["packet_id"], JUDGE_MODEL) for entry in judged]
    tasks += [Task(entry["packet_id"], SECOND_JUDGE) for entry in judged if entry["second_judge"]]
    spend = asyncio.run(buy(fs, tasks))

    primary = load_verdicts(fs, JUDGE_MODEL)
    second = load_verdicts(fs, SECOND_JUDGE)
    logger.info("verdicts: %d primary, %d second judge", len(primary), len(second))

    labels = label_table(entries, primary, second)
    with fs.open(LABELS_PATH, "wb") as stream:
        labels.write_parquet(stream)

    known = labels.filter(pl.col(ESCALATE_COLUMN).is_not_null())
    by_source: dict[str, int] = defaultdict(int)
    for name in labels["label_source"].to_list():
        by_source[name] += 1
    report = {
        "judge": JUDGE_MODEL,
        "second_judge": SECOND_JUDGE,
        "documents": labels.height,
        "labelled": known.height,
        "domains": known["domain"].n_unique(),
        "label_sources": dict(sorted(by_source.items())),
        "escalate_rate": float(known[ESCALATE_COLUMN].mean()),
        "escalate_rate_judged_only": float(known.filter(pl.col("label_source") == "judged")[ESCALATE_COLUMN].mean()),
        "by_stratum": rates_by(known, "stratum"),
        "by_trustworthy": rates_by(known, "trustworthy"),
        "inter_judge": judge_agreement(labels),
        "margins": margin_distribution(labels),
        "prompt_compliance": prompt_compliance(primary),
        "spend": spend,
    }
    with fs.open(REPORT_PATH, "w") as stream:
        json.dump(report, stream, indent=2, default=float)
    print(json.dumps(report, indent=2, default=float))
    logger.info("wrote %s and %s", LABELS_PATH, REPORT_PATH)


if __name__ == "__main__":
    main()

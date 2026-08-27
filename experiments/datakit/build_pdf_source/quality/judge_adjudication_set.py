# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adjudicate the three-route packets against the rendered pages, and read the verdicts out.

:mod:`~experiments.datakit.build_pdf_source.quality.build_adjudication_set` writes the packets; this
module judges them and reports what they say. The ground truth is the page image, which is the whole
point: every number Stages 1-2 produced is *agreement with the VLM*, and agreement cannot rank two
cheap routes against each other when one of them shares the VLM's dialect. Here the VLM is one of
the three things being judged rather than the thing everything is measured against.

**The judge is forced to choose, and asked separately whether the choice mattered.** A judge that
answers "equivalent" most of the time has measured nothing, and the obvious fix -- forbidding ties --
manufactures signal out of noise instead. So every verdict carries a strict ranking of the three
extractions *and* a ``margin`` saying whether the gap is real. The ranking supplies the power; the
margin supplies the honesty, and :data:`DECISIVE_MARGINS` is what the decisive-verdict rate counts.

Three judging passes, all over the same packets:

``canonical``
    Every packet, dialect-neutral presentation. This is the pass the headline verdict comes from.
``native``
    The style-control packets only, each route in its own serialization. Paired with the same
    document's canonical verdict, so the difference between the two *is* the style effect -- the
    documents, the pages, the blinding and the label assignment are all held fixed and only the
    presentation moves. Re-measured rather than carried over: 1.17.0 recovers tables the previous
    build did not, and a dialect effect measured on a dialect that changed is not a measurement.
``second judge``
    The style-control packets again under :data:`SECOND_JUDGE`, canonical presentation, so the
    headline has an inter-judge agreement number attached rather than resting on one model's taste.

**Results are reported per :class:`~...build_adjudication_set.Arm` and are never pooled.** The
paired arm re-judges the documents the 1.14.1 pass judged, with the pages and the blinding frozen,
so its statistic is a *change*: :func:`paired_shift` counts the documents whose inspector-versus-
Docling call flipped in each direction and tests them with McNemar's exact form, which conditions
on the discordant pairs and throws away nothing that the pairing bought. The extension arm is fresh
documents in the strata the previous allocation under-served; it has no 1.14.1 verdict to be paired
against, so it contributes width to the per-stratum intervals and nothing to the before-and-after.
One number spanning both would average a paired contrast against an unpaired estimate.

The judge is ``openai/gpt-5.6-luna`` at medium reasoning effort over OpenRouter, keyed by
``OR_KEY_SCALE_UP``, following the oracle sample's precedent. Every verdict is written to its own
object and an existing one is never re-bought, so an interrupted driver resumes where it stopped.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --job-name pdf-adjudication-judge \\
        --cpu 8 --memory 16GB --disk 16GB --enable-extra-resources \\
        -e OR_KEY_SCALE_UP "$OR_KEY_SCALE_UP" \\
        -- python -m experiments.datakit.build_pdf_source.quality.judge_adjudication_set
"""

import asyncio
import base64
import json
import logging
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import pairwise
from math import comb

import httpx
import numpy as np
from rigging.log_setup import configure_logging

from experiments.datakit.build_pdf_source.quality.build_adjudication_set import (
    BASELINE_KEY_PATH,
    BASELINE_PREFIX,
    BLIND_LABELS,
    KEY_PATH,
    LIBRARY_VERSION,
    OUTPUT_PREFIX,
    PACKETS_PREFIX,
    ROUTES,
    Arm,
    Presentation,
)
from experiments.datakit.build_pdf_source.quality.build_route_study import storage

logger = logging.getLogger(__name__)

VERDICT_PREFIX = f"{OUTPUT_PREFIX}/verdicts"
REPORT_PATH = f"{OUTPUT_PREFIX}/adjudication_report.json"
HUMAN_PACKET_PREFIX = f"{OUTPUT_PREFIX}/human_subset"

# The 1.14.1 verdicts, read but never rewritten: they are the other half of the paired head-to-head.
BASELINE_VERDICT_PREFIX = f"{BASELINE_PREFIX}/verdicts"

JUDGE_MODEL = "openai/gpt-5.6-luna"
# A different vendor's model, so inter-judge agreement is not two samples of one model's taste.
SECOND_JUDGE = "google/gemini-3.7-flash"
JUDGE_KEY_VAR = "OR_KEY_SCALE_UP"
REASONING_EFFORT = "medium"

JUDGE_CONCURRENCY = 48
JUDGE_MAX_ATTEMPTS = 6
JUDGE_TIMEOUT = 300.0
MAX_COMPLETION_TOKENS = 4000

# Margins that count as a decisive verdict. "none" means the judge ranked the three but says the
# ordering is arbitrary, which is a tie reported honestly rather than a preference.
DECISIVE_MARGINS = ("large", "small")

# Bigram-recall buckets the proxy label is validated over, matching the published pass's reading
# that the metric validates below 0.2 and that Docling still wins throughout.
RECALL_BUCKETS = (0.0, 0.2, 0.5, 0.8, 1.01)

SYSTEM_PROMPT = """You adjudicate PDF text extraction. You are shown one or more rendered pages of a \
PDF and, for each page, three extractions of that same page produced by three different systems, \
labelled Extraction A, Extraction B and Extraction C.

The rendered page image is the ground truth. Judge only how faithfully each extraction reproduces \
what is actually on the page:

- Content completeness: text present on the page that an extraction dropped, and text an extraction \
produced that is not on the page.
- Reading order: whether multi-column, sidebar and caption text is serialized in the order a reader \
would follow. Scrambled order is a severe failure even when every word is present.
- Character fidelity: garbled or wrongly-mapped characters, broken diacritics, wrong script, \
mojibake, missing ligatures.
- Structure: whether tables keep their rows and cells, whether headings and lists survive as \
distinct from body text, whether equations are readable.

Explicitly ignore, because they are presentation conventions and not quality:
- Which markup dialect an extraction uses, and whether it uses any.
- Whether tables are drawn with pipes, tags, or plain lines.
- Whitespace, padding, line-wrapping and blank-line differences.
- The markers [table], [formula] and [figure], which are inserted by the packaging and are not \
something a system produced.
- Whether an extraction transcribes text inside charts and diagrams. Systems are under different \
instructions about figures, so figure text is not evidence either way.

You must produce a strict ranking of all three even when they are close, and then say separately \
whether the difference is real. Reply with JSON only, no prose outside it:

{"pages": [{"page": <int>, "ranking": ["A","B","C"], "worst_error": "<short phrase or null>"}],
 "ranking": ["A","B","C"],
 "margin": "large" | "small" | "none",
 "reason": "<one or two sentences>"}

"ranking" at the top level orders the three extractions over the whole document, best first. \
"margin" is "large" if the best is clearly better than the worst, "small" if the difference is \
real but minor, and "none" if the three are equivalent and the ranking is arbitrary."""


# ---------------------------------------------------------------------------
# One judging task
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Task:
    """One verdict to buy: a packet, a presentation, and the model that judges it."""

    packet_id: str
    presentation: Presentation
    model: str

    @property
    def slot(self) -> str:
        return f"{self.model.replace('/', '_')}/{self.presentation}"

    @property
    def path(self) -> str:
        return f"{VERDICT_PREFIX}/{self.slot}/{self.packet_id}.json"


def _image_part(payload: bytes) -> dict:
    encoded = base64.b64encode(payload).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}


def build_request(fs, entry: dict, task: Task) -> list[dict]:
    """The multimodal message for one packet: every chosen page image, then the packet document.

    Images lead so the judge reads the page before it reads anyone's transcription of it.
    """
    content: list[dict] = []
    for page in entry["pages"]:
        with fs.open(f"{PACKETS_PREFIX}/{task.packet_id}/{page['image']}", "rb") as stream:
            content.append(_image_part(stream.read()))
    with fs.open(f"{PACKETS_PREFIX}/{task.packet_id}/document_{task.presentation}.md", "r") as stream:
        document = stream.read()
    content.append({"type": "text", "text": document})
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


def parse_verdict(text: str) -> dict:
    """Read the judge's JSON, tolerating a fenced block around it."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("```")[1]
        stripped = stripped[4:] if stripped.startswith("json") else stripped
    verdict = json.loads(stripped)
    ranking = verdict["ranking"]
    if sorted(ranking) != sorted(BLIND_LABELS):
        raise ValueError(f"ranking is not a permutation of {BLIND_LABELS}: {ranking}")
    if verdict["margin"] not in (*DECISIVE_MARGINS, "none"):
        raise ValueError(f"unknown margin {verdict['margin']}")
    return verdict


async def judge_one(client: httpx.AsyncClient, fs, entry: dict, task: Task, gate: asyncio.Semaphore) -> dict | None:
    """Buy one verdict, or return ``None`` if the judge never produced a usable one."""
    messages = build_request(fs, entry, task)
    payload = {
        "model": task.model,
        "messages": messages,
        "max_tokens": MAX_COMPLETION_TOKENS,
        "reasoning": {"effort": REASONING_EFFORT},
    }
    for attempt in range(JUDGE_MAX_ATTEMPTS):
        async with gate:
            try:
                response = await client.post("/chat/completions", json=payload, timeout=JUDGE_TIMEOUT)
                response.raise_for_status()
                body = response.json()
                verdict = parse_verdict(body["choices"][0]["message"]["content"])
            except Exception as error:
                logger.info("%s attempt %d: %s", task.packet_id, attempt + 1, error)
                await asyncio.sleep(min(2**attempt, 30))
                continue
        return {
            "packet_id": task.packet_id,
            "presentation": str(task.presentation),
            "model": task.model,
            "verdict": verdict,
            "cost": body.get("usage", {}).get("cost"),
            "usage": body.get("usage"),
        }
    logger.warning("%s: no verdict after %d attempts", task.packet_id, JUDGE_MAX_ATTEMPTS)
    return None


async def run_tasks(fs, entries: dict[str, dict], tasks: list[Task]) -> None:
    """Buy every verdict not already on storage."""
    pending = [task for task in tasks if not fs.exists(task.path)]
    logger.info("judging: %d tasks, %d already bought", len(tasks), len(tasks) - len(pending))
    if not pending:
        return
    gate = asyncio.Semaphore(JUDGE_CONCURRENCY)
    headers = {"Authorization": f"Bearer {os.environ[JUDGE_KEY_VAR]}"}
    spent = 0.0
    async with httpx.AsyncClient(base_url="https://openrouter.ai/api/v1", headers=headers) as client:
        for start in range(0, len(pending), JUDGE_CONCURRENCY * 4):
            batch = pending[start : start + JUDGE_CONCURRENCY * 4]
            results = await asyncio.gather(
                *(judge_one(client, fs, entries[task.packet_id], task, gate) for task in batch)
            )
            for task, result in zip(batch, results, strict=True):
                if result is None:
                    continue
                spent += result.get("cost") or 0.0
                with fs.open(task.path, "w") as stream:
                    json.dump(result, stream)
            logger.info("judged %d/%d, spent $%.4f", min(start + len(batch), len(pending)), len(pending), spent)
    logger.info("judging complete, spent $%.4f", spent)


# ---------------------------------------------------------------------------
# Reading the verdicts out
# ---------------------------------------------------------------------------


def unblind(entry: dict, verdict: dict) -> list[str]:
    """The judge's ranking, translated from shown labels back into route names."""
    return [entry["labels"][label] for label in verdict["ranking"]]


@dataclass
class Tally:
    """Verdicts over one slice of documents."""

    documents: int = 0
    decisive: int = 0
    firsts: Counter = field(default_factory=Counter)
    lasts: Counter = field(default_factory=Counter)
    inspector_over_docling: int = 0
    inspector_over_docling_decisive: int = 0
    pairs: int = 0
    pairs_decisive: int = 0

    def add(self, ranking: list[str], decisive: bool) -> None:
        self.documents += 1
        self.decisive += decisive
        self.firsts[ranking[0]] += 1
        self.lasts[ranking[-1]] += 1
        if "inspector" in ranking and "docling" in ranking:
            self.pairs += 1
            ahead = ranking.index("inspector") < ranking.index("docling")
            self.inspector_over_docling += ahead
            if decisive:
                self.pairs_decisive += 1
                self.inspector_over_docling_decisive += ahead

    def summary(self) -> dict:
        return {
            "documents": self.documents,
            "decisive_rate": self.decisive / max(self.documents, 1),
            "win_rate": {route: self.firsts[route] / max(self.documents, 1) for route in ROUTES},
            "last_rate": {route: self.lasts[route] / max(self.documents, 1) for route in ROUTES},
            "inspector_over_docling": self.inspector_over_docling / max(self.pairs, 1),
            "inspector_over_docling_decisive": self.inspector_over_docling_decisive / max(self.pairs_decisive, 1),
            "decisive_pairs": self.pairs_decisive,
        }


def _wilson(successes: int, total: int) -> tuple[float, float]:
    """A 95% Wilson interval, which is what an ``n`` of 40 needs instead of a bare proportion."""
    if total == 0:
        return (float("nan"), float("nan"))
    z, phat = 1.96, successes / total
    denominator = 1 + z**2 / total
    centre = (phat + z**2 / (2 * total)) / denominator
    spread = z * np.sqrt(phat * (1 - phat) / total + z**2 / (4 * total**2)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def tally_by(entries: dict, verdicts: dict[str, dict], key: str) -> dict[str, dict]:
    """Verdicts grouped by one field of the key entry.

    Each group reports its domain count beside its document count. Near-duplicates cluster by
    publisher, so domains are the independent unit: RTL's 55 documents come from ~30 domains, and a
    confidence interval computed on 55 would overstate what the stratum can support. The interval on
    the document count is reported as the optimistic bound and the domain count sits next to it as
    the number to believe.
    """
    grouped: dict[str, Tally] = defaultdict(Tally)
    domains: dict[str, set[str]] = defaultdict(set)
    for packet_id, result in verdicts.items():
        entry = entries[packet_id]
        verdict = result["verdict"]
        name = str(entry[key])
        grouped[name].add(unblind(entry, verdict), verdict["margin"] in DECISIVE_MARGINS)
        domains[name].add(entry.get("domain", ""))
    output = {}
    for name, tally in sorted(grouped.items()):
        summary = tally.summary()
        low, high = _wilson(tally.inspector_over_docling, tally.pairs)
        summary["inspector_over_docling_ci95"] = [low, high]
        summary["domains"] = len(domains[name])
        output[name] = summary
    return output


def proxy_label_check(entries: dict, verdicts: dict[str, dict]) -> dict:
    """Does ``inspector_ok`` track the verdict, and where does it invert?

    Two readings, because the label is a claim about two different things. Bucketed by pdf-inspector's
    own bigram recall against the VLM, it says whether the metric orders documents the way a judge
    does. Split by the boolean label itself, it says whether the threshold is in the right place.
    """
    buckets: dict[str, Tally] = defaultdict(Tally)
    by_label: dict[str, Tally] = defaultdict(Tally)
    for packet_id, result in verdicts.items():
        entry = entries[packet_id]
        metrics = entry["document_metrics"]
        recall = metrics["inspector_vlm_bigram_recall_mean"]
        ranking = unblind(entry, result["verdict"])
        decisive = result["verdict"]["margin"] in DECISIVE_MARGINS
        if recall is not None:
            for low, high in pairwise(RECALL_BUCKETS):
                if low <= recall < high:
                    buckets[f"[{low:.1f},{high:.1f})"].add(ranking, decisive)
                    break
        by_label[f"inspector_ok={metrics['inspector_ok']}"].add(ranking, decisive)
        by_label[f"docling_ok={metrics['docling_ok']}"].add(ranking, decisive)
    return {
        "by_inspector_recall_bucket": {name: tally.summary() for name, tally in sorted(buckets.items())},
        "by_label": {name: tally.summary() for name, tally in sorted(by_label.items())},
    }


def style_effect(entries: dict, canonical: dict[str, dict], native: dict[str, dict]) -> dict:
    """What changes when the dialects are put back, on the documents judged both ways.

    Paired: the same packet, the same pages, the same blinding. A verdict that moves between the two
    arms moved because of presentation, which is the only thing that differs.
    """
    shared = sorted(set(canonical) & set(native))
    if not shared:
        return {"documents": 0}
    agree = flips = 0
    canonical_ahead = native_ahead = 0
    for packet_id in shared:
        entry = entries[packet_id]
        one = unblind(entry, canonical[packet_id]["verdict"])
        two = unblind(entry, native[packet_id]["verdict"])
        agree += one[0] == two[0]
        if "inspector" in one and "docling" in one:
            first = one.index("inspector") < one.index("docling")
            second = two.index("inspector") < two.index("docling")
            canonical_ahead += first
            native_ahead += second
            flips += first != second
    return {
        "documents": len(shared),
        "same_winner_rate": agree / len(shared),
        "inspector_over_docling_canonical": canonical_ahead / len(shared),
        "inspector_over_docling_native": native_ahead / len(shared),
        "style_effect": (native_ahead - canonical_ahead) / len(shared),
        "pairwise_flip_rate": flips / len(shared),
    }


def inter_judge(entries: dict, primary: dict[str, dict], second: dict[str, dict]) -> dict:
    """Agreement between the two judges on the documents both saw."""
    shared = sorted(set(primary) & set(second))
    if not shared:
        return {"documents": 0}
    same_winner = same_pair = 0
    for packet_id in shared:
        entry = entries[packet_id]
        one = unblind(entry, primary[packet_id]["verdict"])
        two = unblind(entry, second[packet_id]["verdict"])
        same_winner += one[0] == two[0]
        same_pair += (one.index("inspector") < one.index("docling")) == (two.index("inspector") < two.index("docling"))
    # Chance agreement on a three-way winner is 1/3; on a pairwise call it is 1/2.
    winner_rate, pair_rate = same_winner / len(shared), same_pair / len(shared)
    return {
        "documents": len(shared),
        "same_winner_rate": winner_rate,
        "same_pairwise_call_rate": pair_rate,
        "winner_kappa": (winner_rate - 1 / 3) / (1 - 1 / 3),
        "pairwise_kappa": (pair_rate - 0.5) / 0.5,
    }


def load_verdicts(fs, model: str, presentation: Presentation, prefix: str = VERDICT_PREFIX) -> dict[str, dict]:
    slot = f"{model.replace('/', '_')}/{presentation}"
    verdicts = {}
    for path in fs.glob(f"{prefix}/{slot}/*.json"):
        with fs.open(path, "r") as stream:
            result = json.load(stream)
        verdicts[result["packet_id"]] = result
    return verdicts


def _mcnemar(discordant_up: int, discordant_down: int) -> float:
    """Two-sided exact binomial p for a paired change, conditioned on the discordant pairs.

    The right test for a before-and-after on the *same* documents: the packets whose verdict did not
    move carry no information about whether it moved, and pooling them into two independent
    proportions throws the pairing away -- which is the one thing worth having here, since the
    published pass measured split-draw noise larger than several of the effects at issue.
    """
    total = discordant_up + discordant_down
    if total == 0:
        return 1.0
    smaller = min(discordant_up, discordant_down)
    tail = sum(comb(total, k) for k in range(smaller + 1)) / 2**total
    return min(1.0, 2 * tail)


def paired_shift(
    baseline_entries: dict[str, dict],
    baseline: dict[str, dict],
    entries: dict[str, dict],
    candidate: dict[str, dict],
    key: str | None = None,
) -> dict:
    """How the head-to-head moved on the documents judged under both builds.

    Each side is unblinded with **its own** key. The packets are built to carry identical blinding,
    but reading the label mapping from the key that belongs to each verdict is what makes that a
    checkable property rather than an assumption -- and a silent mismatch here would invert results
    rather than break them.

    ``inspector_over_docling`` is the pairwise call the report leads on, so the paired statistic is
    the count of documents whose call flipped in each direction, tested with McNemar's exact form.
    """
    shared = sorted(set(baseline) & set(candidate) & set(baseline_entries) & set(entries))
    grouped: dict[str, list[str]] = defaultdict(list)
    for packet_id in shared:
        grouped["ALL"].append(packet_id)
        if key is not None:
            grouped[str(entries[packet_id][key])].append(packet_id)

    output = {}
    for name, packets in sorted(grouped.items()):
        gained = lost = held = 0
        wins_before = wins_after = 0
        domains = set()
        for packet_id in packets:
            before = unblind(baseline_entries[packet_id], baseline[packet_id]["verdict"])
            after = unblind(entries[packet_id], candidate[packet_id]["verdict"])
            ahead_before = before.index("inspector") < before.index("docling")
            ahead_after = after.index("inspector") < after.index("docling")
            wins_before += ahead_before
            wins_after += ahead_after
            gained += ahead_after and not ahead_before
            lost += ahead_before and not ahead_after
            held += ahead_after == ahead_before
            domains.add(entries[packet_id].get("domain", ""))
        count = len(packets)
        output[name] = {
            "documents": count,
            "domains": len(domains),
            "inspector_over_docling_before": wins_before / max(count, 1),
            "inspector_over_docling_after": wins_after / max(count, 1),
            "delta": (wins_after - wins_before) / max(count, 1),
            "flipped_to_inspector": gained,
            "flipped_to_docling": lost,
            "unchanged": held,
            "mcnemar_p": _mcnemar(gained, lost),
        }
    return output


# ---------------------------------------------------------------------------
# Human subset
# ---------------------------------------------------------------------------

HUMAN_INSTRUCTIONS = """# Human adjudication subset

{count} documents drawn from the stratified adjudication set. Each `doc_XXXX/` directory holds the
rendered pages of one PDF and three extractions of those same pages, labelled Extraction A, B and C.
The label-to-system mapping is deliberately not in this directory; it is in `key.json` next to it,
and the point of the exercise is lost if you read that first.

For each document, open the page images alongside `document_canonical.md` and judge only how
faithfully each extraction reproduces what is on the page: content completeness, reading order,
character fidelity, and whether tables, headings and equations survive. Ignore markup dialect,
whitespace, and the `[table]`/`[formula]`/`[figure]` markers, which the packaging inserted.

Record one row per document in `verdicts.csv`:

    packet_id,ranking,margin,reason

`ranking` is the three labels best-first with no separator, for example `BAC`. `margin` is `large`,
`small`, or `none` -- `none` means the three are equivalent and your ranking was arbitrary. Always
give a full ranking even when you record `margin=none`; the ranking supplies the statistical power
and the margin is what says whether it meant anything.

`document_native.md` is the same three extractions in their original markup. Judge from
`document_canonical.md`. Open the native file only after recording your verdict, if at all.
"""


def write_human_subset(fs, entries: list[dict]) -> None:
    """Copy the human-adjudication packets somewhere self-contained, with a form to fill in."""
    chosen = [entry for entry in entries if entry["human_subset"]]
    for entry in chosen:
        source = f"{PACKETS_PREFIX}/{entry['packet_id']}"
        for path in fs.glob(f"{source}/*"):
            name = path.rsplit("/", 1)[-1]
            destination = f"{HUMAN_PACKET_PREFIX}/{entry['packet_id']}/{name}"
            if not fs.exists(destination):
                with fs.open(path, "rb") as reader, fs.open(destination, "wb") as writer:
                    writer.write(reader.read())
    with fs.open(f"{HUMAN_PACKET_PREFIX}/README.md", "w") as stream:
        stream.write(HUMAN_INSTRUCTIONS.format(count=len(chosen)))
    with fs.open(f"{HUMAN_PACKET_PREFIX}/verdicts.csv", "w") as stream:
        stream.write("packet_id,ranking,margin,reason\n")
        for entry in chosen:
            stream.write(f"{entry['packet_id']},,,\n")
    logger.info("human subset: %d packets -> %s", len(chosen), HUMAN_PACKET_PREFIX)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    configure_logging(logging.INFO)
    if JUDGE_KEY_VAR not in os.environ:
        raise RuntimeError(f"{JUDGE_KEY_VAR} is not set; the judge cannot run without it")
    fs = storage()

    with fs.open(KEY_PATH, "r") as stream:
        key = json.load(stream)
    entries = {entry["packet_id"]: entry for entry in key}
    control = [entry["packet_id"] for entry in key if entry["style_control"]]
    logger.info("key: %d packets, %d style-control", len(entries), len(control))

    tasks = [Task(packet_id, Presentation.CANONICAL, JUDGE_MODEL) for packet_id in entries]
    tasks += [Task(packet_id, Presentation.NATIVE, JUDGE_MODEL) for packet_id in control]
    tasks += [Task(packet_id, Presentation.CANONICAL, SECOND_JUDGE) for packet_id in control]
    asyncio.run(run_tasks(fs, entries, tasks))

    canonical = load_verdicts(fs, JUDGE_MODEL, Presentation.CANONICAL)
    native = load_verdicts(fs, JUDGE_MODEL, Presentation.NATIVE)
    second = load_verdicts(fs, SECOND_JUDGE, Presentation.CANONICAL)
    logger.info("verdicts: %d canonical, %d native, %d second judge", len(canonical), len(native), len(second))

    with fs.open(BASELINE_KEY_PATH, "r") as stream:
        baseline_entries = {entry["packet_id"]: entry for entry in json.load(stream)}
    baseline_canonical = load_verdicts(fs, JUDGE_MODEL, Presentation.CANONICAL, BASELINE_VERDICT_PREFIX)
    logger.info("baseline verdicts: %d canonical over %d packets", len(baseline_canonical), len(baseline_entries))

    # The two arms are reported separately and never pooled. The paired arm is the same documents
    # the baseline judged, so it carries the before-and-after; the extension is fresh documents in
    # strata the baseline under-sampled, so it carries width but no pair. One number over both
    # would be a weighted average of a paired contrast and an unpaired estimate.
    arms = {
        str(arm): {packet: result for packet, result in canonical.items() if entries[packet]["arm"] == str(arm)}
        for arm in Arm
    }

    report = {
        "library_version": LIBRARY_VERSION,
        "packets_judged": len(canonical),
        "judge": JUDGE_MODEL,
        "second_judge": SECOND_JUDGE,
        "by_arm": {},
        # The head-to-head as it moved between builds, on the documents judged under both, with the
        # pages and the blinding held fixed. This is the number the re-run exists to produce.
        "paired_shift": {
            "overall": paired_shift(baseline_entries, baseline_canonical, entries, arms[str(Arm.PAIRED)]),
            "by_stratum": paired_shift(
                baseline_entries, baseline_canonical, entries, arms[str(Arm.PAIRED)], key="stratum"
            ),
        },
        "style_effect": style_effect(entries, canonical, native),
        "inter_judge": inter_judge(entries, canonical, second),
        "margins": dict(Counter(result["verdict"]["margin"] for result in canonical.values())),
        "cost": sum(result.get("cost") or 0.0 for result in (*canonical.values(), *native.values(), *second.values())),
    }
    for name, verdicts in arms.items():
        if not verdicts:
            continue
        tally = Tally()
        for packet_id, result in verdicts.items():
            tally.add(unblind(entries[packet_id], result["verdict"]), result["verdict"]["margin"] in DECISIVE_MARGINS)
        low, high = _wilson(tally.inspector_over_docling, tally.pairs)
        report["by_arm"][name] = {
            "overall": {**tally.summary(), "inspector_over_docling_ci95": [low, high]},
            "by_stratum": tally_by(entries, verdicts, "stratum"),
            "by_pdf_type": tally_by(
                {
                    packet: {**entry, "pdf_type": entry["document_metrics"]["inspector_pdf_type"]}
                    for packet, entry in entries.items()
                },
                verdicts,
                "pdf_type",
            ),
            "proxy_label": proxy_label_check(entries, verdicts),
        }
    with fs.open(REPORT_PATH, "w") as stream:
        json.dump(report, stream, indent=2, default=float)

    write_human_subset(fs, key)
    print(json.dumps(report, indent=2, default=float))
    logger.info("wrote %s", REPORT_PATH)


if __name__ == "__main__":
    main()

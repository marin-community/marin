# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adjudicate the two renderings' readings against the page, and read the verdicts out.

:mod:`~experiments.datakit.build_pdf_source.quality.build_render_adjudication_set` writes the
packets; this module judges them. The question is direction, not divergence: the render study proved
the two readings differ and could not say which is right, so the statistic here is
**P(the PDFium-rendered page's reading is judged the more faithful one)** and the null is 0.5.

**Forced pairwise choice, with equivalence recorded separately and never used as a weight.** The
route adjudication measured this instrument against 45 human verdicts and the split was sharp: the
pairwise rank agreed at 0.756 overall and 1.000 where the human was confident, while the *margin*
agreed at 0.22 -- the model called 36/45 "large" where the human called 6/45 large and 14/45
equivalent. So the ranking carries the result and the equivalence flag is reported as description
only. It is not a confidence, nothing is gated on it, and no interval is widened or narrowed by it.
A high equivalence rate is a substantive answer here rather than a failure to discriminate: if two
readings of the same page are genuinely interchangeable, that is the finding that overturns the
rejection.

**Three judging arms, and the third is the control.**

``mupdf``
    Reference page drawn by MuPDF.
``pdfium``
    Reference page drawn by PDFium. Same packet, same text, same blinding, same label order -- only
    the image differs, so a verdict that moves between the two arms moved because of the reference.
``mupdf_repeat``
    The ``mupdf`` arm bought a second time. Without it, "the verdict flips on 18% of pages between
    references" has no scale: a judge asked the same question twice does not answer it the same way
    twice either, and the reference effect is only the excess over that floor. This mirrors the
    render study's own design, where the MuPDF-against-MuPDF pair is what made 0.9591 readable.

:data:`SECOND_JUDGE` re-runs both reference arms under a different vendor's model, so inter-judge
agreement is attached to the headline rather than the result resting on one model's taste.

**Two headline numbers, never one.** The draw oversamples the divergent tail, so the stratified
mean is what a corpus made entirely of hard pages would experience and the post-stratified estimate
is what this corpus would. :func:`post_stratified` reweights by each stratum's share of the 1,795
study pages. The route adjudication is why both are printed: its stratified headline read 0.414 and
post-stratifying moved it to ~0.51, and only one of those is a statement about the corpus.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-08a \\
        --job-name pdfium-render-adjudication-judge \\
        --cpu 8 --memory 16GB --disk 16GB --enable-extra-resources \\
        -e OR_KEY_SCALE_UP "$OR_KEY_SCALE_UP" \\
        -- python -m experiments.datakit.build_pdf_source.quality.judge_render_adjudication_set
"""

import asyncio
import base64
import json
import logging
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import httpx
import numpy as np
import polars as pl
from rigging.log_setup import configure_logging

from experiments.datakit.build_pdf_source.quality.build_render_adjudication_set import (
    ARMS,
    BLIND_LABELS,
    KEY_PATH,
    OUTPUT_PREFIX,
    PACKETS_PREFIX,
    READINGS_PATH,
)
from experiments.datakit.build_pdf_source.quality.probe_pdf_inspector import storage

logger = logging.getLogger(__name__)

VERDICT_PREFIX = f"{OUTPUT_PREFIX}/verdicts"
REPORT_PATH = f"{OUTPUT_PREFIX}/render_adjudication_report.json"

JUDGE_MODEL = "openai/gpt-5.6-luna"
# A different vendor's model, so inter-judge agreement is not two samples of one model's taste.
SECOND_JUDGE = "google/gemini-3.7-flash"
JUDGE_KEY_VAR = "OR_KEY_SCALE_UP"
REASONING_EFFORT = "medium"

JUDGE_CONCURRENCY = 48
JUDGE_MAX_ATTEMPTS = 6
JUDGE_TIMEOUT = 300.0
MAX_COMPLETION_TOKENS = 4000

# Which reference each arm shows, and which arms each judge runs. The repeat arm exists only under
# the primary judge: it measures that judge's own reproducibility, which is what the reference
# effect is compared against, and a second model's reproducibility would answer a different question.
REFERENCE_BY_ARM = {"mupdf": "mupdf", "pdfium": "pdfium", "mupdf_repeat": "mupdf"}
PRIMARY_ARMS = ("mupdf", "pdfium", "mupdf_repeat")
SECOND_JUDGE_ARMS = ("mupdf", "pdfium")

SYSTEM_PROMPT = """You adjudicate PDF text extraction. You are shown one rendered page of a PDF and \
two transcriptions of that same page, labelled Extraction A and Extraction B.

The rendered page image is the ground truth. Judge only how faithfully each transcription \
reproduces what is actually on the page:

- Content completeness: text present on the page that a transcription dropped, and text a \
transcription produced that is not on the page.
- Reading order: whether multi-column, sidebar and caption text is serialized in the order a reader \
would follow. Scrambled order is a severe failure even when every word is present.
- Character fidelity: garbled or wrongly-mapped characters, broken diacritics, wrong script, \
mojibake, missing ligatures, wrong digits.
- Structure: whether tables keep their rows and cells, whether headings and lists survive as \
distinct from body text, whether equations are readable.

Explicitly ignore, because they are presentation conventions and not quality:
- Whitespace, padding, line-wrapping and blank-line differences.
- Whether a transcription transcribes text inside charts and diagrams.
- Where a transcription was clipped by the packaging, marked "[clipped at N characters]". Judge only \
the text shown, and do not penalise a transcription for stopping at the clip.

You must name the more faithful transcription even when they are close, and then say separately \
whether the two are equivalent. Reply with JSON only, no prose outside it:

{"better": "A" | "B",
 "equivalent": true | false,
 "worst_error": "<short phrase naming the worse transcription's worst error, or null>",
 "reason": "<one or two sentences>"}

"better" must be "A" or "B" and must be filled in even when the two are interchangeable; if they \
are, set "equivalent" to true and pick either. "equivalent" is true when the two transcriptions \
would serve a reader equally well."""


# ---------------------------------------------------------------------------
# One judging task
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Task:
    """One verdict to buy: a packet, a reference arm, and the model that judges it."""

    packet_id: str
    arm: str
    model: str

    @property
    def path(self) -> str:
        return f"{VERDICT_PREFIX}/{self.model.replace('/', '_')}/{self.arm}/{self.packet_id}.json"


def build_request(fs, task: Task) -> list[dict]:
    """The multimodal message for one verdict: the reference page, then the two readings.

    The image leads so the judge reads the page before it reads either transcription of it.
    """
    with fs.open(f"{PACKETS_PREFIX}/{task.packet_id}/reference_{REFERENCE_BY_ARM[task.arm]}.png", "rb") as stream:
        image = base64.b64encode(stream.read()).decode()
    with fs.open(f"{PACKETS_PREFIX}/{task.packet_id}/document.md", "r") as stream:
        document = stream.read()
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
                {"type": "text", "text": document},
            ],
        },
    ]


def parse_verdict(text: str) -> dict:
    """Read the judge's JSON, tolerating a fenced block around it."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("```")[1]
        stripped = stripped[4:] if stripped.startswith("json") else stripped
    verdict = json.loads(stripped)
    if verdict["better"] not in BLIND_LABELS:
        raise ValueError(f"better is not one of {BLIND_LABELS}: {verdict['better']}")
    if not isinstance(verdict["equivalent"], bool):
        raise ValueError(f"equivalent is not a boolean: {verdict['equivalent']!r}")
    return verdict


async def judge_one(client: httpx.AsyncClient, fs, task: Task, gate: asyncio.Semaphore) -> dict | None:
    """Buy one verdict, or return ``None`` if the judge never produced a usable one."""
    messages = build_request(fs, task)
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
                logger.info("%s/%s attempt %d: %s", task.packet_id, task.arm, attempt + 1, error)
                await asyncio.sleep(min(2**attempt, 30))
                continue
        return {
            "packet_id": task.packet_id,
            "arm": task.arm,
            "model": task.model,
            "verdict": verdict,
            "cost": body.get("usage", {}).get("cost"),
        }
    logger.warning("%s/%s: no verdict after %d attempts", task.packet_id, task.arm, JUDGE_MAX_ATTEMPTS)
    return None


def bought(fs, model: str, arm: str) -> set[str]:
    """Packet ids already judged under one model and arm, from one listing.

    A ``fs.exists`` per task would be one S3 round trip per packet per arm -- 3,575 of them for a
    715-page draw, twice over, before a single verdict is bought. The prefix holds one object per
    packet, so listing it once answers the same question.
    """
    prefix = f"{VERDICT_PREFIX}/{model.replace('/', '_')}/{arm}"
    if not fs.exists(prefix):
        return set()
    return {path.rsplit("/", 1)[-1].removesuffix(".json") for path in fs.ls(prefix, detail=False)}


async def run_tasks(fs, tasks: list[Task]) -> None:
    """Buy every verdict not already on storage."""
    already = {(model, arm): bought(fs, model, arm) for model, arm in {(task.model, task.arm) for task in tasks}}
    pending = [task for task in tasks if task.packet_id not in already[(task.model, task.arm)]]
    logger.info("judging: %d tasks, %d already bought", len(tasks), len(tasks) - len(pending))
    if not pending:
        return
    gate = asyncio.Semaphore(JUDGE_CONCURRENCY)
    headers = {"Authorization": f"Bearer {os.environ[JUDGE_KEY_VAR]}"}
    spent = 0.0
    async with httpx.AsyncClient(base_url="https://openrouter.ai/api/v1", headers=headers) as client:
        for start in range(0, len(pending), JUDGE_CONCURRENCY * 4):
            batch = pending[start : start + JUDGE_CONCURRENCY * 4]
            results = await asyncio.gather(*(judge_one(client, fs, task, gate) for task in batch))
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


def preferred(entry: dict, verdict: dict) -> str:
    """Which engine's reading the judge picked, translated out of the blinding."""
    return entry["labels"][verdict["better"]]


def _wilson(successes: int, total: int) -> tuple[float, float]:
    """A 95% Wilson interval, which is what an ``n`` of 40 needs instead of a bare proportion."""
    if total == 0:
        return (float("nan"), float("nan"))
    z, phat = 1.96, successes / total
    denominator = 1 + z**2 / total
    centre = (phat + z**2 / (2 * total)) / denominator
    spread = z * np.sqrt(phat * (1 - phat) / total + z**2 / (4 * total**2)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


@dataclass
class Tally:
    """Verdicts over one slice of pages."""

    pages: int = 0
    pdfium_preferred: int = 0
    equivalent: int = 0
    domains: set = field(default_factory=set)

    def add(self, engine: str, equivalent: bool, domain: str) -> None:
        self.pages += 1
        self.pdfium_preferred += engine == "pdfium"
        self.equivalent += equivalent
        self.domains.add(domain)

    def summary(self) -> dict:
        low, high = _wilson(self.pdfium_preferred, self.pages)
        return {
            "pages": self.pages,
            "domains": len(self.domains),
            "pdfium_preferred": self.pdfium_preferred / max(self.pages, 1),
            "pdfium_preferred_ci95": [low, high],
            "equivalent_rate": self.equivalent / max(self.pages, 1),
        }


def tally_by(entries: dict, verdicts: dict[str, dict], key: str) -> dict[str, dict]:
    """Verdicts grouped by one field of the key entry, with a domain count beside each.

    Near-duplicates cluster by publisher, so domains are the independent unit and the interval on
    the page count is the optimistic bound.
    """
    grouped: dict[str, Tally] = defaultdict(Tally)
    members: dict[str, dict[str, dict]] = defaultdict(dict)
    for packet_id, result in verdicts.items():
        entry = entries[packet_id]
        name = str(entry[key])
        grouped[name].add(
            preferred(entry, result["verdict"]),
            bool(result["verdict"]["equivalent"]),
            entry.get("domain", ""),
        )
        members[name][packet_id] = result
    return {
        name: {**tally.summary(), "balanced": label_balanced(entries, members[name])}
        for name, tally in sorted(grouped.items())
    }


def label_balanced(entries: dict, verdicts: dict[str, dict]) -> dict:
    """P(PDFium's reading preferred), estimated so the judge's position preference cancels exactly.

    This judge breaks ties by position, and hard: told to "pick either" when the two readings are
    equivalent, it names the first one. On the pages whose two readings are *byte-identical* it
    called them equivalent 99.6% of the time and then picked label A on 240 of 241. Overall it names
    A on roughly three quarters of pages.

    That is not a bias toward either renderer -- which engine hides behind A is drawn per page -- so
    the naive rate is still unbiased. It is, however, noise, and on a stratum where most pages are
    ties the naive rate is mostly reporting how the label draw happened to fall: ``unchanged`` came
    out at 0.443 because PDFium drew label A on 44.0% of its pages, not because anything about the
    two readings differed.

    Conditioning on the draw removes it. With ``a = P(pick A | A is PDFium's reading)`` and
    ``b = P(pick A | A is MuPDF's reading)``, the estimate is ``(a + 1 - b) / 2``: any position
    preference shared by both halves adds to ``a`` and to ``b`` alike and cancels in the difference,
    and what survives is the part of the judge's choice that tracked the text. A judge that only ever
    looked at position gives ``a = b`` and lands on exactly 0.5.
    """
    pdfium_first = [key for key in verdicts if entries[key]["labels"][BLIND_LABELS[0]] == "pdfium"]
    mupdf_first = [key for key in verdicts if entries[key]["labels"][BLIND_LABELS[0]] == "mupdf"]
    if not pdfium_first or not mupdf_first:
        return {"pages": len(verdicts), "pdfium_preferred": float("nan"), "stderr": float("nan")}

    def pick_first(keys: list[str]) -> float:
        return sum(verdicts[key]["verdict"]["better"] == BLIND_LABELS[0] for key in keys) / len(keys)

    a, b = pick_first(pdfium_first), pick_first(mupdf_first)
    estimate = 0.5 * (a + 1 - b)
    variance = 0.25 * (a * (1 - a) / len(pdfium_first) + b * (1 - b) / len(mupdf_first))
    stderr = float(np.sqrt(variance))
    return {
        "pages": len(verdicts),
        "pdfium_preferred": estimate,
        "stderr": stderr,
        "ci95": [estimate - 1.96 * stderr, estimate + 1.96 * stderr],
        "pick_first_given_pdfium_first": a,
        "pick_first_given_mupdf_first": b,
        "position_preference": (a + b) / 2,
    }


def post_stratified(entries: dict, verdicts: dict[str, dict]) -> dict:
    """The corpus-page-weighted preference rate, reweighting the draw back to corpus shape.

    Each stratum's per-page rate is estimated from the pages drawn from it and weighted by that
    stratum's share of the 1,795 study pages, so the deliberate oversampling of the divergent tail
    does not travel into the headline. A stratum with a single page contributes weight without
    contributing variance -- honest for ``reverse_catastrophic``, which is one page in the whole
    corpus, and would not be for a large one.

    Both estimators are returned over the same weights. ``pdfium_preferred`` is the plain rate;
    ``pdfium_preferred_balanced`` applies :func:`label_balanced` within each stratum, which is the
    one to read -- the judge's position tie-break makes the plain per-stratum rate partly a report
    of how the label draw fell.
    """
    by_stratum: dict[str, dict[str, dict]] = defaultdict(dict)
    weights: dict[str, float] = {}
    for packet_id, result in verdicts.items():
        entry = entries[packet_id]
        by_stratum[entry["stratum"]][packet_id] = result
        weights[entry["stratum"]] = entry["page_share"]

    # A stratum's weight is scaled by the share of its drawn pages that reached this subset. On the
    # full set every share is 1 and this is ordinary post-stratification; on a subset -- the pages
    # whose two readings are not byte-identical, say -- it is what makes the answer a statement about
    # the corpus's *subset* rather than about a reweighted average of within-stratum rates. Without
    # it, a cut that removes 90% of one stratum and 10% of another still weights them as drawn.
    drawn: Counter = Counter(entry["stratum"] for entry in entries.values())
    weights = {name: weight * len(by_stratum[name]) / drawn[name] for name, weight in weights.items()}

    total_weight = sum(weights[name] for name in by_stratum)
    estimate = variance = 0.0
    balanced_estimate = balanced_variance = 0.0
    for name, group in by_stratum.items():
        weight = weights[name] / total_weight
        count = len(group)
        rate = sum(preferred(entries[key], result["verdict"]) == "pdfium" for key, result in group.items()) / count
        estimate += weight * rate
        if count > 1:
            variance += weight**2 * rate * (1 - rate) / (count - 1)
        balanced = label_balanced(entries, group)
        # A stratum whose draw put every page's PDFium reading on the same label cannot separate
        # position from engine; it falls back to the plain rate rather than dropping its weight.
        if np.isnan(balanced["pdfium_preferred"]):
            balanced_estimate += weight * rate
        else:
            balanced_estimate += weight * balanced["pdfium_preferred"]
            balanced_variance += weight**2 * balanced["stderr"] ** 2
    stderr = float(np.sqrt(variance))
    balanced_stderr = float(np.sqrt(balanced_variance))
    return {
        "pdfium_preferred": estimate,
        "stderr": stderr,
        "ci95": [estimate - 1.96 * stderr, estimate + 1.96 * stderr],
        "pdfium_preferred_balanced": balanced_estimate,
        "stderr_balanced": balanced_stderr,
        "ci95_balanced": [balanced_estimate - 1.96 * balanced_stderr, balanced_estimate + 1.96 * balanced_stderr],
        "weight_covered": total_weight,
        "strata": len(by_stratum),
    }


def arm_agreement(entries: dict, left: dict[str, dict], right: dict[str, dict]) -> dict:
    """How often two sets of verdicts on the same pages name the same engine.

    Used twice and it means the same thing both times: against the repeat arm it is the judge's own
    reproducibility floor, and against the other reference arm it is reproducibility plus whatever
    the reference contributes. The difference between the two is the reference effect.
    """
    shared = sorted(set(left) & set(right))
    if not shared:
        return {"pages": 0}
    agree = 0
    left_pdfium = right_pdfium = 0
    for packet_id in shared:
        entry = entries[packet_id]
        one, two = preferred(entry, left[packet_id]["verdict"]), preferred(entry, right[packet_id]["verdict"])
        agree += one == two
        left_pdfium += one == "pdfium"
        right_pdfium += two == "pdfium"
    return {
        "pages": len(shared),
        "same_engine_rate": agree / len(shared),
        "left_pdfium_preferred": left_pdfium / len(shared),
        "right_pdfium_preferred": right_pdfium / len(shared),
    }


def load_verdicts(fs, entries: dict, model: str, arm: str) -> dict[str, dict]:
    """Every bought verdict for one judge and one arm, keyed by packet."""
    verdicts = {}
    for packet_id in bought(fs, model, arm) & set(entries):
        with fs.open(f"{VERDICT_PREFIX}/{model.replace('/', '_')}/{arm}/{packet_id}.json", "r") as stream:
            verdicts[packet_id] = json.load(stream)
    return verdicts


def _log_tally(title: str, table: dict[str, dict]) -> None:
    logger.info("--- %s ---", title)
    for name, summary in table.items():
        balanced = summary["balanced"]
        logger.info(
            "%-24s n=%4d (%3d domains)  PDFium %.3f [%.3f, %.3f]  balanced %.3f +/- %.3f  equivalent %.3f",
            name,
            summary["pages"],
            summary["domains"],
            summary["pdfium_preferred"],
            summary["pdfium_preferred_ci95"][0],
            summary["pdfium_preferred_ci95"][1],
            balanced["pdfium_preferred"],
            balanced["stderr"],
            summary["equivalent_rate"],
        )


def _pooled(entries: dict, verdicts: dict[str, dict]) -> dict:
    tally = Tally()
    for packet_id, result in verdicts.items():
        entry = entries[packet_id]
        tally.add(preferred(entry, result["verdict"]), bool(result["verdict"]["equivalent"]), entry.get("domain", ""))
    return {**tally.summary(), "balanced": label_balanced(entries, verdicts)}


def identical_readings(fs, entries: dict) -> set[str]:
    """Packets whose two readings came back byte-identical.

    Read from ``readings.parquet`` rather than the key, because the key carries agreement scores and
    a bigram F1 of 1.0 is not the same claim as textual identity -- the metric normalises case,
    whitespace and markup before it compares.
    """
    with fs.open(READINGS_PATH, "rb") as stream:
        readings = pl.read_parquet(stream, columns=["source_id", "page_index", "mupdf_text", "pdfium_text"])
    packet_of = {(entry["source_id"], entry["page_index"]): packet_id for packet_id, entry in entries.items()}
    identical = set()
    for row in readings.iter_rows(named=True):
        packet_id = packet_of.get((row["source_id"], row["page_index"]))
        if packet_id is not None and row["mupdf_text"] == row["pdfium_text"]:
            identical.add(packet_id)
    return identical


def analyze(fs, entries: dict) -> dict:
    """Every arm's verdicts, read out per stratum, pooled, and reweighted to the corpus."""
    loaded = {
        (model, arm): load_verdicts(fs, entries, model, arm)
        for model, arms in ((JUDGE_MODEL, PRIMARY_ARMS), (SECOND_JUDGE, SECOND_JUDGE_ARMS))
        for arm in arms
    }

    report: dict = {"judge": JUDGE_MODEL, "second_judge": SECOND_JUDGE, "arms": {}}
    for (model, arm), verdicts in loaded.items():
        if not verdicts:
            logger.warning("no verdicts for %s / %s", model, arm)
            continue
        entry = {
            "model": model,
            "stratified": _pooled(entries, verdicts),
            "post_stratified": post_stratified(entries, verdicts),
            "by_stratum": tally_by(entries, verdicts, "stratum"),
        }
        report["arms"][f"{model}/{arm}"] = entry
        logger.info(
            "=== %s / %s over %d pages: stratified %.3f [%.3f, %.3f], balanced %.3f +/- %.3f "
            "| post-stratified %.3f +/- %.3f, balanced %.3f +/- %.3f | equivalent %.3f, picks first %.3f",
            model,
            arm,
            entry["stratified"]["pages"],
            entry["stratified"]["pdfium_preferred"],
            entry["stratified"]["pdfium_preferred_ci95"][0],
            entry["stratified"]["pdfium_preferred_ci95"][1],
            entry["stratified"]["balanced"]["pdfium_preferred"],
            entry["stratified"]["balanced"]["stderr"],
            entry["post_stratified"]["pdfium_preferred"],
            entry["post_stratified"]["stderr"],
            entry["post_stratified"]["pdfium_preferred_balanced"],
            entry["post_stratified"]["stderr_balanced"],
            entry["stratified"]["equivalent_rate"],
            entry["stratified"]["balanced"]["position_preference"],
        )
        _log_tally(f"{model} / {arm}: by stratum", entry["by_stratum"])

    primary_mupdf = loaded.get((JUDGE_MODEL, "mupdf"), {})
    report["judge_reproducibility"] = arm_agreement(
        entries, primary_mupdf, loaded.get((JUDGE_MODEL, "mupdf_repeat"), {})
    )
    report["reference_effect"] = arm_agreement(entries, primary_mupdf, loaded.get((JUDGE_MODEL, "pdfium"), {}))
    report["inter_judge_mupdf"] = arm_agreement(entries, primary_mupdf, loaded.get((SECOND_JUDGE, "mupdf"), {}))
    report["inter_judge_pdfium"] = arm_agreement(
        entries, loaded.get((JUDGE_MODEL, "pdfium"), {}), loaded.get((SECOND_JUDGE, "pdfium"), {})
    )
    logger.info(
        "judge reproducibility (same reference, judged twice): %.3f over %d pages",
        report["judge_reproducibility"].get("same_engine_rate", float("nan")),
        report["judge_reproducibility"].get("pages", 0),
    )
    logger.info(
        "reference effect (MuPDF reference against PDFium reference): %.3f agreement over %d pages",
        report["reference_effect"].get("same_engine_rate", float("nan")),
        report["reference_effect"].get("pages", 0),
    )
    logger.info(
        "inter-judge agreement: %.3f on the MuPDF reference, %.3f on the PDFium reference",
        report["inter_judge_mupdf"].get("same_engine_rate", float("nan")),
        report["inter_judge_pdfium"].get("same_engine_rate", float("nan")),
    )

    # A third of the draw has *byte-identical* readings, and on those the judge is choosing between
    # two copies of the same string: the verdict is a coin flip by construction and carries no
    # information about either renderer. Reported separately because the cut is an objective fact
    # about the text rather than a model opinion -- unlike the equivalence flag below, this one can
    # be trusted -- and because it separates two different questions. The headline answers "what
    # would this corpus experience"; this answers "when the renderers actually move the reading,
    # which way does it move".
    identical = identical_readings(fs, entries)
    differing = {packet_id: result for packet_id, result in primary_mupdf.items() if packet_id not in identical}
    report["identical_reading_rate"] = len(identical & set(primary_mupdf)) / max(len(primary_mupdf), 1)
    if differing:
        report["readings_differ"] = {
            "stratified": _pooled(entries, differing),
            "post_stratified": post_stratified(entries, differing),
        }
        logger.info(
            "readings byte-identical on %.3f of judged pages; over the rest: stratified %.3f "
            "over %d pages, post-stratified %.3f +/- %.3f",
            report["identical_reading_rate"],
            report["readings_differ"]["stratified"]["pdfium_preferred"],
            report["readings_differ"]["stratified"]["pages"],
            report["readings_differ"]["post_stratified"]["pdfium_preferred"],
            report["readings_differ"]["post_stratified"]["stderr"],
        )

    # A page whose two readings differ mostly because one of them ran away or was cut off is the
    # render study's known confound, reported here the same way it was reported there.
    clean = {
        packet_id: result
        for packet_id, result in primary_mupdf.items()
        if not entries[packet_id]["truncated"] and not entries[packet_id]["runaway_length"]
    }
    if clean:
        report["clean_of_truncation"] = {
            "stratified": _pooled(entries, clean),
            "post_stratified": post_stratified(entries, clean),
        }
        logger.info(
            "dropping truncated and runaway-length pages: stratified %.3f over %d pages, post-stratified %.3f",
            report["clean_of_truncation"]["stratified"]["pdfium_preferred"],
            report["clean_of_truncation"]["stratified"]["pages"],
            report["clean_of_truncation"]["post_stratified"]["pdfium_preferred"],
        )

    # Where direction would live if there is any: the pages the judge did *not* call equivalent.
    # Secondary and caveated, never the headline. Conditioning on the equivalence flag is
    # conditioning on the model's own margin, and the margin is the part of this instrument that
    # failed validation -- 0.22 agreement against 45 human verdicts. A number here that disagrees
    # with the headline says the flag is unreliable, not that the headline is wrong.
    decisive = {packet_id: result for packet_id, result in primary_mupdf.items() if not result["verdict"]["equivalent"]}
    if decisive:
        report["judge_called_not_equivalent"] = {
            "stratified": _pooled(entries, decisive),
            "post_stratified": post_stratified(entries, decisive),
        }
        logger.info(
            "pages the judge did not call equivalent (secondary, margin-conditioned): "
            "stratified %.3f over %d pages, post-stratified %.3f",
            report["judge_called_not_equivalent"]["stratified"]["pdfium_preferred"],
            report["judge_called_not_equivalent"]["stratified"]["pages"],
            report["judge_called_not_equivalent"]["post_stratified"]["pdfium_preferred"],
        )

    counts = Counter(entries[packet_id]["stratum"] for packet_id in primary_mupdf)
    logger.info("packets judged per stratum: %s", dict(sorted(counts.items())))
    return report


def main() -> None:
    configure_logging(logging.INFO)
    fs = storage()
    with fs.open(KEY_PATH, "r") as stream:
        entries = {entry["packet_id"]: entry for entry in json.load(stream)}
    logger.info("key: %d packets, arms %s", len(entries), ARMS)

    tasks = [
        Task(packet_id, arm, model)
        for model, arms in ((JUDGE_MODEL, PRIMARY_ARMS), (SECOND_JUDGE, SECOND_JUDGE_ARMS))
        for arm in arms
        for packet_id in entries
    ]
    asyncio.run(run_tasks(fs, tasks))

    report = analyze(fs, entries)
    with fs.open(REPORT_PATH, "w") as stream:
        json.dump(report, stream, indent=2)
    logger.info("report -> %s", REPORT_PATH)


if __name__ == "__main__":
    main()

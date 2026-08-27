# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Detect and repair repetition loops in a vision model's page transcription.

A VLM asked to transcribe a page can fall into a degenerate cycle: it emits one unit -- a row of
empty table cells, a run of leader dots, a fabricated URL, a counter -- over and over until the
token cap stops it. The result is not an obvious failure. It is ordinary-looking Markdown that
nothing in the response distinguishes from a faithful transcription of a repetitive page, and in the
10% sample it accounts for roughly 3% of all extracted characters.

What separates a loop from a page that is *genuinely* repetitive is not redundancy but **invariance**.
A real table is redundant -- ``| 1.000 | 1.000 | 1.000 |`` -- yet its rows keep introducing new cell
values, while a loop repeats one fixed unit to the end of the output. A detector built on
compressibility cannot tell them apart: measured against hand labels it scored 0.98 precision inside
``partial`` documents and 0.04 inside ``success`` ones, condemning benchmark tables, census counts
and cross-tabs. So the feature here is the extent of the maximal *exactly periodic* span ending at
the end of the page, which is at once the evidence and the salvage cut point.

The period is recovered rather than searched. The last :data:`_PROBE_CHARS` characters of a periodic
tail must occur again exactly one period earlier, so a single ``str.rfind`` hands back the period and
slice comparisons grow the span -- all at C speed, over the ~3% of pages that survive a compression
gate. The whole pass costs about 23 microseconds per KB, against roughly 15 milliseconds per page for
rendering alone.

Digits are folded to a single symbol before the search, because an incrementing counter (``Fig. 1``,
``Fig. 2``, ... ``Fig. 525``) is a loop that no exact-repetition test would see. Folding is also
dangerous: a register of filings whose blocks differ *only* in reference numbers and dates folds to
an exactly repeating template, and one such document supplied 61 of the 67 false positives an
unguarded folded search produced across the whole sample. A span that is periodic only after folding
therefore has to earn it -- :func:`counter_score` requires the swallowed digits to be themselves
degenerate, near-constant or near-arithmetic. Real record numbers are neither; a counter is both.

Thresholds are calibrated against 897 hand-labeled pages, each labeled by looking at the rendered
page beside its transcription. At the settings here the detector scores precision 1.000 [0.964,
1.000] and recall 0.866 inside ``partial`` documents. It does not catch near-periodic block cycling,
and it does not address bounded fabricated fill (a model inventing plausible values for empty table
cells), which is a distinct defect with no detector.

**The calibration assumes the token cap.** A runaway loop hits ``max_tokens`` and marks the page
truncated, which is why :func:`repair_page` only examines truncated pages: the gate costs no measured
recall and removes every remaining false positive. Raise the cap and this reasoning has to be redone.
"""

import re
import zlib
from dataclasses import dataclass
from itertools import pairwise

# Probe taken from the end of the text; its earlier occurrence gives the period. Long enough that a
# chance re-occurrence of ordinary prose is negligible, short enough that a period of a few dozen
# characters still has room to repeat inside one page.
_PROBE_CHARS = 64
# Characters back from the end at which to take the probe. The later anchors let the search tolerate
# a short non-periodic coda after the loop -- a closing table row, a stray fence.
_PROBE_ANCHORS = (0, 250, 1200)
# A span must hold at least this many whole periods to be degeneracy rather than a document that
# happens to repeat a heading twice.
_MIN_PERIODS = 4
# Below this many digit runs, folding cannot have been what created the periodicity.
_MIN_DIGIT_RUNS = 4

_DIGIT_FOLD = str.maketrans("0123456789", "0000000000")
_DIGIT_RUN = re.compile(r"[0-9]+")

# Cheap gate before the period search: no periodic span of the sizes acted on here leaves a page this
# incompressible. Skips the scan on the great majority of pages, which are ordinary prose.
_GATE_COMPRESSION_RATIO = 0.42
# zlib runs on at most this many characters; the ratio of a long tail is already decisive.
_GATE_SAMPLE_CHARS = 8000

# Length of the probe used to walk the cut back to where the repeated unit *first* appears.
_ONSET_PROBE_CHARS = 40
# ...but only this many periods back. The repeated unit often has a legitimate first occurrence -- a
# real figure caption the model then got stuck on -- and an unbounded search walks the cut back to it,
# discarding the entire correct transcription in front.
_ONSET_WALKBACK_PERIODS = 3


@dataclass(frozen=True)
class LoopOptions:
    """Thresholds for calling a periodic span a loop and for cutting it out.

    Every field changes the text that reaches the corpus, so the whole set belongs in the extraction
    step's ``hash_attrs``.
    """

    # Nothing shorter is worth acting on. A genuine runaway fills thousands of characters before the
    # cap stops it; a short page that repeats is almost always a form, a stub table, or three lines.
    min_page_chars: int = 3000
    # The span must be this long in absolute terms...
    min_loop_chars: int = 1200
    # ...and this much of the page, so a long page with one repetitive block is not condemned for it.
    min_loop_fraction: float = 0.15
    # ...and must run to (near) the end. A repetitive block the model exits and continues past is a
    # table it transcribed, not a cycle it fell into.
    max_trailing_chars: int = 1500
    # How degenerate the digits inside a folded-only span must be. This guard, not the size
    # thresholds, is what holds precision: switching it off multiplies false positives by 67.
    min_counter_score: float = 0.5
    # A retained prefix shorter than this is not a transcription of anything -- it is the first
    # fragment of the loop itself. A policy floor, not a fitted discriminator.
    min_salvage_prefix: int = 250


@dataclass(frozen=True)
class PageLoop:
    """The maximal periodic span ending at (or near) the end of one page of OCR text.

    A page with no qualifying span reports ``start == end == nchars``, so :attr:`chars` is zero.
    """

    nchars: int
    start: int
    end: int
    period: int
    # Whether the span repeats exactly, without folding digits. A folded-only span is believed only
    # when :attr:`counter_score` vouches for the digits it ignored.
    exact: bool
    counter_score: float

    @property
    def chars(self) -> int:
        return self.end - self.start

    @property
    def fraction(self) -> float:
        return self.chars / self.nchars if self.nchars else 0.0

    @property
    def trailing_chars(self) -> int:
        return self.nchars - self.end


@dataclass(frozen=True)
class PageRepair:
    """What :func:`repair_page` decided about one page.

    :attr:`text` is what belongs in the corpus: the page unchanged when no loop was found, otherwise
    the transcription that precedes the degeneracy, which may be empty.
    """

    text: str
    looped: bool
    dropped_chars: int
    loop_period: int


def _periodic_span(text: str, nchars: int) -> tuple[int, int, int]:
    """Longest exactly periodic span ending at or near the end of ``text``.

    Returns ``(start, end, period)`` in character offsets, or ``(nchars, nchars, 0)`` when no
    qualifying span exists. ``text`` is the right-stripped page, optionally digit-folded; folding is
    character-for-character, so the offsets stay valid for the original string.
    """
    length = len(text)
    best = (nchars, nchars, 0)
    best_span = 0
    for anchor in _PROBE_ANCHORS:
        end = length - anchor
        if end < 2 * _PROBE_CHARS:
            continue
        probe = text[end - _PROBE_CHARS : end]
        previous = text.rfind(probe, 0, end - _PROBE_CHARS)
        if previous < 0:
            continue
        period = (end - _PROBE_CHARS) - previous
        if period <= 0:
            continue
        # Deliberately not reduced to the unit's minimal period. ``rfind`` requires a match to lie
        # entirely before the probe, so this is the true period rounded up to a multiple that clears
        # it, and the backward walk strides in that multiple -- which can leave a repetition or two
        # in the salvaged text. Reducing it is worse: measured over 60k corpus pages, the shorter
        # stride is a weaker constraint and walks back through legitimate table rows and prose that
        # merely align with it, taking real content off 24 of the 39 pages it changed.
        start = end - period
        while start - period >= 0 and text[start - period : start] == text[start : start + period]:
            start -= period
        stop = end
        while stop + period <= length and text[stop - period : stop] == text[stop : stop + period]:
            stop += period
        # A trailing partial period is still degeneracy; count the characters that keep matching.
        while stop < length and text[stop - period] == text[stop]:
            stop += 1

        span = stop - start
        if span >= _MIN_PERIODS * period and span > best_span:
            best_span = span
            best = (start, stop, period)
    return best


def counter_score(span: str) -> float:
    """How degenerate the digit runs inside ``span`` are, in ``[0, 1]``.

    1.0 means they are all the same value or step by a constant -- a counter. Values near 0 mean
    unrelated numbers, which is what a genuine list of records, dates or measurements looks like. A
    span with almost no digits scores 1.0, because there folding cannot have created the periodicity.
    """
    runs = _DIGIT_RUN.findall(span)
    if len(runs) < _MIN_DIGIT_RUNS:
        return 1.0
    modal_value = max(map(runs.count, set(runs))) / len(runs)
    # Oversized runs are page noise, not counters, and int() on them is pointless work.
    values = [int(run) for run in runs if len(run) <= 18]
    if len(values) < _MIN_DIGIT_RUNS:
        return modal_value
    steps = [later - earlier for earlier, later in pairwise(values)]
    modal_step = max(map(steps.count, set(steps))) / len(steps)
    return max(modal_value, modal_step)


def find_loop(text: str, options: LoopOptions) -> PageLoop:
    """Locate the degenerate span at the end of one page's transcription.

    Two searches run: one on the raw text, one with digits folded. A raw span is self-evidently
    degenerate. A folded-only span is scored by :func:`counter_score`, and the caller decides whether
    that score is good enough.
    """
    nchars = len(text)
    empty = PageLoop(nchars=nchars, start=nchars, end=nchars, period=0, exact=True, counter_score=1.0)
    if nchars < options.min_page_chars:
        return empty

    body = text.rstrip()
    folded = body.translate(_DIGIT_FOLD)
    # Gate on the *folded* tail: an incrementing counter does not compress until its digits are
    # folded, and counters are exactly the class an unfolded test misses.
    sample = folded[-_GATE_SAMPLE_CHARS:].encode()
    if len(zlib.compress(sample, 1)) / max(len(sample), 1) > _GATE_COMPRESSION_RATIO:
        return empty

    start, end, period = _periodic_span(body, nchars)
    if end > start:
        return PageLoop(nchars=nchars, start=start, end=end, period=period, exact=True, counter_score=1.0)

    start, end, period = _periodic_span(folded, nchars)
    if end == start:
        return empty
    return PageLoop(
        nchars=nchars,
        start=start,
        end=end,
        period=period,
        exact=False,
        counter_score=counter_score(text[start:end]),
    )


def is_loop(loop: PageLoop, options: LoopOptions) -> bool:
    """Whether a located span is degeneracy rather than a faithfully transcribed repetitive page."""
    if loop.nchars < options.min_page_chars:
        return False
    if loop.chars < options.min_loop_chars or loop.fraction < options.min_loop_fraction:
        return False
    if loop.trailing_chars > options.max_trailing_chars:
        return False
    return loop.exact or loop.counter_score >= options.min_counter_score


def loop_onset(text: str, loop: PageLoop) -> int:
    """Character offset where the degeneracy begins, at or before the exactly periodic span.

    :attr:`PageLoop.start` is where *exact* repetition begins, but a cycle that drifts -- an extra
    space, a separator that changes every few rounds -- was already degenerate before that. The first
    occurrence of the repeated unit is the better onset, bounded to a few periods back so that a unit
    with a legitimate earlier occurrence does not drag the cut through good text.
    """
    if not loop.period:
        return loop.start
    probe = text[loop.start : loop.start + min(loop.period, _ONSET_PROBE_CHARS)].translate(_DIGIT_FOLD)
    if len(probe) < _ONSET_PROBE_CHARS:
        return loop.start
    floor = max(0, loop.start - _ONSET_WALKBACK_PERIODS * loop.period)
    first = text.translate(_DIGIT_FOLD).find(probe, floor, loop.start)
    return first if first >= 0 else loop.start


def salvage(text: str, loop: PageLoop, options: LoopOptions) -> str:
    """The transcription to keep from a page that looped.

    Everything from the onset of the degeneracy onwards is dropped, including whatever the model
    emitted *after* the span: that text sits on the far side of a several-thousand-character cycle,
    so its place on the page is unknown. The cut snaps back to the preceding newline so the retained
    text does not end mid-row, and a prefix too short to be a transcription becomes an empty page
    rather than a stub.
    """
    cut = loop_onset(text, loop)
    newline = text.rfind("\n", 0, cut)
    if newline >= 0:
        cut = newline + 1
    kept = text[:cut].rstrip()
    return kept if len(kept) >= options.min_salvage_prefix else ""


def repair_page(text: str, truncated: bool, options: LoopOptions) -> PageRepair:
    """Detect a repetition loop in one page and cut it out.

    Only truncated pages are examined. A runaway cycle runs until the token cap stops it, so the cap
    is the loop's own signature; gating on it costs no measured recall and removes the last false
    positives, at the price of not seeing a bounded cycle that stops on its own.
    """
    if not truncated:
        return PageRepair(text=text, looped=False, dropped_chars=0, loop_period=0)
    loop = find_loop(text, options)
    if not is_loop(loop, options):
        return PageRepair(text=text, looped=False, dropped_chars=0, loop_period=0)
    kept = salvage(text, loop, options)
    return PageRepair(text=kept, looped=True, dropped_chars=len(text) - len(kept), loop_period=loop.period)

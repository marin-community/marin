# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Detect and repair repetition loops in a vision model's page transcription.

A VLM asked to transcribe a page can fall into a degenerate cycle, emitting one unit over and over
until the token cap stops it. The feature here is the extent of the maximal *exactly periodic* span
ending at the end of the page, which is at once the evidence and the salvage cut point; the period
is recovered with a single ``str.rfind`` of the page's tail and grown by slice comparison.

Digits are folded to a single symbol before the search so that an incrementing counter reads as a
loop. A span that is periodic only after folding has to earn it: :func:`counter_score` requires the
swallowed digits to be themselves near-constant or near-arithmetic.

The calibration assumes the token cap: a runaway loop hits ``max_tokens`` and marks the page
truncated, which is why :func:`repair_page` only examines truncated pages. Raise the cap and this
reasoning has to be redone.
"""

import re
import zlib
from dataclasses import dataclass
from itertools import pairwise

# Probe taken from the end of the text; its earlier occurrence gives the period.
_PROBE_CHARS = 64
# Characters back from the end at which to take the probe; the later anchors tolerate a short
# non-periodic coda after the loop.
_PROBE_ANCHORS = (0, 250, 1200)
# A span must hold at least this many whole periods to count as degeneracy.
_MIN_PERIODS = 4
# Below this many digit runs, folding cannot have been what created the periodicity.
_MIN_DIGIT_RUNS = 4

_DIGIT_FOLD = str.maketrans("0123456789", "0000000000")
_DIGIT_RUN = re.compile(r"[0-9]+")

# Cheap gate before the period search: no periodic span of the sizes acted on here leaves a page this
# incompressible.
_GATE_COMPRESSION_RATIO = 0.42
# zlib runs on at most this many characters.
_GATE_SAMPLE_CHARS = 8000

# Length of the probe used to walk the cut back to where the repeated unit *first* appears.
_ONSET_PROBE_CHARS = 40
# ...but only this many periods back, so a unit with a legitimate earlier occurrence does not drag
# the cut through good text.
_ONSET_WALKBACK_PERIODS = 3


@dataclass(frozen=True)
class LoopOptions:
    """Thresholds for calling a periodic span a loop and for cutting it out.

    Every field changes the text that reaches the corpus, so the whole set belongs in the extraction
    step's ``hash_attrs``.
    """

    # Nothing shorter is worth acting on.
    min_page_chars: int = 3000
    # The span must be this long in absolute terms...
    min_loop_chars: int = 1200
    # ...and this much of the page...
    min_loop_fraction: float = 0.15
    # ...and must run to (near) the end.
    max_trailing_chars: int = 1500
    # How degenerate the digits inside a folded-only span must be. This guard is what holds precision.
    min_counter_score: float = 0.5
    # A retained prefix shorter than this is the first fragment of the loop itself, not a transcription.
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
    # Whether the span repeats exactly, without folding digits.
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
        # Deliberately not reduced to the unit's minimal period: the shorter stride is a weaker
        # constraint and walks back through legitimate rows that merely align with it.
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

    1.0 means they are all the same value or step by a constant, as a counter does. A span with almost
    no digits scores 1.0, because there folding cannot have created the periodicity.
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

    Two searches run: one on the raw text, one with digits folded. A folded-only span is scored by
    :func:`counter_score`, and the caller decides whether that score is good enough.
    """
    nchars = len(text)
    empty = PageLoop(nchars=nchars, start=nchars, end=nchars, period=0, exact=True, counter_score=1.0)
    if nchars < options.min_page_chars:
        return empty

    body = text.rstrip()
    folded = body.translate(_DIGIT_FOLD)
    # Gate on the *folded* tail: an incrementing counter does not compress until its digits are folded.
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

    A cycle that drifts was already degenerate before exact repetition began, so the first occurrence
    of the repeated unit is the better onset, bounded to a few periods back.
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
    emitted after the span. The cut snaps back to the preceding newline, and a prefix too short to be
    a transcription becomes an empty page rather than a stub.
    """
    cut = loop_onset(text, loop)
    newline = text.rfind("\n", 0, cut)
    if newline >= 0:
        cut = newline + 1
    kept = text[:cut].rstrip()
    return kept if len(kept) >= options.min_salvage_prefix else ""


def repair_page(text: str, truncated: bool, options: LoopOptions) -> PageRepair:
    """Detect a repetition loop in one page and cut it out.

    Only truncated pages are examined: a runaway cycle runs until the token cap stops it, so the cap
    is the loop's own signature.
    """
    if not truncated:
        return PageRepair(text=text, looped=False, dropped_chars=0, loop_period=0)
    loop = find_loop(text, options)
    if not is_loop(loop, options):
        return PageRepair(text=text, looped=False, dropped_chars=0, loop_period=0)
    kept = salvage(text, loop, options)
    return PageRepair(text=kept, looped=True, dropped_chars=len(text) - len(kept), loop_period=loop.period)

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Live smoke test for the intruder panel against an OpenRouter endpoint.

Two modes:

* ``--probe`` (default): one intruder trial, each panelist votes once. ~4
  API calls total -- the cheapest way to confirm every model slug, the
  OpenAI-compatible auth, the JSON parsing, and the GPT reasoning-effort
  passthrough all work.
* ``--run``: a short sequential test over two synthetic bucketings (a clearly
  coherent side vs. a shuffled incoherent side). Confirms the live panel calls
  the coherent side more coherent. Costs more (tens of calls, GPT-5.5 xhigh is
  the pricey one) -- bounded by ``--max-trials``.

Reads ``OPENROUTER_API_KEY`` and points at OpenRouter by default. Run it with
your own key so the secret stays out of the agent environment::

    ! OPENROUTER_API_KEY=sk-or-... uv run python -m experiments.datakit.intruder_live_smoke
    ! OPENROUTER_API_KEY=sk-or-... uv run python -m experiments.datakit.intruder_live_smoke --run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from openai import OpenAI

from experiments.datakit.intruder import (
    BucketPool,
    LlmPanelist,
    default_panel,
    run_intruder_test,
)

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Topic-coherent buckets: each bucket is internally about one thing, so an
# intruder drawn from a different bucket should be easy to spot.
COHERENT_BUCKETS: dict[str, list[str]] = {
    "python": [
        "Use a list comprehension to filter even numbers: [n for n in xs if n % 2 == 0].",
        "The with-statement guarantees the file handle is closed even if an exception is raised.",
        "Decorators wrap a function, returning a new callable that adds behavior before or after.",
        "A generator yields values lazily; it keeps its local state between successive next() calls.",
        "Type hints are not enforced at runtime but power static checkers like mypy and pyrefly.",
        "Virtual environments isolate a project's dependencies from the system Python install.",
    ],
    "cooking": [
        "Sear the steak on high heat to build a browned crust, then rest it before slicing.",
        "Whisk egg yolks with sugar until pale, then temper in the hot milk to avoid scrambling.",
        "Salt the pasta water generously; it seasons the noodles from the inside as they cook.",
        "Let bread dough proof until doubled, then knock it back to redistribute the yeast.",
        "Deglaze the pan with wine to lift the fond into a quick pan sauce.",
        "Toast whole spices in a dry skillet to bloom their aromatic oils before grinding.",
    ],
    "astronomy": [
        "A light-year measures distance, not time: how far light travels in one year.",
        "Red dwarfs are the most common stars, burning slowly over trillions of years.",
        "A solar eclipse occurs when the Moon passes directly between Earth and the Sun.",
        "Spectral lines reveal a star's composition by which wavelengths it absorbs.",
        "The habitable zone is the orbital band where liquid water can persist on a surface.",
        "Neutron stars pack more than a solar mass into a sphere a dozen kilometers across.",
    ],
    "law": [
        "Consideration is the bargained-for exchange that makes a promise an enforceable contract.",
        "The burden of proof in a criminal trial rests on the prosecution beyond reasonable doubt.",
        "Stare decisis binds lower courts to the precedents set by higher courts in the hierarchy.",
        "A tort is a civil wrong for which a court can impose liability and award damages.",
        "Mens rea, the guilty mind, is the mental element most crimes require alongside the act.",
        "An affidavit is a written statement of fact sworn under oath before an authorized officer.",
    ],
}


def _openrouter_panel() -> list[LlmPanelist]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY not set. Run with `! OPENROUTER_API_KEY=sk-or-... uv run python -m ...`")
    return default_panel(OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key))


def _probe(panel: list[LlmPanelist]) -> None:
    """One trial, every panelist votes once; print pick vs. truth and any error."""
    import numpy as np  # noqa: PLC0415  -- local to keep the import out of --run callers' path

    pool = BucketPool("coherent", COHERENT_BUCKETS)
    trial = pool.sample_trial(np.random.default_rng(0))
    print(
        f"Trial: in-group={trial.in_group_bucket!r} intruder={trial.intruder_bucket!r} "
        f"true_intruder_index={trial.intruder_index + 1} (1-based)\n"
    )
    for p in panel:
        try:
            pick = p.vote(trial, max_doc_chars=2000)
            mark = "correct" if pick == trial.intruder_index else "WRONG"
            print(f"  {p.name:34} -> pick {pick + 1}  [{mark}]")
        except Exception as e:
            print(f"  {p.name:34} -> ERROR  {type(e).__name__}: {str(e)[:160]}")


def _run(panel: list[LlmPanelist], max_trials: int) -> None:
    """Coherent side vs. shuffled-incoherent side; expect coherent to win."""
    import numpy as np  # noqa: PLC0415

    # Incoherent side: same docs, reshuffled across buckets so no bucket shares a topic.
    all_docs = [d for docs in COHERENT_BUCKETS.values() for d in docs]
    rng = np.random.default_rng(0)
    rng.shuffle(all_docs)
    n_buckets = len(COHERENT_BUCKETS)
    incoherent = {f"mix{i}": all_docs[i::n_buckets] for i in range(n_buckets)}

    result = run_intruder_test(
        lhs=COHERENT_BUCKETS,
        rhs=incoherent,
        panel=panel,
        lhs_name="coherent",
        rhs_name="incoherent",
        min_trials=8,
        max_trials=max_trials,
        batch_size=4,
        seed=0,
    )
    print(f"\ndecision: {result.decision}")
    print(f"  coherent   acc={result.lhs_accuracy:.3f} {tuple(round(x, 3) for x in result.lhs_interval)}")
    print(f"  incoherent acc={result.rhs_accuracy:.3f} {tuple(round(x, 3) for x in result.rhs_interval)}")
    print(f"  diff interval: {tuple(round(x, 3) for x in result.difference_interval)}")
    print(f"  trials/side: {result.n_trials_per_side}  abstained: {result.n_abstained}  chance: {result.chance_level}")
    print("  per-model accuracy (coherent / incoherent):")
    for name, acc in result.per_model_accuracy.items():
        print(f"    {name:34} {acc['coherent']:.3f} / {acc['incoherent']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Run the short statistical test instead of the cheap probe.")
    parser.add_argument("--max-trials", type=int, default=24, help="Per-side trial cap for --run (default: 24).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    panel = _openrouter_panel()
    print(f"panel: {[p.name for p in panel]}\n")
    if args.run:
        _run(panel, args.max_trials)
    else:
        _probe(panel)


if __name__ == "__main__":
    main()

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The TPP-10 domain sweeps of the epoch-cap figure (`fig:domain-repetition`): one row per (training run, evaluation).

Each sweep mixes one domain with the unrepeated six-part Nemotron-CC web blend at domain weights 0, 0.05, 0.1, 0.2,
0.3, 0.5, 0.7 and 1. The domains are Dolmino FLAN (`instruction`), Dolmino's Wikipedia reading-comprehension QA
(`rcqa`), Common Pile arXiv papers (`arxiv`) and FineMath-3+ (`finemath`). The proxy has 16,587,008 parameters and
trains on 165,937,152 tokens; the target has 301,241,344 parameters and trains on 3,012,296,704 tokens (10 tokens per
parameter). Each domain contributes a 190,316,544-token pool. The target draws the domain from the whole pool. The
simulated-epoching (matched) proxy draws it from a fixed 10,485,760-token subset (subset seed 20260912), so its epochs
match the target's at every weight. The unmatched proxy draws from the whole pool and stays below one epoch. The figure
draws the matched proxy and the target of FLAN on SciQ (panel A) and of FineMath on GSM8K (panel B). The instruction
and FineMath sweeps have all three arms; the rcQA and arXiv sweeps have the matched proxy only.

- `epochs_per_unit_weight` is the run's tokens over its domain pool: the epochs the domain would see at weight 1 at
  the run's scale, as in the `partition` table. It is 15.828 for the target, 15.825 for the matched proxy and 0.872 for
  the unmatched proxy.
- `domain_weight` is the requested weight. `domain_epochs` are the realized epochs: the domain sequences that the
  allocator draws (the design's allocation audit) over the pool's sequences. The allocator fills 2,048-sequence blocks,
  which puts them within 0.44% of domain_weight x epochs_per_unit_weight. The sweeps' analyses and the FineMath math
  receipt record the same epochs for the matched proxies and the targets. The figure's builder takes the unmatched
  proxies' epochs as domain_weight x epochs_per_unit_weight.
- FineMath's weight-0 runs are the original TPP-10 experiment's web-only runs, one per scale. A run without domain data
  serves FineMath's matched and unmatched proxy curves alike; these keep their launch record's whole pool (the proxy's
  epochs_per_unit_weight is 0.872). The other sweeps retrained their weight-0 runs, whose evaluations reproduce one
  another.

`tokenizer` names the Hugging Face tokenizer that the TPP-10 design pins for every run,
`TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` (32,000 tokens). The `tasks` table's tokenizer describes the
paper's swarms and does not apply to these runs. Every evaluation is the final step's bits per byte under this
tokenizer, pooling the scored bits and bytes of each set (Levanter's BPB schema version 2).

- The seven Uncheatable components score the paper's 1-14 September 2025 snapshot and carry the paper's task ids
  (`naming.task_public`). `macro_bpb` is their equal-weight mean, the sweeps' selection metric; the paper's Uncheatable
  objective weights the components by bytes. `paloma_bpb` is Paloma's programming-languages set.
- `arc_easy_bpb`, `arc_challenge_bpb`, `openbookqa_bpb`, `qasc_bpb` and `sciq_bpb` score each validation example as one
  document, `Question: <question>\\nAnswer: <answer>`. They differ from the paper's OlmoBaseEval tasks, which score
  only the answer continuation.
- `dolmino_flan_heldout_bpb` and `wiki_to_rcqa_heldout_bpb` score 4,194,304 tokens of a held-out shard of the domain;
  `s2orc_bpb` is Paloma's `m2d2_s2orc_unsplit`.
- `gsm8k_bpb` (1,319 test problems) and `math500_bpb` (500 problems) score the reference solutions only. For the
  unmatched FineMath proxies, the source converts token loss to BPB with the shared web-only run's ratio of the two.

Every run has trainer seed 20260910 and the launcher's data seed 20260910. `subset_seed` is 20260912 for the matched
proxies and null for runs that draw the domain from the whole pool. Observation ids key on the training run name (these
launches append no hash); the runs appear in no other config.

`domain_sweeps_olmix_fits` holds Olmix's log-linear fit to each matched-proxy curve that the figure draws:
BPB(p) = exp(log_c) + exp(coefficient_domain * p + coefficient_web * (1 - p)) at domain weight p, fitted to the
curve's eight points under Olmix's Huber loss (delta 0.02). For each curve, the figure's v03 builder kept the
lowest-loss fit among Olmix's 48-start fit and two refits restricted to a falling and a rising response. The fits exist
only in that builder's receipt on Drive.

`domain_sweeps_sources.json` lists, by domain, each file the export reads and each other file that the figure's
builder chain reads (`figure4_cap_policy_drafts` in the paper folder: v16_full_width -> v11_options -> v03 ->
plot_alternatives), with its original and the names the export takes from it.
"""

import json

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.release import naming, paths
from experiments.domain_phase_mix.release.context import ExportContext

SOURCES = paths.REPO / "experiments/domain_phase_mix/release/configs/domain_sweeps_sources.json"
DESIGN = paths.REPO / "experiments/domain_phase_mix/starcoder_tpp10_assets/design.json"
ALLOCATIONS = paths.AG / "starcoder_tpp10/allocation_audit.json"
TABLE, FITS_TABLE = "domain_sweeps", "domain_sweeps_olmix_fits"
COLUMNS = [
    "observation_id",
    "domain",
    "scale",
    "model_total_parameters",
    "tokenizer",
    "train_tokens",
    "epochs_per_unit_weight",
    "domain_weight",
    "domain_epochs",
    "evaluation",
    "bpb",
    "trainer_seed",
    "subset_seed",
]
FIT_COLUMNS = [
    "domain",
    "scale",
    "epochs_per_unit_weight",
    "evaluation",
    "log_c",
    "coefficient_domain",
    "coefficient_web",
    "huber_loss",
]
# The fields a source's columns map onto; `source_domain` is the source's own name for the domain.
RUN_FIELDS = ["run_name", "source_domain", "arm", "percent", "trainer_seed", "subset_seed"]
# Fields a source may also record; they must equal the values the export derives from the design.
CHECKED_FIELDS = ["train_tokens", "pool_sequences", "domain_epochs"]
SEQUENCE_LENGTH = 2048
# Launch arm -> scale, the design's name for its model, and the design's size of its domain pool.
ARMS = {
    "matched": ("proxy", "unmatched", "matched_sequences"),
    "unmatched": ("proxy", "unmatched", "parent_sequences"),
    "target": ("target", "target", "parent_sequences"),
}
ALLOCATION_KEYS = {"proxy": "proxy_allocation", "target": "target_allocation"}
DOMAIN_COMPONENT = "starcoder"  # the allocator's name for the swept domain in every sweep
PAPER_TASK_METRICS = frozenset(naming.task_table().internal_metric)


def _arm_designs(design: dict) -> pd.DataFrame:
    """Scale, model size, token budget, domain pool and epochs at weight 1 of each launch arm."""
    rows = []
    for arm, (scale, model, pool) in ARMS.items():
        tokens = design["models"][model]["tokens"]
        rows.append(
            {
                "arm": arm,
                "scale": scale,
                "model_total_parameters": design["models"][model]["parameters"],
                "train_tokens": tokens,
                "pool_sequences": design[pool],
                "epochs_per_unit_weight": tokens / (design[pool] * SEQUENCE_LENGTH),
            }
        )
    return pd.DataFrame(rows)


def _domain_sequences(design: dict) -> pd.DataFrame:
    """The sequences of the swept domain that the allocator draws at each audited weight and scale."""
    audit = json.loads(paths.require(ALLOCATIONS).read_text())
    if audit["design_sha256"] != design["design_sha256"]:
        raise ValueError(f"{ALLOCATIONS.name} audits another design than {DESIGN.name}")
    return pd.DataFrame(
        [
            {"scale": scale, "percent": point["percent"], "domain_sequences": point[key][DOMAIN_COMPONENT]}
            for point in audit["coordinates"]
            for scale, key in ALLOCATION_KEYS.items()
        ]
    )


def _source_frame(entry: dict) -> pd.DataFrame:
    """A source file's runs, one row each, under the source's own column names."""
    path = paths.require(paths.REPO / entry["original"])
    if entry["reader"] == "measurements":
        # pandas' default float parser misreads some 17-digit values by one ulp; the figure's builder parses exactly.
        return pd.read_csv(path, float_precision="round_trip")
    if entry["reader"] == "math_receipt":
        rows = json.loads(path.read_text())["rows"]
        return pd.json_normalize([row for group in entry["groups"] for row in rows[group]])
    raise ValueError(f"unknown reader {entry['reader']!r} for {entry['original']}")


def _long(entry: dict) -> pd.DataFrame:
    """Each source is wide per run (one column per evaluation); melt it to one row per (run, evaluation), keeping the
    run fields the sources file maps its columns to."""
    frame = _source_frame(entry)
    names = {**entry["columns"], **entry["evaluations"]}
    absent = sorted(set(names) - set(frame.columns))
    if absent:
        raise ValueError(f"{entry['original']} lacks the mapped columns {absent}")
    frame = frame[list(names)].rename(columns=names)
    unmapped = sorted(set(RUN_FIELDS) - set(frame.columns))
    if unmapped:
        raise ValueError(f"the sources file maps no column of {entry['original']} to {unmapped}")
    if not frame.source_domain.eq(entry["source_domain"]).all():
        raise ValueError(f"{entry['original']} holds runs of another domain than {entry['source_domain']}")
    fields = [c for c in frame.columns if c in RUN_FIELDS or c in CHECKED_FIELDS]
    long = frame.melt(
        id_vars=fields,
        value_vars=list(entry["evaluations"].values()),
        var_name="evaluation",
        value_name="bpb",
        ignore_index=False,
    )
    return long.sort_index(kind="stable")  # run by run, evaluations in the sources file's order


def _public_evaluation(name: str) -> str:
    """The paper's task id for one of its 58 tasks (named by its internal metric), else the source's own name."""
    return naming.task_public(name) if name in PAPER_TASK_METRICS else name


def _olmix_fit(entry: dict, domain: str, arms: pd.DataFrame) -> dict:
    """Olmix's fit to one matched-proxy curve, as the figure's builder chain reads it from the v03 receipt on Drive."""
    receipt = json.loads(paths.require(paths.DRIVE / entry["drive_copy"]).read_text())
    fit = receipt["analysis"]["fit_audit"][f"{entry['source_domain']}/{entry['arm']}"]["selected"]
    arm = arms.set_index("arm").loc[entry["arm"]]
    row = {
        "domain": domain,
        "scale": arm.scale,
        "epochs_per_unit_weight": arm.epochs_per_unit_weight,
        "evaluation": entry["evaluation"],
    }
    for field, columns in entry["columns"].items():
        row.update(zip(columns, fit[field], strict=True) if isinstance(columns, list) else {columns: fit[field]})
    return row


def export(ctx: ExportContext) -> dict[str, pd.DataFrame]:
    sources = json.loads(paths.require(SOURCES).read_text())
    design = json.loads(paths.require(DESIGN).read_text())
    arms = _arm_designs(design)
    parts, fits = [], []
    for domain, entries in sources.items():
        for entry in entries:
            if entry["reader"] == "olmix_fits":
                fits.append(_olmix_fit(entry, domain, arms))
            elif entry["reader"] is not None:
                parts.append(_long(entry).assign(domain=domain))
    df = pd.concat(parts, ignore_index=True).rename(columns={c: f"{c}_recorded" for c in CHECKED_FIELDS})
    df = df.merge(arms, on="arm", how="left", validate="many_to_one")
    if df.scale.isna().any():
        raise ValueError(f"unknown launch arms {sorted(set(df.arm[df.scale.isna()]))}")
    df = df.merge(_domain_sequences(design), on=["scale", "percent"], how="left", validate="many_to_one")
    if df.domain_sequences.isna().any():
        raise ValueError(f"weights off the audited grid: {sorted(set(df.percent[df.domain_sequences.isna()]))}")
    df["domain_epochs"] = df.domain_sequences / df.pool_sequences
    for field in CHECKED_FIELDS:
        recorded = df.get(f"{field}_recorded", pd.Series(np.nan, index=df.index))
        known = recorded.notna()
        if not np.allclose(recorded[known], df.loc[known, field], rtol=0, atol=1e-12):
            raise ValueError(f"a source's {field} differs from the one its arm and weight give")
    if df.bpb.isna().any():
        raise ValueError(f"missing evaluations of {sorted(set(df.run_name[df.bpb.isna()]))}")
    df["domain_weight"] = df.percent / 100
    df["tokenizer"] = design["tokenizer_pins"]["tokenizer_repo"]
    df["observation_id"] = df.run_name.map(ctx.registry.id_for)
    df["evaluation"] = df.evaluation.map(_public_evaluation)
    df = df.astype({"trainer_seed": "int64", "subset_seed": "Int64"})
    if df.duplicated(["observation_id", "evaluation"]).any():
        raise ValueError("a run's evaluation comes from two sources")
    return {TABLE: df[COLUMNS], FITS_TABLE: pd.DataFrame(fits, columns=FIT_COLUMNS)}

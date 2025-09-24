import argparse
import os
import re
from typing import Any
from collections.abc import Iterable

# W&B can be optional at import-time in some environments; import lazily in main
try:
    from marin.utilities.wandb_utils import WANDB_ENTITY, WANDB_PROJECT
except Exception:
    WANDB_ENTITY = os.getenv("WANDB_ENTITY", "stanford-mercury")
    WANDB_PROJECT = os.getenv("WANDB_PROJECT", "marin")


MetricKey = str
RunId = str

# Target metrics
METRICS: dict[str, dict[str, Any]] = {
    "bpb": {
        "key": "eval/paloma/c4_en/bpb",
        "title": "eval/paloma/c4_en/bpb",
        "higher_is_better": False,
    },
    "macro_acc_norm": {
        "key": "lm_eval/averages/macro_avg_acc_norm",
        "title": "lm_eval/averages/macro_avg_acc_norm",
        "higher_is_better": True,
    },
    "mmlu_choice_logprob_norm": {
        "key": "lm_eval/mmlu_5shot/choice_logprob_norm",
        "title": "lm_eval/mmlu_5shot/choice_logprob_norm",
        "higher_is_better": True,  # values are typically negative; less negative (higher) is better
    },
}

# Mixture keys to parse from tags
MIXTURE_TAG_KEYS = ["mcq", "nemo_qa", "regular_text", "wrap_med", "wrap_qa"]
MIXTURE_TAG_RE = re.compile(r"^(mcq|nemo_qa|regular_text|wrap_med|wrap_qa)=(\d*\.?\d+)$")


def parse_mixture_from_tags(tags: Iterable[str]) -> dict[str, float]:
    mixture: dict[str, float] = {k: 0.0 for k in MIXTURE_TAG_KEYS}
    for t in tags:
        m = MIXTURE_TAG_RE.match(str(t))
        if not m:
            continue
        key = m.group(1)
        try:
            mixture[key] = float(m.group(2))
        except Exception:
            pass
    return mixture


def _get_summary_float(run, key: str) -> float | None:
    try:
        v = run.summary.get(key)
        return float(v) if v is not None else None
    except Exception:
        return None


def collect_regmix_summary(
    *,
    entity: str,
    project: str,
    api_key: str | None,
    require_tags: list[str],
    also_accept_tag: list[str],
    name_contains: str | None,
) -> list[dict[str, Any]]:
    """Collect finished regmix runs and return summary rows.

    Each row contains:
      - run_id, run_name
      - metrics: dict keyed by METRICS slugs (bpb, macro_acc_norm, mmlu_choice_logprob_norm)
      - mixture: dict with keys in MIXTURE_TAG_KEYS, values in [0,1]
    """
    import wandb

    if api_key:
        try:
            wandb.login(key=api_key, relogin=True)
        except Exception:
            pass

    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    rows: list[dict[str, Any]] = []
    for run in runs:
        try:
            state = (run.state or "").lower()
        except Exception:
            state = ""
        if state != "finished":
            continue

        name: str = run.name or ""
        tags_list: list[str] = list(run.tags or [])
        tags_lower = set(t.lower() for t in tags_list)
        required_ok = all(req.lower() in tags_lower for req in (require_tags or []))
        fallback_ok = ("regmix" in tags_lower) and (any(t.lower() in tags_lower for t in (also_accept_tag or [])))

        if name_contains and name_contains not in name:  # doesn't include name then don't include
            continue

        if not (required_ok or fallback_ok):
            continue

        mixture = parse_mixture_from_tags(tags_list)
        metrics_map = {
            "bpb": _get_summary_float(run, METRICS["bpb"]["key"]),
            "macro_acc_norm": _get_summary_float(run, METRICS["macro_acc_norm"]["key"]),
            "mmlu_choice_logprob_norm": _get_summary_float(run, METRICS["mmlu_choice_logprob_norm"]["key"]),
        }

        rows.append(
            {
                "run_id": run.id,
                "run_name": name or run.id,
                "mixture": mixture,  # fractions 0..1
                "metrics": metrics_map,
            }
        )

    return rows


def main():
    parser = argparse.ArgumentParser(description="Summarize regmix llama-130m runs from W&B into a sortable table.")
    parser.add_argument("--entity", default=WANDB_ENTITY, help="W&B entity")
    parser.add_argument("--project", default=WANDB_PROJECT, help="W&B project")
    parser.add_argument("--api_key", default=os.getenv("WANDB_API_KEY"), help="W&B API key; env WANDB_API_KEY if unset")
    # No local saving; only logging to W&B
    parser.add_argument(
        "--require_tags",
        action="append",
        default=["regmix", "130m"],
        help="Required tags for selecting runs (repeatable)",
    )
    parser.add_argument(
        "--also_accept_tag",
        action="append",
        default=["130m"],
        help="Fallback tags to accept if some required tags are missing (repeatable)",
    )
    parser.add_argument("--name_contains", default="llama-130m", help="Fallback: include runs whose name contains this")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("WANDB_API_KEY not provided. Pass --api_key or export WANDB_API_KEY.")

    import wandb

    rows = collect_regmix_summary(
        entity=args.entity,
        project=args.project,
        api_key=args.api_key,
        require_tags=list(args.require_tags or []),
        also_accept_tag=list(args.also_accept_tag or []),
        name_contains=args.name_contains,
    )

    if not rows:
        raise SystemExit("No runs found matching regmix + llama-130m criteria.")

    # Build summary-only table (fast)

    # Prepare W&B run for logging plots and tables
    plot_run = wandb.init(
        project="marin-analysis",
        entity="marin-community",
        name="regmix-llama130m-plots",
        resume="never",
    )

    # Single sortable table across runs (based on summary values)
    columns = [
        "run_name",
        "run_id",
        METRICS["bpb"]["key"],
        METRICS["macro_acc_norm"]["key"],
        METRICS["mmlu_choice_logprob_norm"]["key"],
        *MIXTURE_TAG_KEYS,
    ]
    table = wandb.Table(columns=columns)

    for r in rows:
        name = r["run_name"]
        run_id = r["run_id"]
        mixture = r["mixture"]
        metrics_map = r["metrics"]
        row = [
            name,
            run_id,
            metrics_map.get("bpb"),
            metrics_map.get("macro_acc_norm"),
            metrics_map.get("mmlu_choice_logprob_norm"),
            *[mixture.get(k, 0.0) * 100.0 for k in MIXTURE_TAG_KEYS],
        ]
        table.add_data(*row)

    wandb.log({"regmix_runs_table": table})
    print(f"Logged plots to W&B run: {plot_run.url}")


if __name__ == "__main__":
    main()

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze the Mochi loop-guards evaluation panel (#12264) into ``cohorts/manifest.json``.

The manifest pins the four cohorts' sources, protocols, and frozen task/row sets that
``vendor/skyrl/scripts/loop_audit.py`` consumes via ``--manifest``. The swesmith-ho rows double
as the I3 RL training exclusions: the held-out 128 rows must stay out of the training pool.

Run on CPU::

    uv run python experiments/post_training/mochi_loop_guards/panel.py
"""

import hashlib
import io
import json
import logging
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

logger = logging.getLogger(__name__)

EXPERIMENT = "mochi-loop-guards (#12264)"
PANEL_VERSION = "2026.11.25.1"
CHECKPOINT_BASELINE = (
    "s3://marin-us-east-08a/marin/models/mochi-sft/hero-sft-fullab-step393k-v2/step-400407/2af3f828aa-cpu-a1"
)

SEED = 20261125
SWESMITH_SELECTED_ROWS = 128
SWESMITH_EXPECTED_SOURCE_ROWS = 5608
SWESMITH_SOURCE_URI = (
    "s3://marin-us-east-02a/marin/rl-data/inputs/swe-verified-task-spec-setup/2026.11.19.1/agentic-train.parquet"
)
SWESMITH_MANIFEST_URI = (
    "s3://marin-us-east-02a/marin/rl-data/inputs/swe-verified-task-spec-setup/2026.11.19.1/manifest.json"
)

SELECTION_RULE = (
    "Seeded-hash ranking: rank rows ascending by hashlib.sha256(f'{seed}:{row_hash}').hexdigest() and keep"
    " the first 128, where row_hash is the SHA-256 hex digest of the parquet row's canonical JSON"
    " serialization json.dumps(row, sort_keys=True, separators=(',', ':'), default=str).encode()."
)
EXCLUSION_REASON = "held-out cohort swesmith-ho; excluded from the I3 training pool"

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPO_ROOT / "cohorts" / "manifest.json"

# Pinned protocol constants of the Terminus-2 cohorts, from experiments/evaluation/benchmarks/agentic.py
# (CAMPAIGN_JOB, CAMPAIGN_MAX_INPUT_TOKENS, TERMINAL_BENCH_ATTEMPT_TIMEOUT) and
# experiments/evaluation/benchmarks/harbor.py (MAX_OUTPUT_TOKENS, MODEL_TURN_TIMEOUT, ATTEMPT_TIMEOUT).
TERMINUS_AGENT_TIMEOUT_SECONDS = 1800
TERMINUS_VERIFIER_TIMEOUT_SECONDS = 300
TERMINUS_MAX_INPUT_TOKENS = 32768
HARBOR_MAX_OUTPUT_TOKENS = 8192
HARBOR_MODEL_TURN_TIMEOUT_SECONDS = 600
HARBOR_ATTEMPT_TIMEOUT_SECONDS = 4 * 60 * 60
TERMINAL_BENCH_ATTEMPT_TIMEOUT_SECONDS = 8 * 60 * 60
TERMINUS_SAMPLING = {
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "max_tokens": HARBOR_MAX_OUTPUT_TOKENS,
    "stop": [],
    "chat_template_kwargs": {},
}

# The swesmith-ho cohort runs the RL harness: experiments/post_training/mochi_rl.py `_AGENTIC_LIMITS`
# and experiments/post_training/rl_agentic_recipe.py `_AGENTIC_RECIPE` over
# experiments/post_training/task_curation/swesmith_training.py constants.
SWESMITH_PI_VERSION = "0.87.0"
SWESMITH_TRIAL_TIMEOUT_SECONDS = 14_400
SWESMITH_AGENT_TIMEOUT_SECONDS = 10_800
SWESMITH_STALL_TIMEOUT_SECONDS = 28_800
SWESMITH_VERIFIER_TIMEOUT_SECONDS = 900
SWESMITH_MEMORY_MB = 4096
SWESMITH_DAYTONA_TTL_MINUTES = 240
SWESMITH_MAX_TURNS = 128
SWESMITH_DATASET = "long-loop-smartie-swesmith"
SWESMITH_MACHINE = {
    "backend": "daytona",
    "network": "allow",
    "gpus": 0,
    "user": "longloop-agent",
    "startup_timeout_seconds": 1800,
    "cleanup_timeout_seconds": 120,
}

# The frozen tail-loop detector parameters loop_audit.py reads from every cohort
# (LoopCreditConfig(4096, 32768, 8), vendor/skyrl/scripts/loop_audit.py).
LOOP_MAX_PERIOD_TOKENS = 4096
LOOP_BUDGET_TOKENS = 32768
LOOP_MINIMUM_OCCURRENCES = 8


class Harness(StrEnum):
    """The agent harness that runs a cohort."""

    TERMINUS_2 = "terminus-2"
    PI_0_87_0 = "pi-0.87.0"


class CohortName(StrEnum):
    """The panel's four evaluation cohorts."""

    TB2 = "tb2"
    TBLITE = "tblite"
    SWE100 = "swe100"
    SWESMITH_HO = "swesmith-ho"


def pinned_task_rows(repo: str, revision: str, count: int) -> list[dict]:
    """List a pinned Harbor source's tasks as source-order placeholders.

    The task IDs inside the pinned package require lowering the Harbor packages, so the frozen
    manifest records source order positions until the cohort runs; ``loop_audit`` matches the
    dumped-evals rows by the task ID the run assigns.
    """
    return [{"task_id": f"{repo}@{revision}:{index}", "prompt_hash": None} for index in range(count)]


def _row_hash(row: dict) -> str:
    """Hash one parquet row's canonical JSON serialization."""
    canonical = json.dumps(row, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(canonical).hexdigest()


def _selection_rank(row_hash: str) -> str:
    """The seeded rank key of one row hash."""
    return hashlib.sha256(f"{SEED}:{row_hash}".encode()).hexdigest()


def select_swesmith_rows() -> tuple[list[dict], dict]:
    """Select the held-out swesmith-ho rows from the pinned TaskSpec parquet.

    Reads the parquet fully, verifies its digest against the release manifest, ranks every row by
    the seeded hash, and keeps the first 128. Returns the selected rows and the cohort's source
    identity.
    """
    configure_coreweave_s3()
    manifest = json.loads(StoragePath(SWESMITH_MANIFEST_URI).read_text())
    parquet_bytes = StoragePath(SWESMITH_SOURCE_URI).read_bytes()
    parquet_sha256 = hashlib.sha256(parquet_bytes).hexdigest()
    if parquet_sha256 != manifest["output_sha256"]:
        raise ValueError(f"agentic-train.parquet digest mismatch: {parquet_sha256} != {manifest['output_sha256']}")
    table = pq.read_table(io.BytesIO(parquet_bytes))
    del parquet_bytes
    rows = table.to_pylist()
    if len(rows) != SWESMITH_EXPECTED_SOURCE_ROWS:
        raise ValueError(f"Expected {SWESMITH_EXPECTED_SOURCE_ROWS} source rows, found {len(rows)}")
    if len(manifest["tasks"]) != len(rows):
        raise ValueError(f"Release manifest lists {len(manifest['tasks'])} tasks for {len(rows)} rows")

    hashed = []
    revisions = set()
    for index, row in enumerate(rows):
        spec = json.loads(row["task_spec"])
        if spec["id"] != manifest["tasks"][index]["task_id"]:
            raise ValueError(f"Row {index} task id differs from the release manifest: {spec['id']}")
        revisions.add(spec["source"]["revision"])
        hashed.append({"row_hash": _row_hash(row), "task_id": spec["id"], "index": index})
    if len(revisions) != 1:
        raise ValueError(f"Expected one source revision across rows, found {sorted(revisions)}")

    selected = sorted(hashed, key=_selection_rank)[:SWESMITH_SELECTED_ROWS]
    selected = sorted(selected, key=lambda row: row["row_hash"])
    if len({row["task_id"] for row in selected}) != SWESMITH_SELECTED_ROWS:
        raise ValueError("The selected rows contain duplicate task ids")
    source = {
        "kind": "task-spec-parquet",
        "uri": SWESMITH_SOURCE_URI,
        "sha256": parquet_sha256,
        "row_count": len(rows),
        "task_spec_column": "task_spec",
        "dataset": SWESMITH_DATASET,
        "source_revision": revisions.pop(),
        "release_manifest": SWESMITH_MANIFEST_URI,
        "selection_rule": SELECTION_RULE,
        "seed": SEED,
        "selected_rows": SWESMITH_SELECTED_ROWS,
    }
    return selected, source


def build_cohorts(swesmith_rows: list[dict], swesmith_source: dict) -> dict:
    """Build the four cohort entries with their pinned sources, protocols, and rows."""
    terminus_protocol = {
        "agent_timeout_seconds": TERMINUS_AGENT_TIMEOUT_SECONDS,
        "verifier_timeout_seconds": TERMINUS_VERIFIER_TIMEOUT_SECONDS,
        "max_input_tokens": TERMINUS_MAX_INPUT_TOKENS,
        "max_output_tokens": HARBOR_MAX_OUTPUT_TOKENS,
        "model_turn_timeout_seconds": HARBOR_MODEL_TURN_TIMEOUT_SECONDS,
        "sampling": TERMINUS_SAMPLING,
    }
    swesmith_protocol = {
        "pi_version": SWESMITH_PI_VERSION,
        "thinking": "high",
        "thinking_format": "chat-template",
        "setup_timeout_seconds": 360,
        "trial_timeout_seconds": SWESMITH_TRIAL_TIMEOUT_SECONDS,
        "agent_timeout_seconds": SWESMITH_AGENT_TIMEOUT_SECONDS,
        "stall_timeout_seconds": SWESMITH_STALL_TIMEOUT_SECONDS,
        "verifier_timeout_seconds": SWESMITH_VERIFIER_TIMEOUT_SECONDS,
        "memory_mb": SWESMITH_MEMORY_MB,
        "daytona_ttl_minutes": SWESMITH_DAYTONA_TTL_MINUTES,
        "max_turns": SWESMITH_MAX_TURNS,
        "task_machine": SWESMITH_MACHINE,
        "dataset": SWESMITH_DATASET,
        "sampling": "pi-own",
        "sampling_note": "Pi chooses its own sampling and output limits; the protocol sets none",
    }
    loop_fields = {
        "max_period_tokens": LOOP_MAX_PERIOD_TOKENS,
        "budget": LOOP_BUDGET_TOKENS,
        "minimum_occurrences": LOOP_MINIMUM_OCCURRENCES,
    }
    pinned = {
        CohortName.TB2: (
            "DCAgent2/terminal_bench_2",
            "693231ec029249e7c91ed2e414bcc9c45d7cd879",
            89,
            3,
            TERMINAL_BENCH_ATTEMPT_TIMEOUT_SECONDS,
        ),
        CohortName.TBLITE: (
            "DCAgent/dev_set_v2",
            "377118ff3031c934f5a647ae2c425eb74eef3b21",
            100,
            2,
            HARBOR_ATTEMPT_TIMEOUT_SECONDS,
        ),
        CohortName.SWE100: (
            "DCAgent2/swebench-verified-random-100-folders",
            "0c553bd6d05d451907afb8db31dba84759ccc993",
            100,
            1,
            HARBOR_ATTEMPT_TIMEOUT_SECONDS,
        ),
    }
    recipes = {
        CohortName.TB2: "terminal-bench-2.0",
        CohortName.TBLITE: "ot-tblite",
        CohortName.SWE100: "swebench-verified-100",
    }
    cohorts = {}
    for name, (repo, revision, tasks, trials_per_task, attempt_timeout) in pinned.items():
        cohorts[name] = {
            **loop_fields,
            "tasks": tasks,
            "trials_per_task": trials_per_task,
            "trials": tasks * trials_per_task,
            "harness": Harness.TERMINUS_2,
            "interrupt": name in (CohortName.TB2, CohortName.TBLITE),
            "task_selection": "pinned-source-order",
            "source": {
                "kind": "huggingface-packages",
                "repo": repo,
                "revision": revision,
                "recipe": recipes[name],
                "agentic_version": "2026.11.17.2",
                "count": tasks,
            },
            "protocol": {**terminus_protocol, "attempt_timeout_seconds": attempt_timeout},
            "rows": pinned_task_rows(repo, revision, tasks),
        }
    cohorts[CohortName.SWESMITH_HO] = {
        **loop_fields,
        "tasks": len(swesmith_rows),
        "trials_per_task": 2,
        "trials": len(swesmith_rows) * 2,
        "harness": Harness.PI_0_87_0,
        "interrupt": False,
        "task_selection": "seeded-hash",
        "source": swesmith_source,
        "protocol": swesmith_protocol,
        "rows": swesmith_rows,
    }
    return cohorts


UNVERIFIED_RESOLUTIONS = {
    "trajectory_retention_format": (
        "RESOLVED from vendor/skyrl code. `trajectory_retention {phases: [eval], required: true}` writes, under"
        " the run's retention output_path (trainer.py appends '/validation' for the eval runner),"
        " `schema_v7/archives/phase=eval/step={global_step:08d}|step=eval_only/{sha256(zip)}.zip` plus a"
        " `_retention_ledger.json` at the output_path root (trajectory_retention.py:54-59, 1110-1118). Each zip"
        " holds `records/{record_id}.json.gz` entries (gzip of canonical JSON) and a `manifest.json`"
        " (trajectory_retention.py:511-536). Each record is a TrajectoryRecord schema_version=7"
        " (trajectory_retention.py:53, 206-224) with trajectory identity {instance_id, repetition_id,"
        " row_indices, environment_class, environment_extras}, prompt {messages, token_ids}, response"
        " {messages, text, token_ids, loss_mask, trainable_spans, step_boundaries[{row_index, prompt_token_ids,"
        " token_start, token_end, stop_reason, is_last_step}], stop_reason, generation_limit, loop_spans}, reward"
        " {outcome, shaped, components}, disposition {exception_type, error_treatment, server_error}, verifier,"
        " verification_result, metrics, provenance {runner, sampling, model_path, model_version_step, ...}."
        " Per-step action/observation text lives in response.messages, populated from the batch's"
        " evidence_messages of the final step row (the parsed multi-turn conversation: tool calls, observations,"
        " final answer) with the decoded assistant text as fallback (trajectory_retention.py:442-461);"
        " step_boundaries gives each step's token span. required:true blocks the run: initialization,"
        " re-initialization, and publication failures raise TrajectoryRetentionPublicationError/Timeout"
        " (trajectory_retention.py:812-814, 877-885, 891-894) and retain_trajectories re-raises any retention"
        " error when required (1140-1145), which exits the rollout worker process. This differs from the"
        " dumped_evals JSONL rows (trainer_utils.py:266-352, evaluate.py:77-80 and 345-383): when"
        " trainer.dump_eval_results is on, every eval row is dumped flat to"
        " `{export_path}/dumped_evals/{attempt_id}/{eval_only|global_step_N_evals}/{data_source}.jsonl` (+"
        " aggregated_results.jsonl) with fields uid, attempt_id, step, ordinal, is_last_step, task_id,"
        " prompt_token_ids, response_token_ids, input_prompt, output_response, messages, score, stop_reason,"
        " exception_type, error_treatment, server_error, env_class, subset_id, data_source, env_metrics, and a"
        " trimmed verification object - no selection, no ledger, no tokens' loss masks or reward shaping. The"
        " Micro210 policy pins trajectory_retention {phases: [eval], required: true, eval_sample_fraction: 1.0,"
        " sample_count_per_step: 0}, so every eval record is retained via the fraction rule"
        " (trajectory_retention.py:642-644) and count sampling is off."
    ),
    "micro210_tb2_tblite_timeouts": (
        "RESOLVED against the pinned policy.json (hf://datasets/open-athena/micro210-policy@"
        " fa38e3eacc25137c9febe898119f08d65cc5ce88/policy.json; downloaded sha256"
        " 8f7ef8170fbef2880214cbf03f70d80cffa3114276e024389617777bf07f53e5, 121292 bytes, matching"
        " experiments/evaluation/micro210_policy.py POLICY_DIGEST/byte_size). The TB2 row is benchmark"
        " 'tb2-recovery' and the OT-TBLite row is 'ot-tblite-recovery'; each runs 10 tasks"
        " (evaluation.tasks_per_benchmark=10) x 3 attempts (evaluation.attempts_per_selected_task) = 30 attempts."
        " Agent timeout IS 1800 s: evaluation.config.terminal_bench.harbor.eval_timeout_override_sec=1800 and"
        " override_timeout_sec=1800, and installed_agent.run_timeout=1800 - equal to CAMPAIGN_JOB.agent_timeout."
        " Verifier timeout is NOT 300 s: harbor.verifier_override_timeout_sec=null, so no verifier override is"
        " set and each prepared task keeps its own verifier timeout; the only 300 in the policy is the LLM"
        " judge's request_timeout=300.0 (evaluation.judge), not a task verifier. The rows also differ from the"
        " campaign TB2 eval protocol elsewhere: harbor.trial_attempt_timeout_sec=10800 (3 h, vs the panel's"
        " TERMINAL_BENCH_ATTEMPT_TIMEOUT 28800) and context_budget.request_window_tokens=65536 with"
        " max_new_tokens_per_turn=16384, max_turns=1024 (Pi model_info 65536/16384, compaction_reserve_tokens"
        " 16384, version 0.87.0, thinking high, thinking_format chat-template, override_setup_timeout_sec 360,"
        " n_concurrent_trials=8, max_retries=0). Sampling: generator.eval_sampling_params temperature=0.7,"
        " top_p=0.95, stop_token_ids=[128001, 128009], skip_special_tokens=false, eval_n_samples_per_prompt=1."
        " So the DESIGN's 'same settings' claim holds for the agent timeout only."
    ),
    "swe_smith_screen_469_attempts": (
        "PARTIALLY RESOLVED; the exact storage path cannot be confirmed from the repository or the issues alone."
        " What the receipts establish (issue 11536 comment 6941306305, issue 11723 section 2.3): the screen ran"
        " Mochi SFT step 400407 on the same 128 SWE-smith tasks x 8 tries as the Snowball pass-rate screen"
        " (run 43604, 'grug-datakit-sft-262k-20261005-ac60d940-r1'), Pi 0.87.0 with the nested edit tool, one"
        " DP16 GB200 serve in cw-us-east-08a, 32 concurrent trials, a 900 s agent cap, and no retries; it was"
        " value-stopped after the second interim look and cancelled about 06:42Z with 469 of 1,024 attempts"
        " recorded; two retained-trace snapshots taken about 15 minutes apart agree exactly; the traces are"
        " private and not published. Neither #11723 (body plus 13 comments), #11536 (body plus comments), nor"
        " the readout comment names the storage path. The public trace viewer app"
        " https://public.applets.marina.oa.dev/a/193c0a18-17c8-4348-9df3-765f3811a5d5/ reads run 'Pass-rate"
        " 43604' from temporary storage whose traces expire about 2026-11-28; the app's backing URI is not"
        " stated. Best query recipe: (1) read Marina app 193c0a18-17c8-4348-9df3-765f3811a5d5's release"
        " metadata/package for its backing storage prefix and list the run's attempt objects there; (2) via"
        " evalrecords, scan the record.json prefixes (gs://marin-eval-metadata/evals,"
        " s3://marin-us-east-02a/marin/evals, and the legacy runs prefixes; evalrecords/records.py"
        " DEFAULT_SCAN_PREFIXES) for a record whose model is the Mochi step-400407 export and whose evaluation"
        " is the SWE-smith screen, then read the run store with finestore ReadView(run_root):"
        " iter_rows('evaluation_attempts') (ATTEMPTS_TABLE, evalrecords/rollout_attempts.py:11) filtered by"
        " dataset/task_id/trial, plus the 'rollouts_v1' archive table (ARCHIVE_ROLLOUTS_TABLE,"
        " finestore/eval.py:216) for the request/turn records; (3) if the screen driver retained SkyRL"
        " trajectory archives instead, the run's trajectory_retention output_path ledger (_retention_ledger.json)"
        " and schema_v7/archives zips hold the per-attempt records. Confirming the path needs cluster or"
        " marina-app access, not GPU work."
    ),
}


def canonical_sha256(manifest: dict) -> str:
    """Hash the manifest's canonical JSON form, excluding its own digest field."""
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def build_manifest(swesmith_rows: list[dict], swesmith_source: dict) -> dict:
    """Assemble the frozen panel manifest."""
    cohorts = build_cohorts(swesmith_rows, swesmith_source)
    total_trials = sum(cohort["trials"] for cohort in cohorts.values())
    if total_trials != 823:
        raise ValueError(f"The panel must freeze 823 trials, built {total_trials}")
    return {
        "experiment": EXPERIMENT,
        "version": PANEL_VERSION,
        "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "checkpoint_baseline": CHECKPOINT_BASELINE,
        "cohorts": cohorts,
        "total_trials": total_trials,
        "rl_training_exclusions": {
            "swesmith_ho_row_hashes": sorted(row["row_hash"] for row in swesmith_rows),
            "reason": EXCLUSION_REASON,
        },
        "unverified_resolutions": UNVERIFIED_RESOLUTIONS,
    }


def main() -> None:
    """Build and write the frozen panel manifest."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    swesmith_rows, swesmith_source = select_swesmith_rows()
    manifest = build_manifest(swesmith_rows, swesmith_source)
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    counts = {name: cohort["trials"] for name, cohort in manifest["cohorts"].items()}
    print(f"wrote {MANIFEST_PATH}")
    print(f"cohort trials: {json.dumps(counts, sort_keys=True)} total={manifest['total_trials']}")
    print(f"manifest_sha256: {manifest['manifest_sha256']}")


if __name__ == "__main__":
    main()

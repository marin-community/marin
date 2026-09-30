#!/usr/bin/env python3
"""Reconcile construct-003 run bases against ``ops/desired.json``.  DRY-RUN by default.

The operator edits ``ops/desired.json`` (which bases run, at what ``--concurrency``); this tool
does the ferrying.  Per cycle, for every base:

  * active, snapshot shows non-terminal items, no live job   -> LAUNCH with --resume under the next
    free suffix (``-v<N>``), rate-limited, recorded in the ledger;
  * all items terminal (accepted/rejected)                    -> never launched;
  * inactive with a live job                                  -> REPORT (CANCEL only with --enforce-cuts);
  * active, live, at a different --concurrency than desired   -> REPORT (RELAUNCH with --enforce-concurrency);
  * with --upgrade-before T: active, live job submitted before T, non-terminal items (held count)
    -> UPGRADE: cancel, wait until Iris no longer lists it live (<= --cancel-settle-s), relaunch with
    --resume under the next suffix at desired.json sizing.  Healthcare first, then bases with most
    capture/publication/image-review/runtime-infrastructure/held items; shares --max-launches per
    --window-min with LAUNCH (parked bases go first) and is capped by --max-upgrades-per-cycle.
    Jobs younger than --heal-guard-min (sync_healer may be mid-heal) and upgrade replacements are skipped.

Nothing is submitted or cancelled without ``--apply``.  The launch is the exact shape of the
existing relaunch scripts (sync_healer.sh, relaunch_k.sh, readd.sh, relaunch_hc.sh), submitted
from the LIVE tree.

  cd /Users/k3sc0re/openathena/marin-construct   # with CW_KEY_ID/CW_KEY_SECRET exported
  PYTHONPATH=/Users/k3sc0re/openathena/capability_env_gen-dev/scripts \\
    uv run --frozen /Users/k3sc0re/openathena/capability_env_gen-dev/ops/shard_supervisor.py --init   # write desired.json
    ... ops/shard_supervisor.py                        # dry run: what it would do and why
    ... ops/shard_supervisor.py --apply                # act (rate-limited)
    ... ops/shard_supervisor.py --apply --loop --interval 300
    ... ops/shard_supervisor.py --upgrade-before 2026-09-29T23:00:00Z          # dry run: the rollout plan
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conveyor as cv  # noqa: E402

DESIRED_PATH = cv.OPS_DIR / "desired.json"
LEDGER_PATH = cv.STATE_DIR / "supervisor-ledger.jsonl"
LAUNCH_DIR = cv.STATE_DIR / "launch"
LOCK_DIR = cv.MARIN_DIR / "capability-pipeline-staging" / ".submit.lock"
HEALER = cv.BUILD_DIR / "sync_healer.sh"
DEFERRED = cv.BUILD_DIR / "stage2-deferred.txt"
QUARANTINE = cv.BUILD_DIR / "quarantine.txt"
HOLD = cv.BUILD_DIR / "healer-hold.txt"
JOB_LEAF = "cap-construct-003-{base}-{suffix}"

PILOT_SIZING = {"concurrency": 30, "cpu": 16, "mem": "192g", "disk": "200GB"}
DEFAULT_SIZING = {"concurrency": 6, "cpu": 6, "mem": "96g", "disk": "150GB"}
HC_SIZING = {"concurrency": 30, "cpu": 16, "mem": "192g", "disk": "200GB"}
HC_BASES = ("hc1", "hc2", "hc3", "hc4")


# --------------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------------


def load_pilots(path: Path = HEALER) -> set[str]:
    """PILOT=" 068 070 ... " from sync_healer.sh, as base names."""
    text = path.read_text()
    match = re.search(r'PILOT="([^"]*)"', text)
    if not match:
        raise SystemExit(f"ERROR: no PILOT list in {path}")
    return {f"shard-{n}" for n in match.group(1).split()}


def normalize_base(token: str) -> str | None:
    token = token.strip().rstrip(",:;")
    if re.fullmatch(r"\d{1,3}", token):
        return f"shard-{int(token):03d}"
    if re.fullmatch(r"shard-\d{3}", token) or re.fullmatch(r"hc\d+", token):
        return token
    return None


def parse_base_list(path: Path) -> dict[str, str]:
    """First token of each line -> base; the rest of the line is kept as the note."""
    out: dict[str, str] = {}
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return out
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        token, _, note = line.partition(" ")
        base = normalize_base(token)
        if base:
            out[base] = note.strip() or path.name
    return out


def sizing_for(base: str, pilots: set[str], concurrency: int | None = None) -> dict[str, Any]:
    """conc/cpu/mem/disk/source/extra_args, copied from the relaunch scripts."""
    if base in HC_BASES:
        spec = dict(HC_SIZING)
        spec["source"] = f"build/construct-003/construct-src-seeded/{base}"
        spec["extra_args"] = ["--", "--acceptance-seed", "acceptance-seeds.json"]
    else:
        spec = dict(PILOT_SIZING if base in pilots else DEFAULT_SIZING)
        spec["source"] = f"build/construct-003/construct-src-003/{base}.json"
        spec["extra_args"] = []
    if concurrency is not None and concurrency != spec["concurrency"]:
        tier = PILOT_SIZING if concurrency >= 30 else DEFAULT_SIZING
        spec.update({k: tier[k] for k in ("cpu", "mem", "disk")})
        spec["concurrency"] = concurrency
    return spec


def all_bases() -> list[str]:
    shards = sorted(p.stem for p in (cv.BUILD_DIR / "construct-src-003").glob("shard-*.json"))
    return list(HC_BASES) + shards


def latest_job(jobs: list[dict[str, Any]], base: str) -> dict[str, Any] | None:
    mine = [j for j in jobs if j.get("base") == base]
    return max(mine, key=lambda j: j.get("submitted") or "", default=None)


def init_desired(jobs: list[dict[str, Any]], *, pilots: set[str], deferred: dict[str, str],
                 quarantine: dict[str, str], hold: dict[str, str], bases: list[str],
                 model: dict[str, Any] | None = None, now: datetime | None = None) -> dict[str, Any]:
    """Desired state = what runs now.  Deliberate cuts (deferred/quarantine/hold, or a last job the
    user terminated) are inactive; a base whose last job died on its own is active (it was not cut)."""
    now = now or datetime.now(UTC)
    live = cv.live_jobs_by_base(jobs)
    out: dict[str, Any] = {}
    for base in bases:
        live_jobs = live.get(base, [])
        observed_conc = None
        if model and base in model.get("bases", {}):
            sub = model["bases"][base].get("submission") or {}
            if sub.get("run_name") in {j["name"].rsplit("/", 1)[-1] for j in live_jobs}:
                observed_conc = sub.get("concurrency")
        spec = sizing_for(base, pilots, observed_conc)
        rule = sizing_for(base, pilots)
        notes = []
        if observed_conc is not None and observed_conc != rule["concurrency"]:
            notes.append(f"live job runs --concurrency {observed_conc} (script rule {rule['concurrency']})")
        if live_jobs:
            active = True
            notes.append("live at init: " + ", ".join(j["name"].rsplit("/", 1)[-1] for j in live_jobs))
            if base in deferred:
                notes.append(f"also listed in stage2-deferred.txt ({deferred[base]})")
        elif base in deferred or base in quarantine or base in hold:
            active = False
            if base in deferred:
                notes.append(f"stage2-deferred.txt: {deferred[base]}")
            if base in quarantine:
                notes.append("quarantine.txt")
            if base in hold:
                notes.append("healer-hold.txt")
        else:
            last = latest_job(jobs, base)
            if last is None:
                active = True
                notes.append("never launched; not deferred")
            elif last["state"] == "killed" and "user" in (last.get("reason") or "").lower():
                active = False
                notes.append(f"last job {last['name'].rsplit('/', 1)[-1]} was terminated by user (treated as a cut): review")
            else:
                active = True
                notes.append(f"last job {last['name'].rsplit('/', 1)[-1]} ended {last['state']} ({last.get('reason') or '-'}); not a cut")
        out[base] = {"active": active, **spec, "note": "; ".join(notes)}
    return {
        "schema": "construct-003-desired-v1",
        "run": cv.RUN,
        "generated_at": cv.iso(now),
        "comment": "Operator-owned. active=false stops the supervisor from relaunching a base; "
                   "concurrency is applied on the next launch (or at once with --enforce-concurrency).",
        "bases": out,
    }


def load_desired(path: Path = DESIRED_PATH) -> dict[str, Any]:
    try:
        desired = json.loads(path.read_text())
    except FileNotFoundError:
        raise SystemExit(f"ERROR: {path} is missing; run with --init first") from None
    except ValueError as error:
        raise SystemExit(f"ERROR: {path} is not valid JSON: {error}") from None
    bases = desired.get("bases")
    if not isinstance(bases, dict) or not bases:
        raise SystemExit(f"ERROR: {path} has no bases")
    for base, spec in bases.items():
        missing = [k for k in ("active", "concurrency", "cpu", "mem", "disk", "source") if k not in spec]
        if missing:
            raise SystemExit(f"ERROR: desired.json {base} lacks {missing}")
        if not isinstance(spec["concurrency"], int) or spec["concurrency"] < 1:
            raise SystemExit(f"ERROR: desired.json {base} concurrency must be a positive integer")
        if not isinstance(spec.get("extra_args", []), list):
            raise SystemExit(f"ERROR: desired.json {base} extra_args must be a list")
    return desired


# --------------------------------------------------------------------------------------------
# Ledger
# --------------------------------------------------------------------------------------------


def read_ledger(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or LEDGER_PATH
    entries = []
    try:
        for line in path.read_text().splitlines():
            try:
                entries.append(json.loads(line))
            except ValueError:
                continue
    except OSError:
        pass
    return entries


def append_ledger(entry: dict[str, Any], path: Path | None = None) -> None:
    path = path or LEDGER_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(entry) + "\n")


# --------------------------------------------------------------------------------------------
# Decisions
# --------------------------------------------------------------------------------------------


@dataclass
class Decision:
    base: str
    action: str  # LAUNCH, RELAUNCH, CANCEL, DEFER, SKIP, REPORT, OK
    reason: str
    run_name: str | None = None
    spec: dict[str, Any] | None = None
    jobs: list[str] = field(default_factory=list)
    priority: tuple = ()
    fingerprint: str | None = None
    queued: int | None = None
    controller: str | None = None
    replaces: str | None = None          # UPGRADE: the live job cancelled first
    unblock: dict[str, int] | None = None  # UPGRADE: items the new code unblocks, by kind
    position: int | None = None          # UPGRADE: place in the rollout queue (1-based)


def controller_revision(tree: Path = cv.LIVE_TREE) -> str | None:
    """Hash of the controller code a launch from the live tree would run.

    A rollout changes what a --resume does with held items without changing
    any item status, so the NO_PROGRESS guard must see it as progress.
    """
    import hashlib

    digest = hashlib.sha256()
    paths = sorted((tree / "capability_pipeline").glob("*.py"))
    if not paths:
        return None
    for path in paths:
        try:
            digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
        except OSError:
            return None
    return digest.hexdigest()[:16]


@dataclass
class Options:
    max_launches: int = 4
    window_min: float = 10.0
    grace_min: float = 15.0
    max_per_base_6h: int = 3
    max_cancels: int = 4
    enforce_cuts: bool = False
    enforce_concurrency: bool = False
    force_bases: tuple[str, ...] = ()
    suffix_letter: str = "v"
    upgrade_before: datetime | None = None  # replace live jobs submitted before this (rollout)
    max_upgrades_per_cycle: int = 8
    heal_guard_min: float = 10.0           # sync_healer.sh may be mid-heal on a job this young
    cancel_settle_s: float = 180.0


UNBLOCK_STAGES = {"image_capture": "capture", "image_publication": "publication", "image_review": "review"}
LAUNCH_ACTIONS = ("launch", "relaunch", "upgrade")


def unblock_counts(model: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Per base: non-terminal items in the states the new controller unblocks (capture, publication,
    image review, runtime infrastructure) plus held items; ``total`` counts each item once."""
    out: dict[str, dict[str, int]] = {}
    for record in model.get("items", []):
        if record.get("disposition") in ("accepted", "rejected"):
            continue
        kinds = []
        if record.get("stage") in UNBLOCK_STAGES:
            kinds.append(UNBLOCK_STAGES[record["stage"]])
        text = f"{record.get('state') or ''} {record.get('failure_stage_raw') or ''} {record.get('reason_class') or ''}"
        if "runtime_infrastructure" in text or "InfrastructureError" in text or "VerifierTimeoutError" in text:
            kinds.append("runtime_infra")
        if record.get("disposition") == "held":
            kinds.append("held")
        if not kinds:
            continue
        counts = out.setdefault(record["run"], {"total": 0})
        counts["total"] += 1
        for kind in kinds:
            counts[kind] = counts.get(kind, 0) + 1
    return out


def _fmt_unblock(counts: dict[str, int] | None) -> str:
    counts = counts or {}
    parts = [f"{k} {counts[k]}" for k in ("capture", "publication", "review", "runtime_infra", "held") if counts.get(k)]
    return f"unblock={counts.get('total', 0)}" + (f" ({', '.join(parts)})" if parts else "")


def next_suffix(base: str, jobs: list[dict[str, Any]], ledger: list[dict[str, Any]], letter: str = "v") -> str:
    """Smallest unused <letter><N> above every name already used for this base (jobs or ledger)."""
    used = set()
    for job in jobs:
        if job.get("base") == base and job.get("suffix"):
            used.add(job["suffix"])
    for entry in ledger:
        if entry.get("base") == base and entry.get("run_name"):
            _, suffix = cv.parse_job_name("/muchanem/" + entry["run_name"])
            if suffix:
                used.add(suffix)
    numbers = [int(s[1:]) for s in used if s[0] == letter and s[1:].isdigit()]
    n = max(numbers, default=0) + 1
    while f"{letter}{n}" in used:
        n += 1
    return f"{letter}{n}"


def _ts(entry: dict[str, Any]) -> datetime | None:
    return cv.parse_time(entry.get("at"))


def decide(model: dict[str, Any], jobs: list[dict[str, Any]], desired: dict[str, Any],
           ledger: list[dict[str, Any]], *, now: datetime | None = None,
           opts: Options | None = None, controller: str | None = None) -> list[Decision]:
    """Pure: what to do for every base, and why."""
    now = now or datetime.now(UTC)
    opts = opts or Options()
    live = cv.live_jobs_by_base(jobs)
    job_names = {j["name"].rsplit("/", 1)[-1]: j for j in jobs}
    specs = desired["bases"]
    bases_model = model.get("bases", {})
    decisions: list[Decision] = []
    launches = [e for e in ledger if e.get("action") in LAUNCH_ACTIONS and e.get("result") == "submitted"]
    candidates: list[Decision] = []
    upgrades: list[Decision] = []
    upgraded_runs = {e.get("run_name") for e in launches if e.get("action") == "upgrade"}
    unblock = unblock_counts(model) if opts.upgrade_before is not None else {}

    for base in sorted(set(specs) | set(live) | set(bases_model)):
        spec = specs.get(base)
        b = bases_model.get(base)
        live_jobs = live.get(base, [])
        live_names = [j["name"].rsplit("/", 1)[-1] for j in live_jobs]
        work = ""
        if b:
            c = b.get("counts", {})
            work = (f"{b.get('non_terminal', 0)} non-terminal (queued {b.get('queued', 0)}, active {c.get('active', 0)}, "
                    f"waiting {c.get('waiting', 0)}, parked {c.get('parked', 0)}, held {c.get('held', 0)}); "
                    f"accepted {c.get('accepted', 0)}, rejected {c.get('rejected', 0)}")
        if spec is None:
            if live_jobs or (b and b.get("non_terminal")):
                decisions.append(Decision(base, "REPORT", f"not in desired.json; live={live_names or '-'}; {work}", jobs=live_names))
            continue
        if not spec["active"]:
            if live_jobs:
                action = "CANCEL" if opts.enforce_cuts else "REPORT"
                why = "inactive in desired.json but live" + ("" if opts.enforce_cuts else " (pass --enforce-cuts to cancel)")
                decisions.append(Decision(base, action, f"{why}: {', '.join(live_names)}; {work}", jobs=live_names))
            else:
                decisions.append(Decision(base, "OK", f"inactive, no live job; {work or 'no snapshot'}"))
            continue
        # ---- active
        recent_self = [e for e in launches if e.get("base") == base and _ts(e) and now - _ts(e) <= timedelta(minutes=opts.grace_min)]
        invisible = [e for e in recent_self if e.get("run_name") not in job_names]
        if live_jobs:
            if len(live_jobs) > 1:
                decisions.append(Decision(base, "REPORT", f"DUPLICATE_LIVE: {len(live_jobs)} live jobs share one output prefix: {', '.join(live_names)}", jobs=live_names))
                continue
            submitted = cv.parse_time(live_jobs[0].get("submitted"))
            if opts.upgrade_before is not None and submitted is not None and submitted < opts.upgrade_before:
                # Rollout: replace an old-code job (cancel, then --resume under the next suffix).
                old = live_names[0]
                # The ledger exemption exists for a FUTURE cutoff (replacements submitted before it
                # would otherwise be replaced again).  A cutoff in the past is a new rollout: every
                # live job submitted before it runs old code, replacements from an earlier wave included.
                if old in upgraded_runs and opts.upgrade_before > now:
                    decisions.append(Decision(base, "OK", f"live {old} is itself an upgrade replacement (ledger); not upgraded again", jobs=live_names))
                elif now - submitted < timedelta(minutes=opts.heal_guard_min):
                    decisions.append(Decision(base, "SKIP", f"UPGRADE_HEAL_GUARD: {old} submitted {cv.fmt_age((now - submitted).total_seconds())} ago "
                                                            f"(< {opts.heal_guard_min:g}m; sync_healer may be mid-heal)", jobs=live_names))
                elif b is None or b.get("manifest_created") is None:
                    decisions.append(Decision(base, "SKIP", f"UPGRADE_NO_SNAPSHOT: cannot see {old}'s items; not replacing it blind", jobs=live_names))
                elif not b.get("non_terminal"):
                    decisions.append(Decision(base, "OK", f"live {old} predates the cutoff but every item is terminal; left to exit", jobs=live_names))
                else:
                    counts = unblock.get(base, {"total": 0})
                    upgrades.append(Decision(
                        base, "UPGRADE", f"live {old} submitted {cv.iso(submitted)} < cutoff {cv.iso(opts.upgrade_before)}; {work}",
                        spec=spec, jobs=live_names, replaces=old, unblock=counts,
                        priority=(0 if base in HC_BASES else 1, -counts.get("total", 0), base),
                        fingerprint=b.get("fingerprint"), queued=b.get("queued"), controller=controller))
                continue
            sub = (b or {}).get("submission") or {}
            live_conc = sub.get("concurrency") if sub.get("run_name") in live_names else None
            if live_conc is not None and live_conc != spec["concurrency"]:
                if opts.enforce_concurrency:
                    candidates.append(Decision(base, "RELAUNCH", f"live {live_names[0]} runs --concurrency {live_conc}, desired {spec['concurrency']}",
                                               spec=spec, jobs=live_names, priority=(0 if base in HC_BASES else 1, -(b or {}).get("progressable", 0)),
                                               fingerprint=(b or {}).get("fingerprint"), queued=(b or {}).get("queued")))
                else:
                    decisions.append(Decision(base, "REPORT", f"CONC_MISMATCH live {live_names[0]} --concurrency {live_conc}, desired {spec['concurrency']} "
                                                              "(pass --enforce-concurrency to relaunch)", jobs=live_names))
                continue
            conc_note = f"c{live_conc}" if live_conc is not None else "concurrency unknown (job has not synced its submission yet)"
            decisions.append(Decision(base, "OK", f"live {', '.join(live_names)} {conc_note}", jobs=live_names))
            continue
        if invisible:
            e = invisible[-1]
            decisions.append(Decision(base, "SKIP", f"IN_FLIGHT: launched {e['run_name']} at {e['at']}, not yet visible in Iris (grace {opts.grace_min:g}m)"))
            continue
        if b is None or b.get("manifest_created") is None:
            why = f"manifest unreadable: {b.get('manifest_error')}" if b and b.get("manifest_error") else "NO_SNAPSHOT: --resume needs a prior snapshot"
            decisions.append(Decision(base, "SKIP", why))
            continue
        if b.get("all_terminal"):
            decisions.append(Decision(base, "SKIP", f"all items terminal; {work}"))
            continue
        if b.get("accepted_count") is None:
            decisions.append(Decision(base, "SKIP", "catalog size unknown (no run.json, no source file): cannot prove work remains"))
            continue
        if not b.get("non_terminal"):
            decisions.append(Decision(base, "SKIP", f"no non-terminal items; {work}"))
            continue
        forced = base in opts.force_bases
        mine = [e for e in launches if e.get("base") == base]
        if mine and not forced:
            last = mine[-1]
            last_job = job_names.get(last.get("run_name") or "")
            ended = last_job is not None and last_job.get("state") in cv.TERMINAL_JOB_STATES
            same_code = controller is None or last.get("controller") == controller
            if ended and same_code and last.get("fingerprint") == b.get("fingerprint") and last.get("queued") == b.get("queued"):
                decisions.append(Decision(base, "SKIP", f"NO_PROGRESS: our last launch {last['run_name']} ended {last_job.get('state')} and no item "
                                                        f"status changed since (fingerprint {b.get('fingerprint')}); fix the cause, or --force-base {base}"))
                continue
            recent = [e for e in mine if _ts(e) and now - _ts(e) <= timedelta(hours=6)]
            if len(recent) >= opts.max_per_base_6h:
                decisions.append(Decision(base, "SKIP", f"FLAPPING: {len(recent)} launches of {base} in 6h (cap {opts.max_per_base_6h}); investigate, or --force-base {base}"))
                continue
        last = cv.parse_time((b.get("terminal") or {}).get("finished_utc")) if b.get("last_job_ended") else None
        ended = f"; last job exited {(b.get('terminal') or {}).get('exit_code')} at {cv.iso(last)}" if last else ""
        candidates.append(Decision(base, "LAUNCH", f"active, no live job, {work}{ended}", spec=spec,
                                   priority=(0 if base in HC_BASES else 1, -b.get("progressable", 0), base),
                                   fingerprint=b.get("fingerprint"), queued=b.get("queued"), controller=controller))

    window_used = [e for e in launches if _ts(e) and now - _ts(e) <= timedelta(minutes=opts.window_min)]
    budget = max(0, opts.max_launches - len(window_used))
    # Parked bases first (pure capacity gain), then the rollout queue (a like-for-like swap):
    # healthcare, then the bases with most items the new code unblocks, then the rest.
    for decision in sorted(candidates, key=lambda d: d.priority):
        if budget > 0:
            decision.run_name = JOB_LEAF.format(base=decision.base, suffix=next_suffix(decision.base, jobs, ledger, opts.suffix_letter))
            budget -= 1
        else:
            decision.reason = f"rate limit ({opts.max_launches}/{opts.window_min:g}m, {len(window_used)} used): would {decision.action}: {decision.reason}"
            decision.action = "DEFER"
        decisions.append(decision)
    done_upgrades = 0
    for position, decision in enumerate(sorted(upgrades, key=lambda d: d.priority), start=1):
        decision.position = position
        decision.run_name = JOB_LEAF.format(base=decision.base, suffix=next_suffix(decision.base, jobs, ledger, opts.suffix_letter))
        if budget > 0 and done_upgrades < opts.max_upgrades_per_cycle:
            budget -= 1
            done_upgrades += 1
        else:
            why = (f"upgrade cap {opts.max_upgrades_per_cycle}/cycle" if done_upgrades >= opts.max_upgrades_per_cycle
                   else f"rate limit ({opts.max_launches}/{opts.window_min:g}m)")
            decision.reason = f"{why}: queued for a later cycle: {decision.reason}"
            decision.action = "DEFER"
        decisions.append(decision)
    cancels = [d for d in decisions if d.action == "CANCEL"]
    for extra in cancels[opts.max_cancels:]:
        extra.action, extra.reason = "DEFER", f"cancel cap {opts.max_cancels}/cycle: {extra.reason}"
    # Submitted-equals-sized: every launch we recorded must become visible.
    for e in launches:
        at = _ts(e)
        if at and timedelta(minutes=opts.grace_min) < now - at <= timedelta(hours=2) and e.get("run_name") not in job_names:
            decisions.append(Decision(e["base"], "REPORT", f"LAUNCH_NOT_VISIBLE: ledger says {e['run_name']} submitted at {e['at']} but Iris does not list it"))
    order = {"CANCEL": 0, "RELAUNCH": 1, "LAUNCH": 2, "UPGRADE": 3, "DEFER": 4, "REPORT": 5, "SKIP": 6, "OK": 7}
    decisions.sort(key=lambda d: (d.position is not None, d.position or 0, order.get(d.action, 9), d.base))
    return decisions


# --------------------------------------------------------------------------------------------
# Launch / cancel (only under --apply)
# --------------------------------------------------------------------------------------------


def submit_command(base: str, spec: dict[str, Any], run_name: str, *, dry_run: bool = False) -> list[str]:
    # --dry-run must precede extra_args, which may start with "--" (arguments for synthesis).
    return ["bash", "scripts/submit.sh", "--stage", "synthesize", "--source", spec["source"],
            "--out", f"runs/{cv.RUN}/{base}", "--run-name", run_name, "--resume", "--sandbox", "silo",
            "--tier", "bulk", "--concurrency", str(spec["concurrency"]), *(["--dry-run"] if dry_run else []),
            *spec.get("extra_args", [])]


def launch_script(base: str, spec: dict[str, Any], run_name: str, log_path: Path, *, dry_run: bool = False) -> str:
    """One submission attempt, exactly as relaunch_k.sh / readd.sh / relaunch_hc.sh do it."""
    cmd = " ".join(shlex.quote(part) for part in submit_command(base, spec, run_name, dry_run=dry_run))
    return "\n".join([
        f"# generated by ops/shard_supervisor.py for {base} -> {run_name}",
        f"cd {shlex.quote(str(cv.LIVE_TREE))}",
        "N=build/construct-003; export MARIN=/Users/k3sc0re/openathena/marin-construct CAPABILITY_MAX_REPAIR_ROUNDS=4 "
        f"CAPABILITY_JUDGE_CALIBRATION_CONCURRENCY=8 CPU={spec['cpu']} MEM={spec['mem']} DISK={spec['disk']} MAX_RETRIES=3 "
        "OMP_MODELS_SOURCE=$PWD/$N/omp-arms/models.armB.yml",
        ". $N/silo_env.sh; unset TARGET_CLUSTER RELAY_JOB CAPABILITY_RELAY_ROUTE_JOB",
        f"{cmd} > {shlex.quote(str(log_path))} 2>&1",
        "",
    ])


def submit_running() -> bool:
    proc = subprocess.run(["pgrep", "-f", "scripts/submit.sh"], capture_output=True, text=True)
    return proc.returncode == 0


def clear_stale_lock(log) -> None:
    if LOCK_DIR.is_dir() and not submit_running():
        try:
            LOCK_DIR.rmdir()
            log(f"cleared stale submit lock {LOCK_DIR}")
        except OSError as error:
            log(f"could not clear {LOCK_DIR}: {error}")


def execute_launch(decision: Decision, log, *, attempts: int = 8, retry_sleep: float = 60.0) -> dict[str, Any]:
    LAUNCH_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LAUNCH_DIR / f"{decision.run_name}.log"
    script = LAUNCH_DIR / f"{decision.run_name}.sh"
    script.write_text(launch_script(decision.base, decision.spec, decision.run_name, log_path))
    result = "failed"
    detail = ""
    for attempt in range(1, attempts + 1):
        clear_stale_lock(log)
        try:
            subprocess.run(["bash", str(script)], timeout=1800, check=False)
        except subprocess.TimeoutExpired:
            detail = "submit.sh timed out after 1800s"
            break
        text = log_path.read_text(errors="replace") if log_path.is_file() else ""
        if "Job submitted" in text:
            result = "submitted"
            break
        if "staging is in use" in text:
            detail = f"staging in use (attempt {attempt}/{attempts})"
            time.sleep(retry_sleep)
            continue
        errors = [line for line in text.splitlines() if re.match(r"^(error|ERROR|fatal)", line)]
        detail = (errors or text.strip().splitlines()[-1:] or ["no output"])[0][:300]
        break
    return {"result": result, "detail": detail, "log": str(log_path), "script": str(script)}


def job_state(job_leaf: str, fetch=None) -> str:
    """Current Iris state of one job ('unknown: ...' when the list cannot be read)."""
    fetch = fetch or cv.fetch_jobs
    try:
        rows = [j for j in fetch(prefix=f"/muchanem/{job_leaf}", limit=50) if j["name"].rsplit("/", 1)[-1] == job_leaf]
    except cv.ReadError as error:
        return f"unknown: {error}"
    return rows[0]["state"] if rows else "unknown: not listed"


DRY_RUN_OK = "dry run: Iris job not submitted"


def execute_preflight(decision: Decision, log) -> dict[str, Any]:
    """submit.sh --dry-run for the replacement: every local preflight (relay route, fleet health,
    bundle) without submitting.  Run BEFORE cancelling the old job, so a failing preflight never
    leaves a base with no job (2026-09-30: three bases cancelled, then refused by the health preflight)."""
    LAUNCH_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LAUNCH_DIR / f"{decision.run_name}.preflight.log"
    script = LAUNCH_DIR / f"{decision.run_name}.preflight.sh"
    script.write_text(launch_script(decision.base, decision.spec, decision.run_name, log_path, dry_run=True))
    clear_stale_lock(log)
    try:
        subprocess.run(["bash", str(script)], timeout=1800, check=False)
    except subprocess.TimeoutExpired:
        return {"result": "preflight_failed", "detail": "submit.sh --dry-run timed out after 1800s"}
    text = log_path.read_text(errors="replace") if log_path.is_file() else ""
    if DRY_RUN_OK in text:
        return {"result": "ok"}
    errors = [line for line in text.splitlines() if re.match(r"^(error|ERROR|fatal)", line)]
    return {"result": "preflight_failed", "detail": (errors or text.strip().splitlines()[-1:] or ["no output"])[0][:300]}


def execute_upgrade(decision: Decision, log, *, cancel=None, launch=None, state_of=None, preflight=None,
                    settle_s: float = 180.0, poll_s: float = 10.0, sleep=time.sleep, clock=time.time) -> dict[str, Any]:
    """Preflight the replacement, cancel the old-code job, wait until Iris no longer shows it live,
    then submit the replacement."""
    cancel = cancel or execute_cancel
    launch = launch or execute_launch
    state_of = state_of or job_state
    preflight = preflight or execute_preflight
    old = decision.replaces
    check = preflight(decision, log)
    if check.get("result") != "ok":
        return {"result": "preflight_failed",
                "detail": f"replacement preflight failed ({check.get('detail')}); {old} left running, nothing cancelled"}
    res = cancel(old)
    append_ledger({"at": cv.iso(datetime.now(UTC)), "base": decision.base, "action": "upgrade-cancel", "job": old,
                   **res, "reason": decision.reason})
    if res.get("result") != "cancelled":
        return {"result": "cancel_failed", "detail": f"cancel of {old} failed: {res.get('detail')}; replacement NOT submitted"}
    deadline = clock() + settle_s
    while True:
        state = state_of(old)
        # 'unknown' is not proof the old writer stopped: keep polling until the deadline.
        if not state.startswith("unknown") and state not in cv.LIVE_JOB_STATES:
            break
        if clock() >= deadline:
            return {"result": "cancel_not_settled",
                    "detail": f"{old} still {state} after {settle_s:.0f}s; replacement NOT submitted (the normal LAUNCH path will pick the base up)"}
        log(f"{old} is {state}; waiting for it to stop before submitting {decision.run_name}")
        sleep(poll_s)
    out = launch(decision, log)
    return {**out, "old_state": state}


def execute_cancel(job_leaf: str) -> dict[str, Any]:
    proc = subprocess.run(["uv", "run", "--frozen", "iris", "--cluster=marin", "job", "cancel", f"/muchanem/{job_leaf}"],
                          cwd=cv.MARIN_DIR, capture_output=True, text=True, timeout=120)
    return {"result": "cancelled" if proc.returncode == 0 else "failed",
            "detail": (proc.stderr or proc.stdout).strip().splitlines()[-1:][0][:300] if (proc.stderr or proc.stdout).strip() else ""}


# --------------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------------


def render(decisions: list[Decision], desired: dict[str, Any], *, apply: bool, opts: Options,
           verbose: bool = False, show_scripts: bool = False, now: datetime | None = None,
           desired_path: Path = DESIRED_PATH) -> str:
    now = now or datetime.now(UTC)
    specs = desired["bases"]
    active = sum(1 for s in specs.values() if s["active"])
    lines = [f"SUPERVISOR {cv.RUN} {cv.iso(now)} mode={'APPLY' if apply else 'DRY-RUN'} desired={desired_path} "
             f"({len(specs)} bases: {active} active, {len(specs) - active} inactive)",
             f"limits: {opts.max_launches} launches/{opts.window_min:g}m, {opts.max_per_base_6h}/base/6h, {opts.max_cancels} cancels/cycle; "
             f"enforce_cuts={opts.enforce_cuts} enforce_concurrency={opts.enforce_concurrency}"]
    if opts.upgrade_before is not None:
        lines.append(f"rollout: --upgrade-before {cv.iso(opts.upgrade_before)}; {opts.max_upgrades_per_cycle} upgrades/cycle; "
                     f"heal guard {opts.heal_guard_min:g}m; cancel settle {opts.cancel_settle_s:.0f}s"
                     + ("; NOTE cutoff is in the future: replacements are exempted via the ledger" if opts.upgrade_before > now else ""))
    counts: dict[str, int] = {}
    plan = [d for d in decisions if d.position is not None]
    for d in decisions:
        counts[d.action] = counts.get(d.action, 0) + 1
        if d.position is not None:
            continue
        if d.action == "OK" and not verbose:
            continue
        head = f"{d.action:8s} {d.base:10s}"
        if d.run_name:
            s = d.spec or {}
            head += f" -> {d.run_name}  conc={s.get('concurrency')} cpu={s.get('cpu')} mem={s.get('mem')} disk={s.get('disk')}"
        lines.append(head)
        lines.append(f"         why: {d.reason}")
        if d.action in ("LAUNCH", "RELAUNCH") and d.run_name:
            if d.action == "RELAUNCH":
                lines.append(f"         cancel first: {', '.join('/muchanem/' + j for j in d.jobs)}")
            cmd = " ".join(shlex.quote(p) for p in submit_command(d.base, d.spec, d.run_name))
            lines.append(f"         cmd (cwd {cv.LIVE_TREE}): {cmd}")
            if show_scripts:
                script = launch_script(d.base, d.spec, d.run_name, LAUNCH_DIR / f"{d.run_name}.log")
                lines.extend("           | " + line for line in script.splitlines())
        if d.action == "CANCEL":
            lines.append(f"         cmd: uv run --frozen iris --cluster=marin job cancel {' '.join('/muchanem/' + j for j in d.jobs)}")
    if plan:
        now_n = sum(1 for d in plan if d.action == "UPGRADE")
        lines.append(f"UPGRADE PLAN ({len(plan)} bases, in order; {now_n} this cycle, the rest on later cycles):")
        for d in plan:
            s = d.spec or {}
            tag = "UPGRADE" if d.action == "UPGRADE" else "later"
            lines.append(f"  #{d.position:<3d} {tag:7s} {d.base:10s} cancel {d.replaces} -> {d.run_name}  conc={s.get('concurrency')} "
                         f"cpu={s.get('cpu')} mem={s.get('mem')} disk={s.get('disk')}  {_fmt_unblock(d.unblock)}")
            if d.action == "UPGRADE":
                lines.append(f"         why: {d.reason}")
                cmd = " ".join(shlex.quote(p) for p in submit_command(d.base, d.spec, d.run_name))
                lines.append(f"         then (after {d.replaces} stops, <= {opts.cancel_settle_s:.0f}s), cwd {cv.LIVE_TREE}: {cmd}")
                if show_scripts:
                    script = launch_script(d.base, d.spec, d.run_name, LAUNCH_DIR / f"{d.run_name}.log")
                    lines.extend("           | " + line for line in script.splitlines())
            elif verbose:
                lines.append(f"         why: {d.reason}")
    lines.append("SUMMARY " + " ".join(f"{k.lower()}={v}" for k, v in sorted(counts.items())))
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def run_cycle(args: argparse.Namespace, store: cv.S3Store, opts: Options) -> int:
    desired = load_desired(Path(args.desired))
    jobs = cv.parse_job_list(Path(args.jobs_file).read_text()) if args.jobs_file else cv.fetch_jobs()
    collected = cv.collect(store, jobs=jobs, bases=args.run_list)
    if collected.errors:
        for error in collected.errors:
            print(f"ERROR {error}", flush=True)
    model = cv.build_model(collected, history=None)
    ledger = read_ledger()
    now = datetime.now(UTC)
    decisions = decide(model, jobs, desired, ledger, now=now, opts=opts, controller=controller_revision())
    if collected.errors:
        # A base whose snapshot could not be read must not be launched on stale evidence.
        bad = {e.split(":", 1)[0] for e in collected.errors}
        for d in decisions:
            if d.base in bad and d.action in ("LAUNCH", "RELAUNCH", "UPGRADE"):
                d.action, d.reason, d.run_name = "SKIP", f"read error this cycle; not acting: {d.reason}", None
    print(render(decisions, desired, apply=args.apply, opts=opts, verbose=args.verbose, show_scripts=args.show_scripts,
                 now=now, desired_path=Path(args.desired)), flush=True)
    if not args.apply:
        return 0
    log = lambda m: print(f"  {m}", flush=True)  # noqa: E731
    for d in decisions:
        if d.action == "CANCEL":
            for job in d.jobs:
                res = execute_cancel(job)
                append_ledger({"at": cv.iso(datetime.now(UTC)), "base": d.base, "action": "cancel", "job": job, **res, "reason": d.reason})
                print(f"CANCEL {d.base} {job}: {res['result']} {res['detail']}", flush=True)
        elif d.action in ("LAUNCH", "RELAUNCH"):
            if d.action == "RELAUNCH":
                ok = True
                for job in d.jobs:
                    res = execute_cancel(job)
                    append_ledger({"at": cv.iso(datetime.now(UTC)), "base": d.base, "action": "cancel", "job": job, **res, "reason": d.reason})
                    ok = ok and res["result"] == "cancelled"
                if not ok:
                    print(f"ERROR {d.base}: cancel before relaunch failed; not launching", flush=True)
                    continue
            res = execute_launch(d, log)
            append_ledger({"at": cv.iso(datetime.now(UTC)), "base": d.base, "action": d.action.lower(), "run_name": d.run_name,
                           "concurrency": d.spec["concurrency"], "fingerprint": d.fingerprint, "queued": d.queued,
                           "controller": d.controller,
                           "reason": d.reason, **res})
            tag = "LAUNCHED" if res["result"] == "submitted" else "ERROR"
            print(f"{tag} {d.base} -> {d.run_name}: {res['result']} {res['detail']} (log {res['log']})", flush=True)
        elif d.action == "UPGRADE":
            res = execute_upgrade(d, log, settle_s=opts.cancel_settle_s)
            append_ledger({"at": cv.iso(datetime.now(UTC)), "base": d.base, "action": "upgrade", "run_name": d.run_name,
                           "replaces": d.replaces, "position": d.position, "unblock": d.unblock,
                           "concurrency": d.spec["concurrency"], "fingerprint": d.fingerprint, "queued": d.queued,
                           "controller": d.controller, "cutoff": cv.iso(opts.upgrade_before), "reason": d.reason, **res})
            tag = "UPGRADED" if res.get("result") == "submitted" else "ERROR"
            print(f"{tag} {d.base}: {d.replaces} -> {d.run_name}: {res.get('result')} {res.get('detail', '')}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--init", action="store_true", help="write ops/desired.json from live jobs + deferred list")
    parser.add_argument("--force", action="store_true", help="--init: overwrite an existing desired.json")
    parser.add_argument("--init-no-s3", action="store_true", help="--init: skip reading live concurrency from S3")
    parser.add_argument("--desired", default=str(DESIRED_PATH))
    parser.add_argument("--apply", action="store_true", help="actually submit/cancel (default: dry run)")
    parser.add_argument("--enforce-cuts", action="store_true", help="cancel live jobs of inactive bases (with --apply)")
    parser.add_argument("--enforce-concurrency", action="store_true", help="relaunch live jobs whose --concurrency differs")
    parser.add_argument("--force-base", action="append", default=[], help="bypass NO_PROGRESS/FLAPPING for a base")
    parser.add_argument("--max-launches", type=int, default=4)
    parser.add_argument("--window-min", type=float, default=10.0)
    parser.add_argument("--max-per-base-6h", type=int, default=3)
    parser.add_argument("--max-cancels", type=int, default=4)
    parser.add_argument("--suffix-letter", default="v")
    parser.add_argument("--upgrade-before", help="rollout: cancel + --resume relaunch every ACTIVE base whose live job was "
                                                 "submitted before this ISO8601 UTC time and still has non-terminal items")
    parser.add_argument("--max-upgrades-per-cycle", type=int, default=8)
    parser.add_argument("--heal-guard-min", type=float, default=10.0, help="never upgrade a job younger than this (healer)")
    parser.add_argument("--cancel-settle-s", type=float, default=180.0, help="max wait for a cancelled job to stop")
    parser.add_argument("--only-runs", dest="run_list", type=lambda s: [x for x in s.split(",") if x])
    parser.add_argument("--jobs-file", help="use a saved 'iris job list' output (testing)")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--show-scripts", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="also list OK bases")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[a-z]", args.suffix_letter):
        parser.error("--suffix-letter must be one lowercase letter")
    cutoff = None
    if args.upgrade_before:
        cutoff = cv.parse_time(args.upgrade_before)
        if cutoff is None:
            parser.error(f"--upgrade-before {args.upgrade_before!r} is not an ISO8601 time")
    opts = Options(max_launches=args.max_launches, window_min=args.window_min, max_per_base_6h=args.max_per_base_6h,
                   max_cancels=args.max_cancels, enforce_cuts=args.enforce_cuts,
                   enforce_concurrency=args.enforce_concurrency, force_bases=tuple(args.force_base),
                   suffix_letter=args.suffix_letter, upgrade_before=cutoff,
                   max_upgrades_per_cycle=args.max_upgrades_per_cycle, heal_guard_min=args.heal_guard_min,
                   cancel_settle_s=args.cancel_settle_s)
    if args.init:
        path = Path(args.desired)
        if path.exists() and not args.force:
            print(f"ERROR {path} exists; pass --force to overwrite", file=sys.stderr)
            return 2
        jobs = cv.parse_job_list(Path(args.jobs_file).read_text()) if args.jobs_file else cv.fetch_jobs()
        model = None
        if not args.init_no_s3:
            store = cv.S3Store()
            store.connect()
            model = cv.build_model(cv.collect(store, jobs=jobs), history=None)
        desired = init_desired(jobs, pilots=load_pilots(), deferred=parse_base_list(DEFERRED),
                               quarantine=parse_base_list(QUARANTINE), hold=parse_base_list(HOLD),
                               bases=all_bases(), model=model)
        path.write_text(json.dumps(desired, indent=1) + "\n")
        bases = desired["bases"]
        print(f"wrote {path}: {len(bases)} bases, {sum(1 for s in bases.values() if s['active'])} active")
        for base, spec in bases.items():
            flag = "ACTIVE  " if spec["active"] else "inactive"
            if not spec["active"] or "live at init" not in spec["note"] or "script rule" in spec["note"]:
                print(f"  {flag} {base:10s} conc={spec['concurrency']:<3d} {spec['note']}")
        print(f"  (+ {sum(1 for s in bases.values() if s['active'] and 'live at init' in s['note'] and 'script rule' not in s['note'])} "
              "active bases live at init with the script-rule sizing)")
        return 0
    store = cv.S3Store()
    try:
        store.connect()
    except cv.ReadError as error:
        print(f"ERROR {error}", file=sys.stderr)
        return 2
    while True:
        started = time.time()
        try:
            if time.time() - store.keys_fetched_at > 50 * 60:
                store.refresh_keys()
            run_cycle(args, store, opts)
        except (cv.ReadError, SystemExit) as error:
            print(f"ERROR {cv.hhmm()} {error}", flush=True)
            if not args.loop:
                return 2
        if not args.loop:
            return 0
        time.sleep(max(5.0, args.interval - (time.time() - started)))


if __name__ == "__main__":
    raise SystemExit(main())

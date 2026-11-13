# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archive-only continuation: no dependency can launch an earlier training stage."""

import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
import wandb
import yaml
from fray.types import ResourceConfig
from marin.evaluation.records import EvalRunRecord
from marin.execution.artifact import RECORD_FILENAME
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.execution.remote import remote
from marin.rl.skyrl import SkyRLRun
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.redaction import redact_string
from rigging.secrets import require_secret
from wandb import env as wandb_env
from wandb.apis.public.runhistory.downloads import IncompleteRunHistoryError

from experiments.grug.moe_hero_ep.ops.upload_hf import RESPONSIBLE_USE, WARNING
from experiments.post_training.long_loop.archive import (
    ArchiveInput,
    checkpoint_cleanup,
    digest,
    seal_archive,
    write_once,
)
from experiments.post_training.long_loop.eval_archive import publish_evaluation
from experiments.post_training.long_loop.eval_runtime.audit_results import require_evaluation_coverage
from experiments.post_training.long_loop.hf_archive import (
    HfUpload,
    file_identity,
    upload_checkpoint,
    verify_hf_checkpoint,
)
from experiments.post_training.long_loop.rl_selection import VALIDATION_EMA_WINDOW
from vendor.skyrl.marinskyrl.checkpoint_paths import BEST_CHECKPOINT_FILENAME

REQUIRED_CATEGORIES = {"source", "config", "dataset", "timing", "eval_trace", "wandb"}
INCOMPLETE_HISTORY_EXCEPTION = "wandb.apis.public.runhistory.downloads.IncompleteRunHistoryError"
EXPORT_ERROR_TAIL_BYTES = 8192


def archive_run(*, output: str, plan: dict, completed_report: str) -> None:
    """Seal an explicit reviewed inventory, then clean only its superseded checkpoints."""
    fs, key = filesystem_for(prefix_join(completed_report, "evaluation.json"))
    evaluation = json.loads(fs.cat_file(key))
    require_evaluation_coverage(evaluation)
    inputs = [ArchiveInput(**item) for item in plan["inputs"]]
    missing = REQUIRED_CATEGORIES - {item.category for item in inputs}
    declared_missing = plan["provenance"].get("missing_evidence", {})
    if missing - declared_missing.keys():
        raise ValueError(f"Archive lacks required evidence categories: {sorted(missing)}")
    provenance = dict(plan["provenance"])
    revision = provenance["marin_commit"]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Archive requires an exact Marin commit")
    if provenance["evaluation_complete"] != evaluation["all_verified"]:
        raise ValueError("Archive plan misstates evaluation completeness")
    if not evaluation["all_verified"] and "evaluation" not in declared_missing:
        raise ValueError("Unscored evaluation attempts must block cleanup")
    receipt_fs, receipt_key = filesystem_for(prefix_join(output, "hf-checkpoint.json"))
    if receipt_fs.exists(receipt_key):
        receipt = json.loads(receipt_fs.cat_file(receipt_key))
        upload = plan["hf_upload"]
        if any(receipt[key] != upload[key] for key in ("source", "repo_id", "native_checkpoint")):
            raise ValueError("Existing HF receipt belongs to a different archive plan")
        verify_hf_checkpoint(receipt)
    else:
        receipt = upload_checkpoint(
            HfUpload(**plan["hf_upload"]), timing_destination=prefix_join(output, "timing/hf-upload.jsonl")
        )
        write_once(prefix_join(output, "hf-checkpoint.json"), json.dumps(receipt, sort_keys=True, indent=2).encode())
    provenance["hf_checkpoint"] = receipt
    publication = plan["evaldash"]
    record = EvalRunRecord.model_validate(publication["record"]).model_copy(
        update={"results_path": prefix_join(output, "evaluation")}
    )
    record_uri = publish_evaluation(
        source=publication["source"], record=record, records_prefix=publication["records_prefix"]
    )
    inputs.append(ArchiveInput(record_uri, "evaldash/record.json", "eval_trace"))
    eval_fs, eval_root = filesystem_for(record.results_path)
    for key in sorted(eval_fs.find(eval_root)):
        inputs.append(
            ArchiveInput(
                eval_fs.unstrip_protocol(key),
                prefix_join("evaldash/store", StoragePath(key).relative_to(StoragePath(eval_root))),
                "eval_trace",
            )
        )
    seal_archive(output, inputs, provenance)
    inventory = provenance["checkpoint_inventory"]
    blockers = provenance.get("cleanup_blockers", [])
    if missing or declared_missing or blockers:
        write_once(
            prefix_join(output, "cleanup-blocked.json"),
            json.dumps(
                {
                    "missing_categories": sorted(missing | declared_missing.keys()),
                    "blockers": blockers,
                    "archive_complete": False,
                },
                sort_keys=True,
            ).encode(),
        )
        return
    checkpoint_cleanup(
        output,
        inventory["root"],
        inventory["selected"],
        [item["source"] for item in inventory["disposable_files"]],
    )


def archive_step(version: str, plan: dict, completed_report: ArtifactStep) -> ArtifactStep:
    return apply(
        "long-loop/archive", archive_run, version=version, output=OUT, plan=plan, completed_report=completed_report
    )


def archive_selected_run(*, output: str, plan: dict, trained: str, completed_report: str) -> None:
    """Resolve the selected export from the committed RL result, never a fixed smoke step."""
    run = SkyRLRun.raw_load(trained)
    if run.hf_model_uri is None or run.global_step is None:
        raise ValueError("Selected RL run has no committed HF export")
    fs, root = filesystem_for(run.checkpoint_root)
    best_uri = prefix_join(run.checkpoint_root, BEST_CHECKPOINT_FILENAME)
    best = json.loads(fs.cat_file(prefix_join(root, BEST_CHECKPOINT_FILENAME)))
    selected = prefix_join(run.checkpoint_root, f"global_step_{best['step']}")
    if (
        best["step"] != run.global_step
        or best["model_uri"] != prefix_join(selected, "policy")
        or best["ema_window"] != VALIDATION_EMA_WINDOW
    ):
        raise ValueError("Exported RL step differs from the five-step validation selection")
    request_sha = hashlib.sha256(
        json.dumps({"plan": plan, "trained": trained, "report": completed_report}, sort_keys=True).encode()
    ).hexdigest()
    archive_fs, archive_key = filesystem_for(prefix_join(output, "resolved-plan.json"))
    if archive_fs.exists(archive_key):
        resolved = json.loads(archive_fs.cat_file(archive_key))
        if (
            resolved["archive_request_sha256"] != request_sha
            or resolved["hf_upload"]["source"] != run.hf_model_uri
            or resolved["provenance"]["checkpoint_selection"]["result"] != best
        ):
            raise ValueError("Resolved archive plan changed on resume")
        archive_run(output=output, plan=resolved, completed_report=completed_report)
        return
    resolved = json.loads(json.dumps(plan))
    resolved["archive_request_sha256"] = request_sha
    resolved["hf_upload"].update(source=run.hf_model_uri, native_checkpoint=selected)
    disposable = []
    for key, info in sorted(fs.find(root, detail=True).items()):
        relative = StoragePath(key).relative_to(StoragePath(root))
        component = relative.split("/", 1)[0]
        if info.get("type") != "file" or re.fullmatch(r"global_step_[0-9]+", component) is None:
            continue
        if component == f"global_step_{best['step']}":
            continue
        disposable.append(
            {"source": prefix_join(run.checkpoint_root, relative), "storage_identity": file_identity(info)}
        )
    resolved["provenance"]["checkpoint_inventory"] = {
        "root": run.checkpoint_root,
        "selected": selected,
        "ttl": resolved["native_checkpoint_ttl"],
        "disposable_files": disposable,
    }
    for filename in (BEST_CHECKPOINT_FILENAME, "validation_selection_history.json"):
        resolved["inputs"].append(
            {
                "source": prefix_join(run.checkpoint_root, filename),
                "name": prefix_join("selection", filename),
                "category": "config",
            }
        )
    resolved["provenance"]["checkpoint_selection"] = {"receipt": best_uri, "result": best}
    write_once(prefix_join(output, "resolved-plan.json"), json.dumps(resolved, sort_keys=True, indent=2).encode())
    archive_run(output=output, plan=resolved, completed_report=completed_report)


def _read_document(uri: str) -> Any:
    fs, key = filesystem_for(uri)
    return yaml.safe_load(fs.cat_file(key))


def _input(source: str, name: str, category: str) -> dict:
    fs, key = filesystem_for(source)
    if not fs.isfile(key):
        raise ValueError(f"Required archive evidence is absent: {source}")
    if source.endswith((".distcp", ".safetensors")):
        raise ValueError("Evidence inventory cannot contain model weights")
    return {"source": source, "name": name, "category": category}


def _scan_wandb_history(run_path: str, directory: Path, base_url: str) -> dict:
    run = wandb.Api(api_key=require_secret("WANDB_API_KEY"), overrides={"base_url": base_url}).run(run_path)
    # The exporter crosses a process boundary. Confirm its reported cause with
    # the typed provider exception before taking the only permitted fallback.
    try:
        run.download_history_exports(directory / "native", require_complete_history=True)
    except IncompleteRunHistoryError:
        pass
    else:
        raise RuntimeError("Strict history export failed, but the provider no longer reports incomplete history")
    rows, first_step, last_step = 0, None, None
    history = directory / "scan-history.jsonl"
    checksum = hashlib.sha256()
    with history.open("wb") as stream:
        for row in run.scan_history():
            raw = (json.dumps(row, sort_keys=True) + "\n").encode()
            stream.write(raw)
            checksum.update(raw)
            rows += 1
            step = row.get("_step")
            if step is not None:
                first_step = step if first_step is None else min(first_step, step)
                last_step = step if last_step is None else max(last_step, step)
    receipt = {
        "run_path": run_path,
        "method": "run.scan_history() without sampling, key filters or step bounds",
        "observed_rows": rows,
        "step_min": first_step,
        "step_max": last_step,
        "bytes": history.stat().st_size,
        "sha256": checksum.hexdigest(),
        "provider_completeness_certified": False,
        "deletion_ready": False,
        "limitation": "Provider refused complete native history; only the observed API rows are preserved.",
    }
    (directory / "scan-history-receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2))
    return receipt


def _wandb_export(refs: list[dict], settings: dict, destination: str) -> tuple[list[dict], list[dict]]:
    """Run the pinned, strict exporter only for fully identified training runs."""
    inputs, status = [], []
    for ref in refs:
        if not all(ref.get(key) for key in ("entity", "project", "id")):
            status.append({**ref, "status": "unresolved_run_identity"})
            continue
        run_path = "/".join(ref[key] for key in ("entity", "project", "id"))
        identity = hashlib.sha256(json.dumps({"ref": ref, "settings": settings}, sort_keys=True).encode()).hexdigest()
        marker = prefix_join(destination, identity + ".json")
        marker_fs, marker_key = filesystem_for(marker)
        if marker_fs.exists(marker_key):
            saved = json.loads(marker_fs.cat_file(marker_key))
            for item in saved["files"]:
                if digest(item["source"])[1] != item["sha256"]:
                    raise ValueError("Previously verified W&B export changed")
                inputs.append({key: item[key] for key in ("source", "name", "category")})
            status.append(saved["status"])
            continue
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory)
            config = {
                "source": {
                    "entity": ref["entity"],
                    "projects": {"include": [ref["project"]]},
                    "base_url": settings["base_url"],
                },
                "destination": {"type": "local", "path": str(local / "archive")},
                "archive": {
                    "strict": True,
                    "include": settings["include"],
                    "security": {"profile": "private", "on_sensitive_value": "fail"},
                },
            }
            path = local / "config.yaml"
            path.write_text(yaml.safe_dump(config))
            command = [
                "uv",
                "tool",
                "run",
                "--from",
                settings["package"],
                "--with",
                f"wandb=={settings['sdk_version']}",
                "wandb-archive",
                "--quiet",
            ]
            env = {
                **os.environ,
                "WANDB_API_KEY": require_secret("WANDB_API_KEY"),
                "WANDB_BASE_URL": settings["base_url"],
            }
            # The isolated SDK must not reuse the parent's different wandb-core protocol.
            env.pop(wandb_env.SERVICE, None)
            backup = subprocess.run([*command, "backup", str(path), "--run", run_path], env=env, capture_output=True)
            if backup.returncode:
                # The pinned CLI serializes exception types as traceback lines.
                # Other export failures, including auth and security errors,
                # must propagate rather than silently becoming partial backups.
                if not any(
                    line.decode().partition(":")[0] == INCOMPLETE_HISTORY_EXCEPTION
                    for line in backup.stderr.splitlines()
                ):
                    try:
                        backup.check_returncode()
                    except subprocess.CalledProcessError as error:
                        detail = backup.stderr[-EXPORT_ERROR_TAIL_BYTES:].decode(errors="replace")
                        detail = detail.replace(env["WANDB_API_KEY"], "[REDACTED]")
                        error.add_note("Strict W&B export stderr: " + redact_string(detail))
                        raise
                receipt = _scan_wandb_history(run_path, local, settings["base_url"])
                files = []
                for name in ("scan-history.jsonl", "scan-history-receipt.json"):
                    raw = (local / name).read_bytes()
                    uri = prefix_join(destination, prefix_join(identity, name))
                    write_once(uri, raw)
                    item = _input(uri, prefix_join("wandb/" + ref["stage"], prefix_join(identity, name)), "wandb")
                    inputs.append(item)
                    files.append({**item, "sha256": hashlib.sha256(raw).hexdigest()})
                result = {**ref, "status": "scan_history_fallback_verified", **receipt}
                status.append(result)
                write_once(marker, json.dumps({"files": files, "status": result}, sort_keys=True).encode())
                continue
            verified = subprocess.run([*command, "verify", str(path), "--deep"], env=env, capture_output=True)
            summary = json.loads(verified.stdout) if verified.returncode == 0 else {}
            if not summary.get("ok") or summary.get("run_count") != 1:
                verified.check_returncode()
                raise ValueError(f"W&B archive verification failed: {summary}")
            generation = hashlib.sha256((local / "archive" / "archive.json").read_bytes()).hexdigest()
            files = []
            for file in sorted((local / "archive").rglob("*")):
                if not file.is_file():
                    continue
                name = prefix_join(prefix_join(ref["stage"], generation), file.relative_to(local / "archive").as_posix())
                uri = prefix_join(destination, name)
                write_once(uri, file.read_bytes())
                item = _input(uri, prefix_join("wandb", name), "wandb")
                inputs.append(item)
                files.append({**item, "sha256": hashlib.sha256(file.read_bytes()).hexdigest()})
            result = {**ref, "status": "strict_export_verified", "verification": summary}
            status.append(result)
            write_once(marker, json.dumps({"files": files, "status": result}, sort_keys=True).encode())
    return inputs, status


def derive_archive_plan(
    *,
    output: str,
    settings: dict,
    resolved_spec: dict,
    artifacts: dict[str, str],
    trained: str,
    evaluated: str,
    completed_report: str,
) -> dict:
    """Resolve evidence only from graph outputs, consumed configs and explicit file pins."""
    roots = {**artifacts, "rl": trained, "eval": evaluated, "report": completed_report}
    run = SkyRLRun.raw_load(trained)
    if run.hf_model_uri is None:
        raise ValueError("Archive requires a committed selected HF export")
    revision = resolved_spec["marin_revision"]
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Source revision must be an exact commit")
    inputs, documents, limitations, blockers = [], {}, [], []
    if "finalization" in roots:
        inputs.append(_input(prefix_join(roots["finalization"], "finalization.json"), "finalization.json", "provenance"))
    for stage, entries in settings["stage_files"].items():
        root = roots[stage]
        for entry in entries:
            uri = prefix_join(root, entry["path"])
            inputs.append(_input(uri, prefix_join(stage, entry["path"]), entry["category"]))
            if entry["path"].endswith((".json", ".yaml")):
                documents[(stage, entry["path"])] = _read_document(uri)
    records = {}
    for stage, root in sorted(roots.items()):
        uri = prefix_join(root, RECORD_FILENAME)
        fs, key = filesystem_for(uri)
        if fs.isfile(key):
            inputs.append(_input(uri, prefix_join("artifact-records", stage + ".json"), "source"))
            records[stage] = _read_document(uri)
    metadata_root = prefix_join(output, "derived-inputs")

    def record(name: str, value: dict, category: str) -> None:
        uri = prefix_join(metadata_root, name)
        write_once(uri, json.dumps(value, sort_keys=True, indent=2).encode())
        inputs.append(_input(uri, name, category))

    record(
        "source.json",
        {
            "repository": settings["source_repository"],
            "marin_revision": revision,
            "artifacts": roots,
            "adopted_sft": resolved_spec.get("adopted_sft"),
        },
        "source",
    )
    record("resolved-spec.json", resolved_spec, "config")
    record("dataset-referents.json", resolved_spec["datasets"], "dataset")
    for stage in ("rl", "eval"):
        launch = documents[(stage, "resolved-launch.yaml")]
        retention = launch["config"]["skyrl"]["generator"]["trajectory_retention"]
        prefix = retention["output_path"]
        ledger_uri = prefix_join(prefix, "_retention_ledger.json")
        ledger = _read_document(ledger_uri)
        inputs.append(
            _input(
                ledger_uri, prefix_join(stage, "retention-ledger.json"), "eval_trace" if stage == "eval" else "rl_trace"
            )
        )
        for path, item in sorted(ledger["archives"].items()):
            if StoragePath(path).scheme or ".." in StoragePath(path).segments or path.startswith("/"):
                raise ValueError("Retention archive must be relative to the consumed retention root")
            uri = prefix_join(prefix, path)
            fs, key = filesystem_for(uri)
            if fs.size(key) != item["bytes"]:
                raise ValueError(f"Retention archive size mismatch: {uri}")
            inputs.append(
                _input(uri, prefix_join(stage + "/retention", path), "eval_trace" if stage == "eval" else "rl_trace")
            )
        if stage == "eval" and not ledger["records"]:
            raise ValueError("Completed evaluation lacks retained trace records")
    eval_source = prefix_join(evaluated, "exports/dumped_evals/eval_only")
    eval_fs, eval_prefix = filesystem_for(eval_source)
    native_rows = sorted(eval_fs.glob(prefix_join(eval_prefix, "*.jsonl")))
    if not native_rows:
        raise ValueError("Completed evaluation lacks native result rows")
    for key in native_rows:
        inputs.append(
            _input(eval_fs.unstrip_protocol(key), prefix_join("eval/native", StoragePath(key).name), "eval_trace")
        )
    timing_sources = documents[("report", "timing-sources.json")]
    for index, uri in enumerate(timing_sources):
        inputs.append(_input(uri, f"timing/journals/{index:06d}.jsonl", "timing"))
    limitations.append("Stage timing journals do not establish complete native phase or allocation coverage.")
    launch = documents[("rl", "resolved-launch.yaml")]
    trainer = launch["config"]["skyrl"]["trainer"]
    order_root = prefix_join(trainer["export_path"], "training_order")
    order_fs, order_prefix = filesystem_for(order_root)
    orders = sorted(order_fs.glob(prefix_join(order_prefix, "step-*.json")))
    for key in orders:
        inputs.append(
            _input(order_fs.unstrip_protocol(key), prefix_join("training-order", StoragePath(key).name), "dataset")
        )
    if not orders:
        limitations.append(
            "Observed RL learner-batch order is unavailable; dataset order is not observed training order."
        )
    extra = settings["additional_evidence"] + (resolved_spec.get("adopted_sft") or {}).get("additional_evidence", [])
    for item in extra:
        if digest(item["source"])[1] != item["sha256"]:
            raise ValueError(f"Pinned additional evidence changed: {item['name']}")
        inputs.append(_input(item["source"], prefix_join("additional", item["name"]), item["category"]))
    sft = documents[("sft", "consumed-config.json")]["trainer"]["trainer"]
    tracker = sft["tracker"]
    refs = [
        {
            "stage": "sft",
            "entity": tracker.get("entity"),
            "project": tracker.get("project"),
            "id": tracker.get("id"),
            "identity_source": prefix_join(roots["sft"], "consumed-config.json"),
        },
        {
            "stage": "rl",
            "entity": resolved_spec["rl"]["execution"].get("wandb_entity"),
            "project": trainer.get("project_name"),
            "id": None,
            "experiment_name": trainer.get("run_name"),
            "identity_source": prefix_join(trained, "resolved-launch.yaml"),
        },
    ]
    rl_wandb = prefix_join(trainer["export_path"], "wandb-run.json")
    wandb_fs, wandb_key = filesystem_for(rl_wandb)
    if wandb_fs.isfile(wandb_key):
        ref = _read_document(rl_wandb)
        ref = {**ref, "stage": "rl", "identity_source": rl_wandb}
        inputs.append(_input(rl_wandb, "wandb/rl-run-reference.json", "provenance"))
        refs = [entry for entry in refs if entry["stage"] != "rl"] + [ref]
    for item in extra:
        if item.get("kind") == "wandb-run-reference":
            ref = _read_document(item["source"])
            refs = [entry for entry in refs if entry["stage"] != ref["stage"]] + [ref]
    wandb_inputs, wandb_status = _wandb_export(refs, settings["wandb"], prefix_join(metadata_root, "wandb"))
    inputs.extend(wandb_inputs)
    record("wandb-status.json", {"exporter": settings["wandb"]["package"], "runs": wandb_status}, "provenance")
    missing = {}
    full_sft = documents[("sft", "consumed-config.json")]
    expected_timing = {
        "native-sft-attempt-archive": full_sft["resources"]["replicas"],
        "native-sft-journal": full_sft["resources"]["replicas"] * full_sft["processes_per_task"],
    }
    timing_coverage = {
        kind: {"expected": count, "observed": len({item["source"] for item in extra if item.get("kind") == kind})}
        for kind, count in expected_timing.items()
    }
    if any(item["observed"] != item["expected"] for item in timing_coverage.values()):
        missing["timing"] = (
            "Native SFT attempt archive/journal coverage is incomplete; stage-wall journals do not substitute."
        )
        blockers.append(missing["timing"])
    if any(item["status"] != "strict_export_verified" for item in wandb_status):
        missing["wandb"] = (
            "One or more exact W&B identities or strict native-history exports unavailable; see wandb-status.json."
        )
        blockers.append(missing["wandb"])
    evaluation = documents[("report", "evaluation.json")]
    require_evaluation_coverage(evaluation)
    if not evaluation["all_verified"]:
        missing["evaluation"] = "Explicitly unscored attempts: " + json.dumps(
            evaluation["unscored_attempts"], sort_keys=True
        )
        limitations.append(missing["evaluation"])
        blockers.append(missing["evaluation"])
    record(
        "evidence-status.json",
        {
            "missing_evidence": missing,
            "limitations": limitations,
            "native_sft_timing_coverage": timing_coverage,
            "cleanup_blockers": blockers,
        },
        "provenance",
    )
    if evaluation["result_uri"] != evaluated:
        raise ValueError("Evaluation report belongs to another run")
    card = (
        "---\nlicense: openmdw-1.1\n---\n\n# Long-loop selected checkpoint\n\n"
        + WARNING
        + "\n\n"
        + f"Selected RL step: {run.global_step}. Selection: five-step EMA validation reward. "
        + f"Marin source commit: `{revision}`. Scored evaluation attempts: {evaluation['verified_rollouts']} "
        + f"of {evaluation['completed_rollouts']}. Unscored attempts are excluded from reward means.\n\n"
        + "This is an experimental training result, not a quality-parity claim. "
        + "Exact private dataset/config/provenance records are retained in the internal evidence archive.\n\n"
        + RESPONSIBLE_USE
        + "\n"
    )
    card_uri = prefix_join(metadata_root, "README.md")
    write_once(card_uri, card.encode())
    version = records["rl"]["version"]
    return {
        "inputs": inputs,
        "hf_upload": {
            "source": run.hf_model_uri,
            "repo_id": settings["hf_upload"]["repo_prefix"] + "-" + version,
            "local_directory": settings["hf_upload"]["local_directory"],
            "model_card": card_uri,
            "license_file": settings["hf_upload"]["license_file"],
            "native_checkpoint": "resolved-from-best-checkpoint",
            "workers": settings["hf_upload"]["workers"],
        },
        "native_checkpoint_ttl": settings["native_checkpoint_ttl"],
        "provenance": {
            "marin_commit": revision,
            "rl_terminal": documents[("rl", "terminal.json")]["result"],
            "evaluation_complete": evaluation["all_verified"],
            "wandb_runs": wandb_status,
            "missing_evidence": missing,
            "cleanup_blockers": blockers,
            "limitations": limitations,
        },
        "evaldash": {
            "source": eval_source,
            "records_prefix": settings["evaldash"]["records_prefix"],
            "record": {
                "run_id": settings["evaldash"]["run_prefix"] + "-" + version,
                "group_id": resolved_spec["name"],
                "created_at": datetime.now(UTC).isoformat(),
                "user": settings["evaldash"]["user"],
                "model": {"name": settings["evaldash"]["model_name"], "location": run.hf_model_uri, "backend": "vllm"},
                "eval": {"name": settings["evaldash"]["eval_name"], "mechanism": "skyrl"},
                "hardware": settings["evaldash"]["hardware"],
                "status": "succeeded" if evaluation["all_verified"] else "infra_failed",
                "error": None if evaluation["all_verified"] else missing["evaluation"],
                "results_path": prefix_join(output, "evaluation"),
                "metrics": {},
                "jobs": {},
                "log_tails": {},
                "provenance": {"git_sha": revision, "launch_host": "archive-postprocessing"},
            },
        },
    }


def archive_graph_run(
    *,
    output: str,
    settings: dict,
    resolved_spec: dict,
    artifacts: dict[str, str],
    trained: str,
    evaluated: str,
    completed_report: str,
) -> None:
    """Archive a completed graph without replaying any producing stage."""
    request = {
        "settings": settings,
        "resolved_spec": resolved_spec,
        "artifacts": artifacts,
        "trained": trained,
        "evaluated": evaluated,
        "completed_report": completed_report,
    }
    request_sha = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    uri = prefix_join(output, "runtime-plan.json")
    fs, key = filesystem_for(uri)
    if fs.exists(key):
        saved = json.loads(fs.cat_file(key))
        if saved["request_sha256"] != request_sha:
            raise ValueError("Runtime archive inputs changed on resume")
        plan = saved["plan"]
    else:
        plan = derive_archive_plan(output=output, **request)
        write_once(uri, json.dumps({"request_sha256": request_sha, "plan": plan}, sort_keys=True, indent=2).encode())
    archive_selected_run(output=output, plan=plan, trained=trained, completed_report=completed_report)


def selected_archive_step(
    version: str, plan: dict, trained: ArtifactStep[SkyRLRun], completed_report: ArtifactStep
) -> ArtifactStep:
    """Append a cluster-local HF upload and evidence archive after full evaluation."""
    return apply(
        "long-loop/archive",
        remote(
            archive_selected_run,
            name=f"long-loop-hf-archive-{version}",
            resources=ResourceConfig.with_gpu("H100", count=8, cpu=32, ram="64GB", disk="500GB", target_cluster=None),
        ),
        version=version,
        output=OUT,
        plan=plan,
        trained=trained,
        completed_report=completed_report,
    )


@click.command(help=__doc__)
@click.option("--plan", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--output", required=True, help="Durable destination without a TTL lifecycle.")
def main(plan: Path, output: str) -> None:
    inventory = json.loads(plan.read_text())
    archive_run(output=output, plan=inventory, completed_report=inventory["completed_report"])


if __name__ == "__main__":
    main()

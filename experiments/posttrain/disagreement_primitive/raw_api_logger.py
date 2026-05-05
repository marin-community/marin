# ruff: noqa: E501
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""RawAPILogger — wrapper around any LM API call that always persists the full
SDK response (or the exception) to disk before returning to the caller.

WHY THIS EXISTS
---------------
Earlier disagreement-primitive experiments (E7, E7v2) silently truncated each
generator response to 120 chars and each user query to 200 chars at save time,
making the resulting jsonl artifacts non-auditable and non-reusable. The actual
API calls returned full text — only the saved record was lossy. This module
enforces that every LM call routes through `.call(...)`, which writes the full
response to a timestamped directory before the caller can drop a single byte.

PROJECT RULE: every LM API call in this directory MUST be wrapped with
RawAPILogger.call(...). Never call `.chat.completions.create(...)` or
`.models.generate_content(...)` directly. If you find yourself wanting to,
you're reintroducing the bug.

DIRECTORY LAYOUT
----------------
    results/raw/<experiment>/<UTC-timestamp>/<role>/<seq>__<key-pairs>__<nonce>.json

For example, an E8 phase-1 run might produce:
    results/raw/e8_paired_indirection/2026-05-03T20-15-00/
      _manifest.json
      compiler/
        000001__statement_id=avoid_abuse__a1b2c3d4.json
        000002__statement_id=be_clear__e5f6g7h8.json
        ...
      judge_variant_a/
        000050__statement_id=avoid_abuse__scenario_idx=0__generator=gpt-5.1__...json
        ...
      judge_variant_b/
        ...

USAGE
-----
    from raw_api_logger import RawAPILogger

    log = RawAPILogger("e8_paired_indirection")

    raw = log.call(
        role="compiler",
        key={"statement_id": "avoid_abuse"},
        fn=lambda: openai_client.chat.completions.create(
            model="gpt-5.1",
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": USER}],
            temperature=0,
            reasoning_effort="none",
            response_format={"type": "json_object"},
        ),
    )
    # `raw` is the SDK response object — full text intact.
    # The full SDK response (model_dump) is now persisted at
    # results/raw/e8_paired_indirection/<ts>/compiler/<id>.json.

    # If the call raises, the exception, traceback, and request kwargs
    # are persisted with status="error" and re-raised.
"""

from __future__ import annotations
import json
import re
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

DEFAULT_BASE_DIR = Path("results/raw")

# Filenames are restricted to [A-Za-z0-9._-]. Anything else gets replaced with `-`.
_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9_.\-]")


def _safe(s: Any) -> str:
    return _FILENAME_SAFE.sub("-", str(s))


class RawAPILogger:
    """Per-experiment raw-response logger. Thread-safe.

    One instance creates one timestamped run directory under `base_dir/experiment/`.
    Subsequent invocations of the same experiment create new timestamped directories
    so concurrent or sequential runs never clobber each other.
    """

    def __init__(self, experiment_name: str, base_dir: Path | str = DEFAULT_BASE_DIR) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        self.experiment_name = experiment_name
        self.run_dir = Path(base_dir) / experiment_name / ts
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._counter = 0
        manifest = self.run_dir / "_manifest.json"
        manifest.write_text(json.dumps({
            "experiment": experiment_name,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "base_dir": str(base_dir),
        }, indent=2))

    def call(self, *, role: str, key: dict[str, Any], fn: Callable[[], Any]) -> Any:
        """Execute fn(); persist raw response (or exception) to disk; re-raise on error.

        Args:
            role: a stable identifier for the API call's purpose (e.g., "compiler",
                "judge_variant_a", "generator_gpt"). Files for each role land in
                their own subdirectory.
            key: a flat dict of identifiers (e.g., {"statement_id": ..., "scenario_idx": ...}).
                Encoded in the filename for human-debuggability.
            fn: zero-arg callable that performs the API call and returns the SDK response.

        Returns:
            Whatever fn() returned (the SDK response object).

        Side effects:
            Writes one JSON file under `<run_dir>/<role>/`. On exception, status="error"
            and the exception is persisted before being re-raised.
        """
        start = time.time()
        record: dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "key": key,
        }
        try:
            raw = fn()
            record["status"] = "ok"
            record["response"] = self._dump(raw)
            record["wall_time_s"] = round(time.time() - start, 4)
            self._save(role, key, record)
            return raw
        except Exception as exc:
            record["status"] = "error"
            record["error_class"] = exc.__class__.__name__
            record["error_message"] = str(exc)
            record["traceback"] = traceback.format_exc()
            record["wall_time_s"] = round(time.time() - start, 4)
            self._save(role, key, record)
            raise

    @staticmethod
    def _dump(obj: Any) -> Any:
        """Best-effort full-fidelity serialization of an SDK response object."""
        if hasattr(obj, "model_dump"):  # Pydantic v2 — OpenAI / google-genai
            try:
                return obj.model_dump()
            except Exception:
                pass
        if hasattr(obj, "to_dict"):
            try:
                return obj.to_dict()
            except Exception:
                pass
        if hasattr(obj, "__dict__"):
            try:
                return json.loads(json.dumps(obj.__dict__, default=str))
            except Exception:
                pass
        try:
            return json.loads(json.dumps(obj, default=str))
        except Exception:
            return repr(obj)

    def _save(self, role: str, key: dict[str, Any], record: dict[str, Any]) -> None:
        with self._lock:
            self._counter += 1
            seq = self._counter
        role_dir = self.run_dir / _safe(role)
        role_dir.mkdir(parents=True, exist_ok=True)
        key_part = "__".join(f"{_safe(k)}={_safe(v)}" for k, v in key.items()) if key else "no_key"
        nonce = uuid.uuid4().hex[:8]
        filename = f"{seq:06d}__{key_part}__{nonce}.json"
        # Most filesystems cap filename at 255 bytes; truncate the key part if needed.
        if len(filename.encode("utf-8")) > 240:
            filename = f"{seq:06d}__{nonce}.json"
        path = role_dir / filename
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str))
        tmp.rename(path)


def _smoke_test() -> int:
    """Self-test: verify the logger persists both success and error records.

    Run with `python raw_api_logger.py`. Uses a temporary directory so production
    artifacts are untouched.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        log = RawAPILogger("smoke_test", base_dir=Path(tmp))

        # success
        out = log.call(role="test_ok", key={"i": 0, "tag": "hello/world"}, fn=lambda: {"ok": True})
        assert out == {"ok": True}

        # failure
        try:
            log.call(role="test_err", key={"i": 1}, fn=lambda: 1 / 0)
        except ZeroDivisionError:
            pass
        else:
            print("FAIL: expected ZeroDivisionError to propagate")
            return 1

        # parallel writes
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(lambda i: log.call(role="test_par", key={"i": i}, fn=lambda i=i: {"i": i}), range(20)))

        # checks
        all_files = sorted(log.run_dir.rglob("*.json"))
        all_files = [f for f in all_files if f.name != "_manifest.json"]
        assert len(all_files) == 22, f"expected 22 files (1 ok + 1 err + 20 par), got {len(all_files)}"

        ok_files = [f for f in all_files if "test_ok" in str(f)]
        assert len(ok_files) == 1
        ok_record = json.loads(ok_files[0].read_text())
        assert ok_record["status"] == "ok"
        assert ok_record["response"] == {"ok": True}
        assert ok_record["key"] == {"i": 0, "tag": "hello/world"}
        # filename should sanitize the slash
        assert "/" not in ok_files[0].name
        assert "tag=hello-world" in ok_files[0].name

        err_files = [f for f in all_files if "test_err" in str(f)]
        assert len(err_files) == 1
        err_record = json.loads(err_files[0].read_text())
        assert err_record["status"] == "error"
        assert err_record["error_class"] == "ZeroDivisionError"
        assert "Traceback" in err_record["traceback"]

        par_files = [f for f in all_files if "test_par" in str(f)]
        assert len(par_files) == 20, f"parallel writes lost: {len(par_files)}/20"
        seen_i = sorted(json.loads(f.read_text())["key"]["i"] for f in par_files)
        assert seen_i == list(range(20)), f"parallel content corrupted: {seen_i}"

        manifest_path = log.run_dir / "_manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["experiment"] == "smoke_test"

        print(f"OK: 22 records persisted under {log.run_dir}")
        print(f"     dirs: {sorted(p.name for p in log.run_dir.iterdir() if p.is_dir())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_smoke_test())

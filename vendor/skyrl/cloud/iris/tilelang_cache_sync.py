#!/usr/bin/env python3
"""TileLang JIT-cache sync shim for CoreWeave iris GDN/FlashQLA RL runs (Fix A).

FlashQLA runs JIT-compile the GatedDeltaNet TileLang
kernels lazily on the first GPU forward into the node-local, ephemeral TileLang
cache (``~/.tilelang/cache`` == ``/root/.tilelang/cache`` for the root pod user).
Kaniko is CPU-only, so those kernels CANNOT be baked into the gpu-rl image at
build time — so every run, and every one of the N gang pods, cold-compiles them
(~71 min on the first r4f run). This shim makes each pod boot with a WARM cache:

  --down  (pod boot, BEFORE the trainer's first GDN forward / TileLang first-call):
          download the keyed warm cache from the CW object store and unpack it
          into ``TILELANG_CACHE_DIR`` so TileLang hash-matches and reuses the
          compiled ``.so`` instead of recompiling. A cache MISS (no key yet,
          S3 hiccup) is a warn-and-continue no-op -> the pod just cold-compiles
          exactly as it does today. NEVER fails the job.

  --up    (pod exit): upload any NEW kernel hash-dirs the pod compiled back under
          the same key so future runs / pods reuse them. Best-effort; a failed
          upload NEVER fails the job.

Reuse mechanism (verified in-pod): the cache is content-hash-addressed — each
compiled kernel lives in ``<TILELANG_CACHE_DIR>/<hash>/`` (``executable.so`` +
``.cu`` + ``params.pkl``). Dropping a hash-dir tree into ``TILELANG_CACHE_DIR``
is SAFE: TileLang hash-matches on the next call; a miss just recompiles (no
corruption). So the sync is purely additive.

16-writer upload race -> PER-HASH-DIR objects (design option (a), chosen).
All N (~16) gang pods compile the SAME content-addressed kernels and each runs
``--up`` at exit. If they all re-tarred+overwrote one ``cache.tgz``, last-writer
wins would DROP kernels other pods compiled. Instead ``--up`` uploads each hash
dir as INDIVIDUAL objects under ``.../<key>/kernels/<hash>/...``. Because the
path IS the content hash, concurrent writers of the same kernel write identical
bytes to the same key (idempotent, no race), and a pod that compiled a NEW
kernel adds it without touching anyone else's. ``--down`` pulls the whole
``kernels/`` prefix (plus the seed ``cache.tgz``). This is race-free and
incremental — no rank-0 gating, no merge step.

Key derivation: ``gdn-flashqla/<model>-<gpu-arch>-<tilelang-tag>``, e.g. the
r4f-lineage ``gdn-flashqla/qwen3-next-80b-a3b-sm90-tilelang018``. Components:
  - model:    ``TILELANG_CACHE_MODEL`` env if set, else normalized from
              ``TILELANG_CACHE_MODEL_PATH`` / ``MODEL_PATH`` (the Qwen3-Next-80B
              family collapses to the canonical ``qwen3-next-80b-a3b``), else the
              default ``qwen3-next-80b-a3b``.
  - gpu-arch: ``torch.cuda.get_device_capability()`` -> ``sm<major><minor>``
              (H100 -> ``sm90``); falls back to ``sm90``.
  - tilelang: ``importlib.metadata.version("tilelang")`` with dots stripped
              (``0.1.8`` -> ``tilelang018``).

Credentials/endpoint are read from the pod env (NEVER hardcoded): the cluster
projects ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` / ``AWS_ENDPOINT_URL``
(= ``http://cwlota.com``, the CoreWeave LOTA object store, NOT AWS) /
``AWS_REGION`` into every task pod via the ``iris-task-env`` Secret. LOTA rejects
path-style addressing, so boto3 MUST use virtual addressing
(``Config(s3={"addressing_style":"virtual"})``) — matching the pod's ``FSSPEC_S3``
setting and the coreweave_gpu_ops.md in-pod one-liner.

The launcher invokes this shim only when ``generator.gdn_backend=flashqla``. A
missing cache key is a no-op, so cache availability never controls correctness.

Standalone by design (no launcher imports) so it runs identically from the pod's
bash wrapper regardless of PYTHONPATH state.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional

# --------------------------------------------------------------------------- #
# Constants (all overridable via env; NO credentials ever hardcoded).
# --------------------------------------------------------------------------- #
DEFAULT_BUCKET = "marin-us-east-02a"  # CW object store; override via TILELANG_CACHE_BUCKET
CACHE_ROOT_PREFIX = "iris/tilelang-cache"  # s3://<bucket>/iris/tilelang-cache/<key>/...
SEED_TARBALL_NAME = "cache.tgz"  # the pre-captured warm cache (expands to cache/<hash>/...)
KERNELS_SUBPREFIX = "kernels"  # per-hash incremental objects: .../<key>/kernels/<hash>/...
DEFAULT_MODEL_KEY = "qwen3-next-80b-a3b"  # r4f lineage default
DEFAULT_GPU_ARCH = "sm90"  # H100
# Files that identify a directory as a TileLang compiled-kernel hash-dir.
KERNEL_SENTINELS = ("executable.so", "params.pkl", "kernel_lib.so", "wrapped_kernel.cu")
# Best-effort marker written by --down listing the hashes present at boot, so
# --up only uploads NEWLY-compiled dirs (minimizes redundant S3 writes).
BASELINE_MARKER = ".otagent_tilelang_baseline.json"


def _log(msg: str) -> None:
    print(f"[tilelang-sync] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Key derivation (pure, unit-testable).
# --------------------------------------------------------------------------- #
def normalize_model_key(model_path: Optional[str]) -> str:
    """Normalize a model path/repo-id into the cache-key model component.

    The Qwen3-Next-80B family (any instruct/date variant) collapses to the
    canonical ``qwen3-next-80b-a3b`` so an r4f-seeded cache is reused across the
    lineage. Anything else is slugified (lowercase, non-alnum -> ``-``). An empty
    input falls back to the r4f-lineage default.
    """
    if not model_path:
        return DEFAULT_MODEL_KEY
    name = str(model_path).rstrip("/").split("/")[-1].lower()
    if not name:
        return DEFAULT_MODEL_KEY
    if "qwen3-next-80b" in name or "qwen3.5-next-80b" in name:
        return "qwen3-next-80b-a3b"
    return re.sub(r"[^a-z0-9]+", "-", name).strip("-") or DEFAULT_MODEL_KEY


def derive_model_key() -> str:
    explicit = os.environ.get("TILELANG_CACHE_MODEL")
    if explicit and explicit.strip():
        return explicit.strip()
    mp = os.environ.get("TILELANG_CACHE_MODEL_PATH") or os.environ.get("MODEL_PATH") or ""
    return normalize_model_key(mp)


def derive_gpu_arch() -> str:
    """``sm<major><minor>`` from the live device; falls back to sm90 (H100)."""
    ovr = os.environ.get("TILELANG_CACHE_GPU_ARCH")
    if ovr and ovr.strip():
        return ovr.strip()
    try:
        import torch  # noqa: PLC0415 - optional/heavy; only imported when acting

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return f"sm{major}{minor}"
    except Exception as exc:  # noqa: BLE001 - best-effort
        _log(f"WARN could not read GPU capability ({type(exc).__name__}: {exc}); defaulting to {DEFAULT_GPU_ARCH}")
    return DEFAULT_GPU_ARCH


def derive_tilelang_tag() -> str:
    """``tilelang<version-without-dots>`` (0.1.8 -> tilelang018)."""
    ovr = os.environ.get("TILELANG_CACHE_TILELANG_TAG")
    if ovr and ovr.strip():
        return ovr.strip()
    try:
        from importlib.metadata import version  # noqa: PLC0415

        v = version("tilelang")
        return "tilelang" + re.sub(r"[^0-9]", "", v)
    except Exception as exc:  # noqa: BLE001 - best-effort
        _log(f"WARN could not read tilelang version ({type(exc).__name__}: {exc})")
        return "tilelangunknown"


def derive_cache_key() -> str:
    """The full content-key: gdn-flashqla/<model>-<gpu>-<tilelang>."""
    ovr = os.environ.get("TILELANG_CACHE_KEY")
    if ovr and ovr.strip():
        return ovr.strip().strip("/")
    return f"gdn-flashqla/{derive_model_key()}-{derive_gpu_arch()}-{derive_tilelang_tag()}"


def cache_dir() -> Path:
    """The node-local TileLang cache dir. Matches TileLang's default so a hash
    written by the trainer and a hash seeded here land in the SAME place. The
    bash wrapper exports the identical default before both the trainer and this
    shim, so they always agree."""
    return Path(os.environ.get("TILELANG_CACHE_DIR", "/root/.tilelang/cache"))


# --------------------------------------------------------------------------- #
# boto3 client (virtual addressing — LOTA rejects path-style).
# --------------------------------------------------------------------------- #
def make_s3_client():
    """boto3 S3 client for the CW LOTA store. Creds come from the pod's injected
    env; endpoint from AWS_ENDPOINT_URL. Returns None (warn) if boto3 or the
    endpoint is unavailable — the caller then no-ops (cold, no failure)."""
    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    if not endpoint:
        _log("WARN AWS_ENDPOINT_URL unset (not in a CW task pod?); skipping S3 sync")
        return None
    try:
        import boto3  # noqa: PLC0415
        from botocore.config import Config  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        _log(f"WARN boto3 unavailable ({type(exc).__name__}: {exc}); skipping S3 sync")
        return None
    # Virtual addressing matches the pod's FSSPEC_S3 setting; path-style is
    # rejected by LOTA (PathStyleRequestNotAllowed). Creds auto-discovered from env.
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        config=Config(s3={"addressing_style": "virtual"}, retries={"max_attempts": 3}),
    )


def bucket_name() -> str:
    return os.environ.get("TILELANG_CACHE_BUCKET", DEFAULT_BUCKET)


def _iter_hash_dirs(root: Path) -> Iterable[Path]:
    """Yield the compiled-kernel hash-dirs directly under ``root`` (a dir that
    contains any kernel sentinel file)."""
    if not root.is_dir():
        return
    for child in sorted(root.iterdir()):
        if child.is_dir() and any((child / s).exists() for s in KERNEL_SENTINELS):
            yield child


def _read_baseline(root: Path) -> set[str]:
    marker = root / BASELINE_MARKER
    try:
        return set(json.loads(marker.read_text()))
    except Exception:  # noqa: BLE001 - absent/corrupt marker => empty baseline
        return set()


def _write_baseline(root: Path, hashes: Iterable[str]) -> None:
    try:
        root.mkdir(parents=True, exist_ok=True)
        (root / BASELINE_MARKER).write_text(json.dumps(sorted(set(hashes))))
    except Exception as exc:  # noqa: BLE001
        _log(f"WARN could not write baseline marker ({type(exc).__name__}: {exc})")


# --------------------------------------------------------------------------- #
# --down: seed the node-local cache from S3.
# --------------------------------------------------------------------------- #
def run_down() -> int:
    key = derive_cache_key()
    root = cache_dir()
    _log(f"--down: key={key!r} cache_dir={root} bucket={bucket_name()}")
    s3 = make_s3_client()
    if s3 is None:
        return 0
    bucket = bucket_name()
    base_prefix = f"{CACHE_ROOT_PREFIX}/{key}"
    root.mkdir(parents=True, exist_ok=True)

    pulled = 0
    # (1) the pre-captured warm seed tarball (expands to cache/<hash>/...).
    pulled += _pull_seed_tarball(s3, bucket, f"{base_prefix}/{SEED_TARBALL_NAME}", root)
    # (2) any incremental per-hash-dir objects a prior --up pushed.
    pulled += _pull_kernels_prefix(s3, bucket, f"{base_prefix}/{KERNELS_SUBPREFIX}/", root)

    present = [p.name for p in _iter_hash_dirs(root)]
    _write_baseline(root, present)
    _log(f"--down: done — {pulled} kernel object(s)/tarball(s) pulled; {len(present)} hash-dir(s) now warm in {root}")
    return 0


def _pull_seed_tarball(s3, bucket: str, key: str, root: Path) -> int:
    """Download+unpack the seed cache.tgz into ``root`` so kernels land at
    ``root/<hash>/``. The tarball may wrap the hashes under a top-level ``cache/``
    dir (its capture layout) or not — both are handled. Missing = no-op."""
    from botocore.exceptions import ClientError  # noqa: PLC0415

    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = obj["Body"].read()
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "404", "NoSuchBucket"):
            _log(f"--down: no seed tarball at s3://{bucket}/{key} (cold start OK)")
        else:
            _log(f"--down: WARN seed tarball fetch failed ({code}: {exc}); continuing cold")
        return 0
    except Exception as exc:  # noqa: BLE001
        _log(f"--down: WARN seed tarball fetch error ({type(exc).__name__}: {exc}); continuing cold")
        return 0

    copied = 0
    try:
        with tempfile.TemporaryDirectory() as tmp:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
                _safe_extract(tf, tmp)
            tmp_path = Path(tmp)
            # The hash-dirs are either under tmp/cache/ or directly under tmp.
            src = tmp_path / "cache" if (tmp_path / "cache").is_dir() else tmp_path
            for hd in _iter_hash_dirs(src):
                dst = root / hd.name
                if dst.exists():
                    continue  # content-addressed; identical bytes, keep existing
                _copytree(hd, dst)
                copied += 1
    except Exception as exc:  # noqa: BLE001
        _log(f"--down: WARN seed tarball unpack error ({type(exc).__name__}: {exc}); continuing cold")
        return 0
    _log(f"--down: seed tarball unpacked ({copied} new hash-dir(s)) from s3://{bucket}/{key}")
    return copied


def _pull_kernels_prefix(s3, bucket: str, prefix: str, root: Path) -> int:
    """Download every object under ``.../kernels/`` into ``root/<hash>/...``.
    Content-addressed, so an already-present file is skipped. Missing = no-op."""
    n = 0
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []) or []:
                s3key = obj["Key"]
                rel = s3key[len(prefix) :]
                if not rel or rel.endswith("/"):
                    continue
                dst = root / rel
                if dst.exists():
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    s3.download_file(bucket, s3key, str(dst))
                    n += 1
                except Exception as exc:  # noqa: BLE001
                    _log(f"--down: WARN could not download {s3key} ({type(exc).__name__}: {exc})")
    except Exception as exc:  # noqa: BLE001
        _log(f"--down: WARN listing {prefix} failed ({type(exc).__name__}: {exc}); continuing with whatever seeded")
    if n:
        _log(f"--down: pulled {n} incremental kernel file(s) from s3://{bucket}/{prefix}")
    return n


# --------------------------------------------------------------------------- #
# --up: upload NEW kernel hash-dirs back under the key (per-hash-dir objects).
# --------------------------------------------------------------------------- #
def run_up() -> int:
    key = derive_cache_key()
    root = cache_dir()
    _log(f"--up: key={key!r} cache_dir={root} bucket={bucket_name()}")
    s3 = make_s3_client()
    if s3 is None:
        return 0
    bucket = bucket_name()
    base_prefix = f"{CACHE_ROOT_PREFIX}/{key}/{KERNELS_SUBPREFIX}"

    baseline = _read_baseline(root)
    hash_dirs = list(_iter_hash_dirs(root))
    if not hash_dirs:
        _log(f"--up: no compiled kernel hash-dirs under {root}; nothing to push")
        return 0

    uploaded_dirs = 0
    uploaded_files = 0
    for hd in hash_dirs:
        h = hd.name
        if h in baseline:
            continue  # was already warm at boot — no need to re-push
        dir_prefix = f"{base_prefix}/{h}"
        # Idempotency belt: skip a hash-dir already fully present in S3 (a sibling
        # pod or a prior run pushed it). Cheap HEAD on a sentinel file.
        if _hash_dir_present(s3, bucket, dir_prefix, hd):
            continue
        pushed = _upload_hash_dir(s3, bucket, dir_prefix, hd)
        if pushed:
            uploaded_dirs += 1
            uploaded_files += pushed
    _log(
        f"--up: done — pushed {uploaded_dirs} new hash-dir(s) "
        f"({uploaded_files} file(s)) to s3://{bucket}/{base_prefix}/"
    )
    return 0


def _hash_dir_present(s3, bucket: str, dir_prefix: str, hd: Path) -> bool:
    """True if a sentinel object for this hash-dir already exists in S3."""
    from botocore.exceptions import ClientError  # noqa: PLC0415

    for s in KERNEL_SENTINELS:
        if not (hd / s).exists():
            continue
        try:
            s3.head_object(Bucket=bucket, Key=f"{dir_prefix}/{s}")
            return True
        except ClientError:
            return False
        except Exception:  # noqa: BLE001 - treat as absent, attempt upload
            return False
    return False


def _upload_hash_dir(s3, bucket: str, dir_prefix: str, hd: Path) -> int:
    n = 0
    for f in sorted(hd.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(hd).as_posix()
        s3key = f"{dir_prefix}/{rel}"
        try:
            s3.upload_file(str(f), bucket, s3key)
            n += 1
        except Exception as exc:  # noqa: BLE001
            _log(f"--up: WARN could not upload {f} -> {s3key} ({type(exc).__name__}: {exc})")
    if n:
        _log(f"--up: pushed hash-dir {hd.name} ({n} file(s))")
    return n


# --------------------------------------------------------------------------- #
# Helpers: safe tar extract + copytree (stdlib-only, py3.12).
# --------------------------------------------------------------------------- #
def _safe_extract(tf: tarfile.TarFile, dest: str) -> None:
    """Extract guarding against path traversal (no member escapes ``dest``)."""
    dest_path = Path(dest).resolve()
    for member in tf.getmembers():
        target = (dest_path / member.name).resolve()
        if not str(target).startswith(str(dest_path)):
            raise RuntimeError(f"unsafe tar member path: {member.name!r}")
    # py3.12 supports the 'data' filter; fall back if unavailable.
    try:
        tf.extractall(dest, filter="data")
    except TypeError:
        tf.extractall(dest)


def _copytree(src: Path, dst: Path) -> None:
    import shutil  # noqa: PLC0415

    shutil.copytree(src, dst, dirs_exist_ok=True)


# --------------------------------------------------------------------------- #
# Entrypoint. ALWAYS exits 0 (best-effort): a sync failure must NEVER kill the job.
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--down", action="store_true", help="pull+unpack the warm cache at pod boot")
    mode.add_argument("--up", action="store_true", help="push newly-compiled kernels at pod exit")
    args = parser.parse_args(argv)

    try:
        if args.down:
            run_down()
        else:
            run_up()
    except Exception as exc:  # noqa: BLE001 - the shim must never fail the job
        _log(
            f"WARN unexpected error in {'--down' if args.down else '--up'} "
            f"({type(exc).__name__}: {exc}); continuing (best-effort)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

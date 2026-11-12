#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
TRAIN_CACHE=${TRAIN_CACHE:?Set TRAIN_CACHE to the complete FineWeb-Edu 10B cache}
EVAL_CACHE=${EVAL_CACHE:?Set EVAL_CACHE to the complete tokenized Paloma C4 validation cache}
for ledger in "$TRAIN_CACHE/train/shard_ledger.json" "$EVAL_CACHE/validation/shard_ledger.json"; do
  test -s "$ledger" || { echo "Missing cache ledger: $ledger" >&2; exit 1; }
done
components=$("$PYTHON_BIN" - "$TRAIN_CACHE" "$EVAL_CACHE" <<'PYCODE'
import json
import sys
from pathlib import Path

for path, split in zip(sys.argv[1:], ["train", "validation"]):
    ledger = Path(path) / split / "shard_ledger.json"
    if not json.loads(ledger.read_text()).get("is_finished"):
        raise SystemExit(f"Incomplete cache: {ledger}")
print(json.dumps({
    "fineweb_edu_10b": {"cache_dir": str(Path(sys.argv[1]).resolve()), "source": {"type": "url"}},
    "paloma/c4_en": {"cache_dir": str(Path(sys.argv[2]).resolve()), "source": {"type": "url"}},
}))
PYCODE
)

exec "$PYTHON_BIN" -m levanter.main.train_lm \
 --config_path "$ROOT/lib/levanter/config/qwen3_130m_rmnph.yaml" \
 --data.components "$components" \
 --trainer.per_device_parallelism "${MICRO_BATCH_SIZE:-8}" \
 --trainer.seed "${SEED:-0}" --data_seed 42 "$@"

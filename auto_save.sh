#!/bin/bash
set -e

REPO_DIR="$HOME/marin_scaling"   # ← 改成你的代码目录
BRANCH="main"
INTERVAL=3600              # 秒

cd "$REPO_DIR"

echo "[autosave] started at $(date)"

while true; do
  (
    # 防止并发
    exec 9>.git/.autosave.lock || exit 0
    flock -n 9 || exit 0

    git add -A

    if git diff --cached --quiet; then
      echo "[autosave] no changes $(date)"
      exit 0
    fi

    git commit -m "autosave: $(date '+%Y-%m-%d %H:%M:%S')"

    git push origin "$BRANCH" || echo "[autosave] push failed"
  )

  sleep "$INTERVAL"
done
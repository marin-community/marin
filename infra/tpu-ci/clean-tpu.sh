#!/bin/bash
set -e

echo "Cleaning up TPU resources..."

# Remove TPU lockfile
rm -f /tmp/libtpu_lockfile || true

# Find and kill processes using /dev/vfio devices by scanning /proc
for pid in /proc/[0-9]*; do
  [ -d "$pid/fd" ] || continue
  pid_num=$(basename "$pid")
  for fd in "$pid"/fd/*; do
    [ -L "$fd" ] || continue
    if readlink "$fd" 2>/dev/null | grep -q "^/dev/vfio/"; then
      echo "Killing process $pid_num using TPU device"
      kill -9 "$pid_num" 2>/dev/null || true
      break
    fi
  done
done

# Wait for /dev/vfio/0 to be actually available (not just process-free)
for i in {1..30}; do
  if timeout 1 python3 -c "open('/dev/vfio/0', 'r')" 2>/dev/null; then
    echo "/dev/vfio/0 is ready"
    break
  fi
  echo "Waiting for /dev/vfio/0... attempt $i"
  sleep 1
done

echo "TPU cleanup complete"

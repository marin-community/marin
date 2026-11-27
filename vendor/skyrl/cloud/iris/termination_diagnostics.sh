#!/usr/bin/env bash

echo "--- df -h /tmp /dev/shm ---"
timeout 7 df -h /tmp /dev/shm 2>&1
echo "--- top /tmp disk hogs (ephemeral-storage eviction cause) ---"
timeout 9 du -sh /tmp/* /tmp/ray/session*/logs /tmp/ray/session*/*spill* 2>/dev/null | sort -rh | head -25
echo "--- nvidia-smi (VRAM OOM cause) ---"
timeout 7 nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv 2>&1
echo "--- top RSS procs (host-RAM OOM cause) ---"
timeout 7 ps -eo pid,rss,comm --sort=-rss 2>/dev/null | head -12
echo "--- dmesg OOM/kill tail ---"
timeout 7 dmesg 2>/dev/null | grep -iE 'oom|killed process|out of memory|Xid' | tail -10

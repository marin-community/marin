#!/bin/bash
set -e

echo "Cleaning up TPU resources..."

# Remove TPU lockfile and logs directory
rm -f /tmp/libtpu_lockfile || true
rm -rf /tmp/tpu_logs || true

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

# Reset TPU PCI devices via sysfs
for pci_dev in /sys/bus/pci/devices/*; do
  if [ -d "$pci_dev/iommu_group" ]; then
    driver_path=$(readlink -f "$pci_dev/driver" 2>/dev/null || echo "")
    if [[ "$driver_path" == *"vfio-pci"* ]] && [ -f "$pci_dev/reset" ]; then
      echo "Resetting PCI device $(basename $pci_dev)"
      echo 1 | sudo tee "$pci_dev/reset" > /dev/null 2>&1 || echo "  Reset failed"
    fi
  fi
done

# Wait for /dev/vfio/0 to be actually available (not just process-free)
for i in {1..30}; do
  if timeout 1 python3 -c "open('/dev/vfio/0', 'rb')" 2>/dev/null; then
    echo "/dev/vfio/0 is ready"
    break
  fi
  echo "Waiting for /dev/vfio/0... attempt $i"
  sleep 1
done

echo "TPU cleanup complete"

#!/bin/bash
# Run debug-setup on all TPU VMs in parallel

set -e

VMS=(
    "tpu-ci-running-phoenix-us-west4-a-3i69"
    "tpu-ci-dashing-dolphin-us-west4-a-as31"
    "tpu-ci-soaring-mountain-us-east1-c-gi2s"
    "tpu-ci-flying-tiger-us-east1-c-fls5"
    "tpu-ci-running-mountain-europe-west4-b-eov4"
    "tpu-ci-climbing-canyon-europe-west4-b-ffs1"
)

LOG_DIR="/tmp/tpu-debug-logs"
mkdir -p "$LOG_DIR"

echo "Starting debug-setup on ${#VMS[@]} VMs in parallel..."
echo "Logs will be written to: $LOG_DIR"

# Launch all VMs in parallel
for vm in "${VMS[@]}"; do
    echo "Starting debug-setup for $vm..."
    (
        uv run vm_manager.py debug-setup "$vm" > "$LOG_DIR/$vm.log" 2>&1
        if [ $? -eq 0 ]; then
            echo "✓ $vm completed successfully"
        else
            echo "✗ $vm failed (see $LOG_DIR/$vm.log)"
        fi
    ) &
done

# Wait for all background jobs to complete
echo "Waiting for all VMs to complete..."
wait

echo ""
echo "All VMs completed. Summary:"
echo "========================================"
for vm in "${VMS[@]}"; do
    if grep -q "TPU VM Setup Complete" "$LOG_DIR/$vm.log" 2>/dev/null; then
        echo "✓ $vm: SUCCESS"
    else
        echo "✗ $vm: FAILED (check $LOG_DIR/$vm.log)"
    fi
done

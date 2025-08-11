#!/bin/bash

# TPU Auto-Relaunch Wrapper Script
# Provides easy commands for managing TPU auto-relaunch functionality

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/tpu_auto_relaunch.py"
LAUNCHER_SCRIPT="$SCRIPT_DIR/launcher.py"
TRAINING_SCRIPT="training_run.sh"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

show_usage() {
    cat << EOF
TPU Auto-Relaunch Wrapper Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    list-tpus                List all available TPUs
    start [TPU_NAME]         Start monitoring and auto-relaunch (optionally with specific TPU)
    check-tpu TPU_NAME ZONE  Check if a specific TPU is accessible
    setup-tpu TPU_NAME       Setup a specific TPU
    launch-job TPU_NAME      Launch job on a specific TPU
    check-checkpoints RUN_NAME  Check for existing checkpoints
    status                   Show current monitoring status
    stop                     Stop the monitoring service
    logs                     Show recent logs

Options:
    --check-interval SECONDS Set check interval (default: 60)
    --launcher-script PATH   Set launcher script path (default: launcher.py)
    --training-script PATH   Set training script path (default: training_run.sh)
    --help                   Show this help message

Examples:
    $0 list-tpus
    $0 start ray-marin-us-central1-worker-39ac1bd9-tpu
    $0 start --check-interval 30
    $0 check-tpu ray-marin-us-central1-worker-39ac1bd9-tpu us-central1-a
    $0 check-checkpoints random_acts_of_pizza_ckpt
    $0 status
    $0 logs

EOF
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if gcloud is installed and authenticated
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI is not installed or not in PATH"
        exit 1
    fi
    
    # Check if authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -1 &> /dev/null; then
        log_error "gcloud is not authenticated. Please run 'gcloud auth login'"
        exit 1
    fi
    
    # Check if Python script exists
    if [[ ! -f "$PYTHON_SCRIPT" ]]; then
        log_error "Python script not found: $PYTHON_SCRIPT"
        exit 1
    fi
    
    # Check if launcher script exists
    if [[ ! -f "$LAUNCHER_SCRIPT" ]]; then
        log_error "Launcher script not found: $LAUNCHER_SCRIPT"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

list_tpus() {
    log_info "Listing available TPUs..."
    python3 "$PYTHON_SCRIPT" --list-tpus
}

check_tpu_accessibility() {
    local tpu_name="$1"
    local zone="$2"
    
    if [[ -z "$tpu_name" || -z "$zone" ]]; then
        log_error "Usage: check-tpu TPU_NAME ZONE"
        exit 1
    fi
    
    log_info "Checking accessibility of TPU: $tpu_name in zone: $zone"
    
    if gcloud compute tpus tpu-vm ssh "$tpu_name" --zone "$zone" --command "echo tpu_accessible" --ssh-flag="-o ConnectTimeout=10" &> /dev/null; then
        log_info "✓ TPU $tpu_name is accessible"
        return 0
    else
        log_warn "✗ TPU $tpu_name is not accessible (possibly preempted)"
        return 1
    fi
}

check_checkpoints() {
    local run_name="$1"
    
    if [[ -z "$run_name" ]]; then
        log_error "Usage: check-checkpoints RUN_NAME"
        exit 1
    fi
    
    log_info "Checking for existing checkpoints for run: $run_name"
    
    if [[ -f "$SCRIPT_DIR/find_checkpoint.sh" ]]; then
        bash "$SCRIPT_DIR/find_checkpoint.sh" "$run_name"
    else
        log_error "Checkpoint detection script not found: $SCRIPT_DIR/find_checkpoint.sh"
        exit 1
    fi
}

setup_tpu() {
    local tpu_name="$1"
    
    if [[ -z "$tpu_name" ]]; then
        log_error "Usage: setup-tpu TPU_NAME"
        exit 1
    fi
    
    log_info "Setting up TPU: $tpu_name"
    
    if python3 "$LAUNCHER_SCRIPT" setup --project="$tpu_name"; then
        log_info "✓ TPU $tpu_name setup completed successfully"
    else
        log_error "✗ TPU $tpu_name setup failed"
        exit 1
    fi
}

launch_job() {
    local tpu_name="$1"
    
    if [[ -z "$tpu_name" ]]; then
        log_error "Usage: launch-job TPU_NAME"
        exit 1
    fi
    
    log_info "Launching job on TPU: $tpu_name"
    
    if python3 "$LAUNCHER_SCRIPT" launch "$TRAINING_SCRIPT" --project="$tpu_name"; then
        log_info "✓ Job launched successfully on TPU $tpu_name"
    else
        log_error "✗ Job launch failed on TPU $tpu_name"
        exit 1
    fi
}

start_monitoring() {
    local tpu_name=""
    local check_interval=60
    local launcher_script="$LAUNCHER_SCRIPT"
    local training_script="$TRAINING_SCRIPT"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --check-interval)
                check_interval="$2"
                shift 2
                ;;
            --launcher-script)
                launcher_script="$2"
                shift 2
                ;;
            --training-script)
                training_script="$2"
                shift 2
                ;;
            *)
                if [[ -z "$tpu_name" ]]; then
                    tpu_name="$1"
                fi
                shift
                ;;
        esac
    done
    
    log_info "Starting TPU monitoring and auto-relaunch service..."
    log_info "Check interval: ${check_interval}s"
    log_info "Launcher script: $launcher_script"
    log_info "Training script: $training_script"
    
    if [[ -n "$tpu_name" ]]; then
        log_info "Initial TPU: $tpu_name"
        python3 "$PYTHON_SCRIPT" \
            --tpu-name "$tpu_name" \
            --check-interval "$check_interval" \
            --launcher-script "$launcher_script" \
            --training-script "$training_script"
    else
        log_info "Will auto-select best available TPU"
        python3 "$PYTHON_SCRIPT" \
            --check-interval "$check_interval" \
            --launcher-script "$launcher_script" \
            --training-script "$training_script"
    fi
}

show_status() {
    log_info "Checking TPU monitoring status..."
    
    if pgrep -f "tpu_auto_relaunch.py" > /dev/null; then
        local pid=$(pgrep -f "tpu_auto_relaunch.py")
        log_info "✓ TPU monitoring service is running (PID: $pid)"
        
        # Show recent log entries
        if [[ -f "tpu_auto_relaunch.log" ]]; then
            log_info "Recent log entries:"
            tail -5 tpu_auto_relaunch.log | while IFS= read -r line; do
                echo "    $line"
            done
        fi
    else
        log_warn "✗ TPU monitoring service is not running"
    fi
}

stop_monitoring() {
    log_info "Stopping TPU monitoring service..."
    
    if pgrep -f "tpu_auto_relaunch.py" > /dev/null; then
        pkill -f "tpu_auto_relaunch.py"
        log_info "✓ TPU monitoring service stopped"
    else
        log_warn "TPU monitoring service was not running"
    fi
}

show_logs() {
    local lines=50
    
    if [[ "$1" =~ ^[0-9]+$ ]]; then
        lines=$1
    fi
    
    if [[ -f "tpu_auto_relaunch.log" ]]; then
        log_info "Showing last $lines log entries:"
        tail -n "$lines" tpu_auto_relaunch.log
    else
        log_warn "Log file not found: tpu_auto_relaunch.log"
    fi
}

main() {
    if [[ $# -eq 0 ]]; then
        show_usage
        exit 1
    fi
    
    local command="$1"
    shift
    
    case "$command" in
        list-tpus)
            check_prerequisites
            list_tpus
            ;;
        start)
            check_prerequisites
            start_monitoring "$@"
            ;;
        check-tpu)
            check_prerequisites
            check_tpu_accessibility "$@"
            ;;
        check-checkpoints)
            check_checkpoints "$@"
            ;;
        setup-tpu)
            check_prerequisites
            setup_tpu "$@"
            ;;
        launch-job)
            check_prerequisites
            launch_job "$@"
            ;;
        status)
            show_status
            ;;
        stop)
            stop_monitoring
            ;;
        logs)
            show_logs "$@"
            ;;
        --help|-h|help)
            show_usage
            ;;
        *)
            log_error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

main "$@"

#!/bin/bash

# Colors for pretty output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
ORANGE='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Helper function for printing steps
# log_start prints the message without a newline, allowing [SUCCESS] to be appended
log_start() { echo -n -e "${BLUE}[INFO]${NC} $1... "; }
# log_end prints [SUCCESS] and a newline
log_end() { echo -e "${GREEN}[SUCCESS]${NC}"; }

# For verbose commands that output text, we print a header first
log_header() { echo -e "${BLUE}[INFO]${NC} $1..."; }

log_warn() { echo -e "${ORANGE}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 1. Welcome Message
echo -e "${CYAN}"
echo "⚓ Welcome to Marin!"
echo -e "${NC}"

# ==========================================
# REPO CHECK & FORKING LOGIC
# ==========================================

is_marin_root() {
    if [[ -f "pyproject.toml" ]] && grep -q 'name = "marin-root"' pyproject.toml; then
        return 0
    else
        return 1
    fi
}

log_start "Checking repository root"
if is_marin_root; then
    log_end
else
    echo "" # Newline because log_start didn't print one and we are failing/warning
    # Check if a 'marin' directory exists in the current path
    if [[ -d "marin" ]]; then
        log_warn "Directory 'marin' found. Attempting to enter..."
        cd marin || { log_error "Failed to enter directory 'marin'. Check permissions."; exit 1; }

        if is_marin_root; then
            echo -e "${GREEN}[SUCCESS]${NC} Entered directory 'marin'."
        else
            log_error "Directory 'marin' exists but does not appear to be the valid project root (missing pyproject.toml configuration)."
            exit 1
        fi
    else
        log_warn "Current directory is not the root of the 'marin' repository."

        REPO_SETUP_DONE=false

        # Check for GitHub CLI and fork automatically
        if command -v gh &> /dev/null && gh auth status &> /dev/null; then
            echo -e "GitHub CLI detected and logged in."
            log_header "Forking and cloning marin"
            # Attempt to fork and clone directly without confirmation
            if gh repo fork marin-community/marin --clone --remote; then
                REPO_SETUP_DONE=true
            else
                log_warn "GitHub fork failed. Falling back to direct clone."
            fi
        fi

        # Fallback to direct clone if fork wasn't performed
        if [ "$REPO_SETUP_DONE" = false ]; then
            log_header "Cloning marin-community/marin (upstream)"
            if git clone https://github.com/marin-community/marin/; then
                REPO_SETUP_DONE=true
            else
                log_error "Failed to clone repository."
                exit 1
            fi
        fi

        # Verify directory entry
        if [[ -d "marin" ]]; then
            cd marin || { log_error "Failed to enter directory 'marin'."; exit 1; }
            log_end
        else
            log_error "Failed to find 'marin' directory after clone operation. Exiting."
            exit 1
        fi
    fi
fi

# ==========================================
# CUDA CHECK
# ==========================================

log_header "Checking CUDA configuration"

if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not found. Please ensure NVIDIA drivers are installed."
    exit 1
fi

# Show output to user
nvidia-smi

# Extract version using regex from the specific output format: "| ... CUDA Version: 12.2 ... |"
VERSION_STRING=$(nvidia-smi | grep "CUDA Version:" | head -n 1 | sed -n 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/p')
VERSION_STRING=$(echo "$VERSION_STRING" | xargs) # Trim whitespace

if [[ -z "$VERSION_STRING" ]]; then
    log_warn "Could not parse CUDA version automatically from nvidia-smi. Assuming valid configuration, but proceed with caution."
else
    MAJOR_VER=$(echo "$VERSION_STRING" | cut -d. -f1)
    MINOR_VER=$(echo "$VERSION_STRING" | cut -d. -f2)

    # Check for 12.1 or higher
    if [[ "$MAJOR_VER" -lt 12 ]]; then
        log_error "CUDA Version $VERSION_STRING detected. Marin requires CUDA 12.1 or higher."
        exit 1
    elif [[ "$MAJOR_VER" -eq 12 && "$MINOR_VER" -lt 1 ]]; then
         log_error "CUDA Version $VERSION_STRING detected. Marin requires CUDA 12.1 or higher."
         exit 1
    fi
fi

echo -e "${GREEN}CUDA environment detected ($VERSION_STRING).${NC}"

# ==========================================
# ENVIRONMENT SETUP (UV)
# ==========================================

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    log_warn "uv not found."
    echo -e "${CYAN}Press enter to install uv (https://docs.astral.sh/uv/)${NC}"
    read -r < /dev/tty

    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env 2>/dev/null || source $HOME/.local/bin/env 2>/dev/null
fi

log_header "Creating virtual environment"
uv venv --python 3.11 --clear
source .venv/bin/activate
log_end

log_header "Syncing dependencies (including CUDA 12)"
uv sync --all-packages --extra=cuda12
log_end

# ==========================================
# SECRETS (WANDB & HF)
# ==========================================

log_start "Checking API keys"
VARS_UPDATED=false
MISSING_KEYS=false

# Check variables silently first
if [[ -z "$WANDB_API_KEY" ]] || [[ -z "$HF_TOKEN" ]]; then
    MISSING_KEYS=true
fi

if [ "$MISSING_KEYS" = false ]; then
    log_end
else
    echo "" # Break the line for prompts

    if [[ -z "$WANDB_API_KEY" ]]; then
        echo -e "${ORANGE}WANDB_API_KEY is not set.${NC}"
        read -p "Please enter your WANDB_API_KEY: " INPUT_WANDB < /dev/tty
        if [[ -n "$INPUT_WANDB" ]]; then
            echo "WANDB_API_KEY=$INPUT_WANDB" >> .env
            VARS_UPDATED=true
        else
            log_warn "Proceeding without WANDB key. Logging might fail."
        fi
    fi

    if [[ -z "$HF_TOKEN" ]]; then
        echo -e "${ORANGE}HF_TOKEN is not set.${NC}"
        read -p "Please enter your HF_TOKEN: " INPUT_HF < /dev/tty
        if [[ -n "$INPUT_HF" ]]; then
            echo "HF_TOKEN=$INPUT_HF" >> .env
            VARS_UPDATED=true
        else
            log_warn "Proceeding without Hugging Face token. Model download might fail."
        fi
    fi

    if [ "$VARS_UPDATED" = true ]; then
        echo -e "${GREEN}[SUCCESS] Keys saved to .env${NC}"
    fi
fi

# ==========================================
# GIT & DIRECTORY SETUP
# ==========================================

echo ""
echo -e "${CYAN}Let's set up your specific run directory and branch.${NC}"
read -p 'Enter a name for your run (press enter for "default_run"): ' RUN_NAME < /dev/tty

# Sanitize run name (remove spaces)
RUN_NAME=${RUN_NAME// /_}

if [[ -z "$RUN_NAME" ]]; then
    RUN_NAME="default_run"
fi

echo "NAME_OF_YOUR_SUBMISSION=$RUN_NAME" >> .env

log_start "Setting up branch '$RUN_NAME'"
# Check if branch exists, if not create it
if git show-ref --verify --quiet "refs/heads/$RUN_NAME"; then
    git checkout "$RUN_NAME" > /dev/null 2>&1
    log_end
    log_warn "Branch '$RUN_NAME' already exists. Checked it out."
else
    git checkout -b "$RUN_NAME" > /dev/null 2>&1
    log_end
fi

TARGET_DIR="experiments/speedrun/$RUN_NAME"
mkdir -p "$TARGET_DIR"

log_start "Configuring experiment template"

TEMPLATE_PATH="experiments/hackable_transformer_starter_template.py"
TARGET_FILE="$TARGET_DIR/main.py"

if [[ -f "$TEMPLATE_PATH" ]]; then
    cp "$TEMPLATE_PATH" "$TARGET_FILE"

    IMPORT_PATH="experiments.speedrun.${RUN_NAME}.main"

    # Use sed to replace placeholders.
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s|__SUBMISSION_IMPORT_PATH__|$IMPORT_PATH|g" "$TARGET_FILE"
        sed -i '' "s|__SUBMISSION_BRANCH__|$RUN_NAME|g" "$TARGET_FILE"
    else
        sed -i "s|__SUBMISSION_IMPORT_PATH__|$IMPORT_PATH|g" "$TARGET_FILE"
        sed -i "s|__SUBMISSION_BRANCH__|$RUN_NAME|g" "$TARGET_FILE"
    fi

    log_end
else
    echo "" # Break line
    log_error "Could not find template at $TEMPLATE_PATH. Please verify repository integrity."
    exit 1
fi

# ==========================================
# FINAL INSTRUCTIONS
# ==========================================

cd "$TARGET_DIR" || { log_error "Failed to enter directory '$TARGET_DIR'."; exit 1; }
FINAL_PATH="$PWD"
REPO_ROOT=$(git rev-parse --show-toplevel)
ORIGIN_URL=$(git remote get-url origin)

echo ""
echo -e "${GREEN}⚓ All set! Your Marin Speedrun environment is ready.${NC}"
echo ""
echo -e "We have created a starter file at ${CYAN}$TARGET_FILE${NC}."
echo -e "This is your dedicated workspace. Hack here to build your submission!"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "1. Activate the venv and load your WANDB_API_KEY and HF_TOKEN (if your setup does not automatically):"
echo -e "   ${CYAN}cd $REPO_ROOT && . .venv/bin/activate && export \$(grep -v '^#' .env | xargs)${NC}"
echo "2. Find your starter file at ${CYAN}$TARGET_FILE${NC}, fill in your author details, and start hacking."
echo "3. Launch your training run with:"
echo -e "   ${CYAN}python -m experiments.speedrun.${RUN_NAME}.main --force_run_failed true --prefix $REPO_ROOT/local_store${NC}"

# Check if origin points to the main community repo (indicating it's not a personal fork)
if [[ "$ORIGIN_URL" == *"marin-community/marin"* ]]; then
    echo -e "4. ${ORANGE}[Note] You are currently working on a clone of marin-community/marin${NC}"
    echo "   To submit your changes via a PR to Marin, you need to create your fork."
    echo "   Please install the GitHub CLI (https://github.com/cli/cli#installation)"
    echo -e "   and run: ${CYAN}gh repo fork${NC}"
fi

echo ""
echo -e "Tips:"
echo "- The '--prefix' argument specifies the output directory of all artifacts. The env var MARIN_PREFIX can also be set to control it."
echo "- WANDB_ENTITY and WANDB_PROJECT can be set to configure WandB logging."
echo "- Check out https://marin.readthedocs.io/en/latest/tutorials/submitting-speedrun/ for more details."
echo ""
echo -e "${BLUE}⚓ Happy sailing!${NC}"

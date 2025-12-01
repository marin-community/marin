#!/bin/bash

# Colors for pretty output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
ORANGE='\033[0;33m' # Dark Yellow/Brown (visible on white)
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Helper function for printing steps
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
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

if is_marin_root; then
    : # Silent success
else
    # Check if a 'marin' directory exists in the current path
    if [[ -d "marin" ]]; then
        log_warn "Directory 'marin' found. Attempting to enter..."
        cd marin

        if is_marin_root; then
            log_success "Entered directory 'marin'."
        else
            log_error "Directory 'marin' exists but does not appear to be the valid project root (missing pyproject.toml configuration)."
            exit 1
        fi
    else
        log_warn "Current directory is not the root of the 'marin' repository."

        # Check for GitHub CLI
        if command -v gh &> /dev/null && gh auth status &> /dev/null; then
            echo -e "GitHub CLI detected and logged in."
            # Redirect input from /dev/tty to allow reading when piped via curl
            read -p "Would you like to fork and clone marin-community/marin now? (y/n): " confirm_fork < /dev/tty
            if [[ "$confirm_fork" =~ ^[Yy]$ ]]; then
                gh repo fork marin-community/marin --clone --remote

                if [[ -d "marin" ]]; then
                    cd marin
                    log_success "Entered directory 'marin'."
                else
                    log_error "Failed to clone directory. Exiting."
                    exit 1
                fi
            else
                log_error "Please fork and clone the repository manually, then run this script at the root."
                exit 1
            fi
        else
            log_error "We could not find the 'marin' repository root, and GitHub CLI is not configured."
            echo "Please install the GitHub CLI (gh) or manually clone your fork of marin-community/marin."
            exit 1
        fi
    fi
fi

# ==========================================
# CUDA CHECK
# ==========================================

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

echo -e "${GREEN}CUDA environment detected ($VERSION_STRING); press enter to setup Marin${NC}"
# Redirect input from /dev/tty
read -r < /dev/tty

# ==========================================
# ENVIRONMENT SETUP (UV)
# ==========================================

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    log_warn "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env 2>/dev/null || source $HOME/.local/bin/env 2>/dev/null
fi

uv venv --python 3.11
source .venv/bin/activate
uv sync --all-packages --extra=cuda12

log_success "Environment created and dependencies synced."

# ==========================================
# SECRETS (WANDB & HF)
# ==========================================

VARS_UPDATED=false

if [[ -z "$WANDB_API_KEY" ]]; then
    # Silent check first, only prompt if missing
    echo -e "${ORANGE}WANDB_API_KEY is not set.${NC}"
    # Redirect input from /dev/tty
    read -p "Please enter your WANDB_API_KEY: " INPUT_WANDB < /dev/tty
    if [[ -n "$INPUT_WANDB" ]]; then
        # We only write to .env; exporting here won't affect parent shell
        echo "WANDB_API_KEY=$INPUT_WANDB" >> .env
        VARS_UPDATED=true
    else
        log_warn "Proceeding without WANDB key. Logging might fail."
    fi
fi

if [[ -z "$HF_TOKEN" ]]; then
    echo -e "${ORANGE}HF_TOKEN is not set.${NC}"
    # Redirect input from /dev/tty
    read -p "Please enter your HF_TOKEN: " INPUT_HF < /dev/tty
    if [[ -n "$INPUT_HF" ]]; then
        echo "HF_TOKEN=$INPUT_HF" >> .env
        VARS_UPDATED=true
    else
        log_warn "Proceeding without Hugging Face token. Model download might fail."
    fi
fi

if [ "$VARS_UPDATED" = true ]; then
    log_success "Keys saved to .env"
fi

# ==========================================
# GIT & DIRECTORY SETUP
# ==========================================

echo ""
echo -e "${CYAN}Let's set up your specific run directory and branch.${NC}"
# Redirect input from /dev/tty
read -p "Enter a name for your run (e.g., 'fast_llama_v1'): " RUN_NAME < /dev/tty

# Sanitize run name (remove spaces)
RUN_NAME=${RUN_NAME// /_}

if [[ -z "$RUN_NAME" ]]; then
    RUN_NAME="default_run"
fi

# Check if branch exists, if not create it
if git show-ref --verify --quiet "refs/heads/$RUN_NAME"; then
    log_warn "Branch '$RUN_NAME' already exists. Checking it out..."
    git checkout "$RUN_NAME"
else
    git checkout -b "$RUN_NAME"
fi

TARGET_DIR="experiments/speedrun/$RUN_NAME"
mkdir -p "$TARGET_DIR"

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

    log_success "Created experiment at $TARGET_FILE"
else
    log_error "Could not find template at $TEMPLATE_PATH. Please verify repository integrity."
    exit 1
fi

# ==========================================
# FINAL INSTRUCTIONS
# ==========================================

# Move user to the target directory inside the script context to get absolute path
cd "$TARGET_DIR"
FINAL_PATH="$PWD"
REPO_ROOT=$(git rev-parse --show-toplevel)

echo ""
echo -e "${GREEN}⚓ All set! Your Marin Speedrun environment is ready.${NC}"
echo ""
echo -e "We have created a starter file at ${CYAN}$TARGET_FILE${NC}."
echo -e "This is your dedicated workspace. Hack here to build your submission!"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "1. Enter your experiment directory:"
echo -e "   ${CYAN}cd $FINAL_PATH${NC}"
echo "2. Open 'main.py' and fill in your author details."
echo "3. Load your WANDB_API_KEY and HF_TOKEN:"
echo -e "   ${CYAN}export \$(grep -v '^#' $REPO_ROOT/.env | xargs)${NC}"
echo "4. Launch your training run:"
echo -e "   ${CYAN}python -m experiments.speedrun.${RUN_NAME}.main --force_run_failed true --prefix $REPO_ROOT/local_store${NC}"
echo ""
echo -e "Tips:"
echo "- The '--prefix' argument specifies the output directory of all artifacts. The env var MARIN_PREFIX can also be set to control it."
echo "- WANDB_ENTITY and WANDB_PROJECT can be set to configure WandB logging."
echo "- Check out https://marin.readthedocs.io/en/latest/tutorials/submitting-speedrun/ for more details."
echo ""
echo -e "${BLUE}⚓ Happy sailing!${NC}"

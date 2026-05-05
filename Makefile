.PHONY: help clean check fix setup_pre_commit rust-dev rust-user rust-status rust-package
.DEFAULT: help


help:
	@echo "make clean"
	@echo "    Remove all temporary pyc/pycache files"
	@echo "make check"
	@echo "    Run code style and linting (black, ruff) *without* changing files!"
	@echo "make fix"
	@echo "    Run infra/pre-commit.py --fix on your modified files and re-stage them."
	@echo "make lint"
	@echo "    Run infra/pre-commit.py --all-files (no auto-fixing)."
	@echo "make test"
	@echo "    Run all tests"
	@echo "make init"
	@echo "    Init the repo for development"
	@echo "make rust-dev"
	@echo "    Switch to dev mode (build dupekit from source)"
	@echo "make rust-user"
	@echo "    Switch to user mode (install dupekit from pre-built wheel)"
	@echo "make rust-status"
	@echo "    Show current Rust build mode"

init:
	conda install -c conda-forge pandoc
	npm install -g pandiff
	uv sync --extra cpu
	huggingface-cli login

clean:
	find . -name "*.pyc" | xargs rm -f && \
	find . -name "__pycache__" | xargs rm -rf

check:
	./infra/pre-commit.py

fix:
	@FILES=$$(git diff --name-only HEAD); \
	if [ -n "$$FILES" ]; then \
		./infra/pre-commit.py --fix $$FILES && git add $$FILES; \
	else \
		echo "No modified files to fix"; \
	fi

lint:
	./infra/pre-commit.py --all-files

test:
	export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
	export HF_HUB_TOKEN=$HF_TOKEN
	RAY_ADDRESS= PYTHONPATH=tests:. pytest tests --durations=0 -n 4 --tb=no -v

# Target to configure GCP registry cleanup policy for all standard regions
CLUSTER_REPOS = us-central2 us-central1 europe-west4 us-west4 us-east5 us-east1
default_registry_name = marin
configure_gcp_registry_all:
	@echo "Configuring GCP registry cleanup policy for all standard regions..."
	$(foreach region,$(CLUSTER_REPOS), \
		python infra/configure_gcp_registry.py $(default_registry_name) --region=$(region) ; \
	)
	@echo "Cleanup policy configured for all regions."


# stuff for setting up locally
install_uv:
	@if ! command -v uv > /dev/null 2>&1; then \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo "uv installed. Please restart your shell or run: source ~/.cargo/env"; \
	else \
		echo "uv is already installed."; \
	fi

install_gcloud:
	@if ! command -v gcloud > /dev/null 2>&1; then \
		echo "Installing gcloud CLI..."; \
		mkdir -p ~/.local; \
		if [ "$$(uname)" = "Darwin" ]; then \
			if [ "$$(uname -m)" = "arm64" ]; then \
				GCLOUD_ARCHIVE="google-cloud-cli-darwin-arm.tar.gz"; \
			else \
				GCLOUD_ARCHIVE="google-cloud-cli-darwin-x86_64.tar.gz"; \
			fi; \
		else \
			GCLOUD_ARCHIVE="google-cloud-cli-linux-x86_64.tar.gz"; \
		fi; \
		cd ~/.local && \
		curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/$$GCLOUD_ARCHIVE && \
		tar -xzf $$GCLOUD_ARCHIVE && \
		rm $$GCLOUD_ARCHIVE && \
		./google-cloud-sdk/install.sh --quiet --usage-reporting=false --path-update=true --command-completion=true && \
		echo "gcloud installed. Please restart your shell or run: source ~/.zshrc (or ~/.bashrc)"; \
	else \
		echo "gcloud is already installed."; \
	fi

	gcloud config set project hai-gcp-models


setup_pre_commit:
	@HOOK_PATH=.git/hooks/pre-commit; \
	mkdir -p .git/hooks; \
	printf '%s\n' '#!/bin/sh' 'set -e' 'REPO_ROOT="$$(git rev-parse --show-toplevel)"' 'cd "$$REPO_ROOT"' './infra/pre-commit.py --fix' > $$HOOK_PATH; \
	chmod +x $$HOOK_PATH; \
	echo "Installed git pre-commit hook -> $$HOOK_PATH"

install_node:
	@if command -v node > /dev/null 2>&1; then \
		echo "Node.js $$(node --version) is already installed."; \
	elif command -v brew > /dev/null 2>&1; then \
		echo "Installing Node.js via Homebrew..."; \
		brew install node; \
	elif command -v apt-get > /dev/null 2>&1; then \
		echo "Installing Node.js via apt..."; \
		curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - && \
		sudo apt-get install -y nodejs; \
	else \
		echo "Cannot auto-install Node.js. Please install manually: https://nodejs.org/"; \
		exit 1; \
	fi


dev_setup: install_uv install_gcloud install_node setup_pre_commit
	@echo "Dev setup complete."


# Rust crate build mode (dupekit)
# User mode (default): pre-built wheels resolved via find-links in pyproject.toml
# Dev mode: adds dupekit path source to [tool.uv.sources] in pyproject.toml (requires Cargo)

rust-dev:
	@python3 scripts/rust_mode.py dev
	uv sync
	@echo "Done. Run 'make rust-user' before committing."

rust-user:
	@python3 scripts/rust_mode.py user
	uv sync

rust-package:
	@python3 scripts/rust_package.py

rust-status:
	@python3 scripts/rust_mode.py status
	@if command -v cargo > /dev/null 2>&1; then \
		echo "Cargo: installed ($$(cargo --version))"; \
	else \
		echo "Cargo: not found (source builds will fail)"; \
	fi

.PHONY: help clean check autoformat cluster_docker cluster_docker_build cluster_docker_push
.DEFAULT: help

# Help, clean, check and autoformat targets remain unchanged
help:
	@echo "make clean"
	@echo "    Remove all temporary pyc/pycache files"
	@echo "make check"
	@echo "    Run code style and linting (black, ruff) *without* changing files!"
	@echo "make autoformat"
	@echo "    Run code styling (black, ruff) and update in place - committing with pre-commit also does this."
	@echo "make lint"
	@echo "    Run linter including changing files"
	@echo "make test"
	@echo "    Run all tests"
	@echo "make init"
	@echo "    Init the repo for development"

init:
	conda install -c conda-forge pandoc
	npm install -g pandiff
	pre-commit install
	uv sync
	huggingface-cli login

clean:
	find . -name "*.pyc" | xargs rm -f && \
	find . -name "__pycache__" | xargs rm -rf

check:
	ruff check --output-format concise .
	black --check .
	mypy .

autoformat:
	ruff check --fix --show-fixes .
	black .

lint:
	pre-commit run --all-files

test:
	export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
	export HF_HUB_TOKEN=$HF_TOKEN
	RAY_ADDRESS= PYTHONPATH=tests:. pytest tests --durations=0 -n 4 --tb=no -v

# Define regions and tags for the Docker images
CLUSTER_REPOS = us-central2 us-central1 europe-west4 us-west4 asia-northeast1 us-east5 us-east1
TAG_DATE = $(shell date -u +"%Y%m%d")
TAG_VERSIONS = latest $(shell git rev-parse --short HEAD) $(TAG_DATE)

# If VLLM is defined, use different Dockerfile and image name
ifdef VLLM
	DOCKERFILE = docker/marin/Dockerfile.vllm
	DOCKER_IMAGE_NAME = marin_vllm
else
	DOCKERFILE = docker/marin/Dockerfile.cluster
	DOCKER_IMAGE_NAME = marin_cluster
endif

# Target to build the Docker image and tag it appropriately
cluster_docker_build:
	@echo "Building Docker image using Dockerfile: $(DOCKERFILE)"
	docker buildx build --platform linux/amd64 --output "type=docker,compression=zstd" -t '$(DOCKER_IMAGE_NAME):latest' -f $(DOCKERFILE) .
	@echo "Tagging Docker image for each region and version..."
	$(foreach region,$(CLUSTER_REPOS), \
		$(foreach version,$(TAG_VERSIONS), \
			docker tag '$(DOCKER_IMAGE_NAME):latest' '$(region)-docker.pkg.dev/hai-gcp-models/marin/$(DOCKER_IMAGE_NAME):$(version)';))
	@echo "Docker image build and tagging complete, updating config.py with latest version..."

cluster_tag:
	@if [ "$$(uname)" = "Darwin" ]; then \
		sed -i '' -e "s/LATEST = \".*\"/LATEST = \"$(TAG_DATE)\"/" lib/marin/src/marin/cluster/config.py; \
	else \
		sed -i -e "s/LATEST = \".*\"/LATEST = \"$(TAG_DATE)\"/" lib/marin/src/marin/cluster/config.py; \
	fi

# Target to push the tagged Docker images to their respective Artifact Registries
cluster_docker_push: cluster_tag
	@echo "Authenticating and preparing repositories..."
	$(foreach region,$(CLUSTER_REPOS), \
		gcloud auth configure-docker $(region)-docker.pkg.dev;)
	$(foreach region,$(CLUSTER_REPOS), \
		gcloud artifacts repositories list --location=$(region) --filter 'name:marin' > /dev/null || \
		gcloud artifacts repositories create --repository-format=docker --location=$(region) marin;)
	@echo "Pushing Docker images for each region and version..."
	$(foreach region,$(CLUSTER_REPOS), \
		$(foreach version,$(TAG_VERSIONS), \
			docker push '$(region)-docker.pkg.dev/hai-gcp-models/marin/$(DOCKER_IMAGE_NAME):$(version)';))

cluster_docker_ghcr_push: cluster_docker_build
	@echo "Pushing Docker image to GitHub Container Registry..."
	$(foreach version,$(TAG_VERSIONS), \
		docker tag '$(DOCKER_IMAGE_NAME):latest' 'ghcr.io/stanford-crfm/marin/$(DOCKER_IMAGE_NAME):$(version)';)

	$(foreach version,$(TAG_VERSIONS), \
		docker push 'ghcr.io/stanford-crfm/marin/$(DOCKER_IMAGE_NAME):$(version)';)


# Meta-target that builds and then pushes the Docker images
cluster_docker: cluster_docker_build cluster_docker_push
	@echo "Docker image build and push complete."


# Target to configure GCP registry cleanup policy for all standard regions
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


# get secret ssh key from gcp secrets
get_secret_key: install_gcloud
	gcloud secrets versions access latest --secret=RAY_CLUSTER_PRIVATE_KEY > ~/.ssh/marin_ray_cluster.pem && \
	chmod 600 ~/.ssh/marin_ray_cluster.pem
	gcloud secrets versions access latest --secret=RAY_CLUSTER_PUBLIC_KEY > ~/.ssh/marin_ray_cluster.pub


dev_setup: install_uv install_gcloud get_secret_key
	echo "Dev setup complete."

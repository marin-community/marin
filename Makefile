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

# Define regions and tags for the Docker images
CLUSTER_REPOS = us-central2 europe-west4 us-west4 asia-northeast1 us-east5 us-east1
TAG_VERSIONS = latest $(shell git rev-parse --short HEAD) $(shell date -u +"%Y%m%d")

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
	docker buildx build --platform linux/amd64 -t '$(DOCKER_IMAGE_NAME):latest' -f $(DOCKERFILE) .
	@echo "Tagging Docker image for each region and version..."
	$(foreach region,$(CLUSTER_REPOS), \
		$(foreach version,$(TAG_VERSIONS), \
			docker tag '$(DOCKER_IMAGE_NAME):latest' '$(region)-docker.pkg.dev/hai-gcp-models/marin/$(DOCKER_IMAGE_NAME):$(version)';))

# Target to push the tagged Docker images to their respective Artifact Registries
cluster_docker_push:
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
	@echo "##################################################################"
	@echo "Don't forget to update the tags in infra/update-cluster-configs.py"
	@echo "##################################################################"

cluster_docker_ghcr_push: cluster_docker_build
	@echo "Pushing Docker image to GitHub Container Registry..."
	$(foreach version,$(TAG_VERSIONS), \
		docker tag '$(DOCKER_IMAGE_NAME):latest' 'ghcr.io/stanford-crfm/marin/$(DOCKER_IMAGE_NAME):$(version)';)

	$(foreach version,$(TAG_VERSIONS), \
		docker push 'ghcr.io/stanford-crfm/marin/$(DOCKER_IMAGE_NAME):$(version)';)


# Meta-target that builds and then pushes the Docker images
cluster_docker: cluster_docker_build cluster_docker_push
	@echo "Docker image build and push complete."

.PHONY: help clean check autoformat
.DEFAULT: help

# Generates a useful overview/help message for various make features - add to this as necessary!
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

CLUSTER_REPOS = us-central2 europe-west4 us-west4 asia-northeast1 us-east5 us-east1
TAG_VERSIONS = latest $(shell git rev-parse --short HEAD) $(shell date -u +"%Y%m%d")

DOCKERFILE = docker/marin/Dockerfile.cluster
DOCKER_IMAGE_NAME = marin_cluster

cluster_docker:
ifdef VLLM
	$(eval DOCKERFILE = docker/marin/Dockerfile.vllm)
	$(eval DOCKER_IMAGE_NAME = marin_vllm)
endif
	@echo "Building and pushing Docker images for clusters $(CLUSTER_REPOS) with tags $(TAG_VERSIONS)"
	@echo "Using Dockerfile: $(DOCKERFILE)"
	# Authenticate for each region
	$(foreach region,$(CLUSTER_REPOS), \
		gcloud auth configure-docker $(region)-docker.pkg.dev;)

	# Create Artifact Repositories if they don't exist
	$(foreach region,$(CLUSTER_REPOS), \
		gcloud artifacts repositories list --location=$(region) --filter 'name:marin' > /dev/null || \
		gcloud artifacts repositories create --repository-format=docker --location=$(region) marin;)

	# Build Docker image
	docker buildx build --platform linux/amd64 -t '$(DOCKER_IMAGE_NAME):latest' -f $(DOCKERFILE) .

	# Tag the Docker image for each region and version
	$(foreach region,$(CLUSTER_REPOS), \
		$(foreach version,$(TAG_VERSIONS), \
			docker tag '$(DOCKER_IMAGE_NAME):latest' '$(region)-docker.pkg.dev/hai-gcp-models/marin/$(DOCKER_IMAGE_NAME):$(version)';))

	# Push the Docker images for each region and version
	$(foreach region,$(CLUSTER_REPOS), \
		$(foreach version,$(TAG_VERSIONS), \
			docker push '$(region)-docker.pkg.dev/hai-gcp-models/marin/$(DOCKER_IMAGE_NAME):$(version)';))

	@echo "##################################################################"
	@echo "Don't forget to update the tags in infra/update-cluster-configs.py"
	@echo "##################################################################"

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


cluster_docker:
	gcloud artifacts repositories list  --location=us-central2  --filter 'name:marin' > /dev/null || gcloud artifacts repositories create --repository-format=docker --location=us-central2 marin
	docker buildx build --platform linux/amd64 -t 'marin_cluster:latest' -f docker/marin/Dockerfile.cluster .
	docker tag 'marin_cluster:latest' 'us-central2-docker.pkg.dev/hai-gcp-models/marin/marin_cluster:latest'
	# also tag with the commit hash and the date
	docker tag 'marin_cluster:latest' 'us-central2-docker.pkg.dev/hai-gcp-models/marin/marin_cluster:$(shell git rev-parse --short HEAD)'
	docker tag 'marin_cluster:latest' 'us-central2-docker.pkg.dev/hai-gcp-models/marin/marin_cluster:$(shell date -u +"%Y%m%d")'
	docker push 'us-central2-docker.pkg.dev/hai-gcp-models/marin/marin_cluster:latest'
	docker push 'us-central2-docker.pkg.dev/hai-gcp-models/marin/marin_cluster:$(shell git rev-parse --short HEAD)'
	docker push 'us-central2-docker.pkg.dev/hai-gcp-models/marin/marin_cluster:$(shell date -u +"%Y%m%d")'

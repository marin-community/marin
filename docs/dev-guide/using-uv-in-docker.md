# Using uv in Docker

This guide describes best practices for installing and using [uv](https://github.com/astral-sh/uv) inside Docker images.

## Getting Started

uv publishes several Docker images. Distroless images expose only the `uv` binaries for copying into your own image, while derived images come with `uv` preinstalled on popular bases such as Debian and Alpine. To quickly run uv from a prebuilt container:

```bash
docker run --rm -it ghcr.io/astral-sh/uv:debian uv --help
```

A list of available images can be found on the [GitHub Container Registry](https://github.com/astral-sh/uv/pkgs/container/uv). Each tag is also available with explicit version numbers, e.g. `ghcr.io/astral-sh/uv:0.7.19`.

## Installing uv

You can either use one of the prebuilt images or copy the binaries from the distroless image:

```Dockerfile
FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.7.19 /uv /uvx /bin/
```

Alternatively use the installer script:

```Dockerfile
FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y curl ca-certificates
ADD https://astral.sh/uv/0.7.19/install.sh /uv-install.sh
RUN sh /uv-install.sh && rm /uv-install.sh
ENV PATH="/root/.local/bin:$PATH"
```

Pinning a specific version ensures reproducible builds.

## Installing a Project

Copy your project into the image and sync it using uv. Avoid including `.venv` in the image by adding it to `.dockerignore`.

```Dockerfile
# Copy the project
ADD . /app
WORKDIR /app
RUN uv sync --locked
```

To run your application:

```Dockerfile
CMD ["uv", "run", "my_app"]
```

You can also activate the virtual environment manually:

```Dockerfile
ENV PATH="/app/.venv/bin:$PATH"
```

For faster rebuilds, consider separating dependency installation into intermediate layers and enabling a cache mount:

```Dockerfile
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project
```

This installs dependencies separately from your project files, improving build times.

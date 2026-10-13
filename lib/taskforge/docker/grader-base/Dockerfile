# Taskforge grader base: the verifier machine of a script grader on a task without its own image.
# CPython 3.12 with the standard library, sh, bash, setsid and tar, running as root in /app.
# The multi-platform index of docker.io/library/python:3.12-slim (Debian trixie); build for linux/amd64.
FROM docker.io/library/python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1
USER root
WORKDIR /app

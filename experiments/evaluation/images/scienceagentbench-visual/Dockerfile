FROM python@sha256:0f086103b026048cc50cb5da25ec435c023c5241a83fc44daea593377abc61c4
COPY requirements.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir --require-hashes --only-binary=:all: -r /tmp/requirements.lock && rm /tmp/requirements.lock
WORKDIR /testbed
RUN mkdir -p benchmark/eval_programs && ln -s /tests/gold_results benchmark/eval_programs/gold_results

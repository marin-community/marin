# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the six v0.1 campaign policies with an explicit architecture-gate provenance."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from marin.evaluation.evalchemy.config import load_evalchemy_config
from marin.evaluation.evalchemy.runner import _run_config_json
from marin.inference.types import OpenAIEndpoint, RunningModel

from experiments.evaluation.evals import evalchemy_run_config
from experiments.weight_merging.rsi.levanter_267.evaluate import STEP12_REVISION, submit_eval

REPO = "open-athena/Marin-Snowball-67B-A2B-v0.1"
REVISION = "adc0078a19a4383bc998fcdb43867b3360baf5f3"
EVALCHEMY = "c1d5c62cdc3939b42f64411af44fd50716b53454"
POLICIES = ("math500", "gsm8k-0shot", "boolq", "piqa", "humanevalplus", "aime24")
CAMPAIGN = Path("experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/evalchemy-configs")


def policy_plan(key, policy_dir, endpoint, output, concurrency, max_length=32704):
    path = policy_dir / f"{key}.yaml"
    source = load_evalchemy_config(path)
    config = replace(
        evalchemy_run_config(key, source),
        apply_chat_template=True,
        num_concurrent=concurrency,
        max_length=max_length,
    )
    if any(not task.generation for task in config.tasks) and (
        config.max_gen_toks is not None or config.extra_gen_kwargs
    ):
        raise ValueError("Loglikelihood policies must not carry generation overrides")
    model = RunningModel(OpenAIEndpoint("http://resolved-inside-iris/v1", "snowball-levanter"), tokenizer=REPO)
    payload = json.loads(_run_config_json(model, config, output))
    payload["policy_provenance"] = dict(
        path=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        yaml=path.read_text(),
        reference_server_context=73728,
        reference_effective_context=73664,
        evaluated_context_cap=max_length,
        reference_concurrency=16,
        evaluated_concurrency=concurrency,
        serving_max_seqs="recorded from live process provenance",
        tokenizer_revision=None,
        numerical_validation="Step12 architecture/code gate; no independent v0.1 item303 oracle exists",
    )
    extras = tuple(sorted({"serve-eval", *source.runtime_extras}))
    return dict(endpoint=endpoint, config=payload, evalchemy_commit=EVALCHEMY, extras=extras, model_revision=REVISION)


def validate_gate(gate, endpoint):
    if not gate or not gate.get("accepted") or gate.get("endpoint") != endpoint:
        raise ValueError("An accepted report for this exact v0.1 endpoint is required")
    if gate.get("model_revision") != REVISION:
        raise ValueError("The report must identify the pinned v0.1 revision")
    architecture = gate.get("architecture_gate", {})
    digest = gate.get("inference_source_sha256")
    if (
        not architecture.get("accepted")
        or not architecture.get("oracle_comparison")
        or architecture.get("model_revision") != STEP12_REVISION
        or not digest
        or architecture.get("inference_source_sha256") != digest
    ):
        raise ValueError("Require the accepted Step12 numerical gate and identical Levanter source digest")
    if not gate.get("production_decode_token_ids") or not gate.get("first_token_matches_cached_score"):
        raise ValueError("Require positive v0.1 production generation and cached first-token agreement")
    if not all(gate.get(key) for key in ("run_id", "source_commit", "serve_job", "model_path", "model_provenance")):
        raise ValueError("Require process identity and verified v0.1 weight/config/tokenizer provenance")
    provenance = gate["model_provenance"]
    if provenance.get("weights_verified") is not True or not all(
        provenance.get(key)
        for key in (
            "weight_manifest",
            "config_sha256",
            "tokenizer_sha256",
            "template_sha256",
            "generation_config_sha256",
        )
    ):
        raise ValueError("Require the matching weight manifest and explicit v0.1 config/tokenizer/template hashes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", choices=POLICIES)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--gate-report", type=Path)
    parser.add_argument("--client-file", type=Path, required=True)
    parser.add_argument("--policy-dir", type=Path, default=CAMPAIGN)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=32704)
    parser.add_argument("--runtime-minutes", type=int, default=200)
    args = parser.parse_args()
    if not 2 <= args.runtime_minutes <= 240 or args.concurrency < 1 or args.max_length < 1:
        parser.error("Require a 2-240-minute bound and positive concurrency")
    if not args.output.startswith("gs://marin-us-central2/"):
        parser.error("Place archives alongside the central2 serving slice")
    gate = json.loads(args.gate_report.read_text()) if args.gate_report else None
    if args.submit:
        try:
            validate_gate(gate, args.endpoint)
        except ValueError as error:
            parser.error(str(error))
    plan = policy_plan(args.policy, args.policy_dir, args.endpoint, args.output, args.concurrency, args.max_length)
    source = args.client_file.read_text()
    plan.update(client_sha256=hashlib.sha256(source.encode()).hexdigest(), runtime_minutes=args.runtime_minutes)
    print(json.dumps(plan, indent=2), flush=True)
    if args.submit:
        submit_eval(
            args.name or f"levanter267-v01-{args.policy}",
            source,
            plan["config"],
            args.endpoint,
            args.output,
            EVALCHEMY,
            args.runtime_minutes,
            gate,
            REVISION,
            plan["extras"],
        )


if __name__ == "__main__":
    main()

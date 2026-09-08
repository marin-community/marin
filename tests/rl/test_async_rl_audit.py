# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import copy
import io
import json
import os
import pathlib
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import pytest
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractBufferedFile

from experiments.post_training import async_rl_audit as audit
from experiments.post_training.math_eval.paired import SeedInference, bootstrap


class BufferedLocalFileSystem(LocalFileSystem):
    """Exercise the real remote-style buffered file over local fixture storage."""

    def open(self, path, mode="rb", **kwargs):
        if mode == "rb":
            return AbstractBufferedFile(LocalFileSystem(), path, mode=mode)
        return super().open(path, mode=mode, **kwargs)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="async-rl-audit-test-")
        self.root = pathlib.Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.request = {
            "run_id": "fixture/run",
            "attempt_id": "fixture-attempt",
            "completion_mode": "metrics",
            "seed": 17,
            "runtime": {"commit": "a" * 40, "profile": "megatron"},
            "config_yaml": "entrypoint: fully_async\n",
            "model": {},
            "train_data": [],
            "validation_data": [],
            "topology": {"role_plan": {"train_batch_size": 2, "n_samples_per_prompt": 1, "num_inference_engines": 2}},
            "output": {
                key: str(self.root / value)
                for key, value in {
                    "terminal_manifest_uri": "terminal.json",
                    "resolved_config_uri": "resolved.json",
                    "attempts_root": "attempts",
                    "checkpoint_root": "checkpoints",
                    "export_root": "exports",
                }.items()
            },
        }
        self.receipt_uri = str(self.root / "receipts/fixture-attempt.json")
        self.digest = audit.canonical_sha(
            {"schema_version": 2, "request": self.request, "receipt_uri": self.receipt_uri}
        )
        self.receipt = {
            "schema_version": 1,
            "run_id": "fixture/run",
            "attempt_id": "fixture-attempt",
            "request_fingerprint": self.digest,
            "completion_mode": "metrics",
            "global_step": 2,
        }
        self.write(self.receipt_uri, self.receipt)
        self.envelope = {
            "schema_version": 2,
            "request": self.request,
            "execution": {},
            "response": {
                "state": "succeeded",
                "iris_job_state": "succeeded",
                "failure": None,
                "run_id": "fixture/run",
                "attempt_id": "fixture-attempt",
                "runtime": self.request["runtime"],
                "iris_job_id": "/fixture/job",
                "training": {
                    "global_step": 2,
                    "checkpoint": None,
                    "receipt_uri": self.receipt_uri,
                    "resolved_config_uri": self.request["output"]["resolved_config_uri"],
                },
            },
        }
        self.write(self.root / "terminal.json", self.envelope)
        self.write(self.root / "attempts/fixture-attempt.json", self.envelope)
        completion = {
            "mode": "metrics",
            "run_id": "fixture/run",
            "attempt_id": "fixture-attempt",
            "request_fingerprint": self.digest,
            "receipt_uri": self.receipt_uri,
        }
        self.write(
            self.root / "resolved.json",
            {
                "entrypoint": "skyrl_train.entrypoints.fully_async",
                "hydra_args": (
                    [f"++trainer.completion.{key}={value}" for key, value in completion.items()]
                    + ["++trainer.ckpt_interval=-1", "++trainer.hf_save_interval=-1"]
                ),
            },
        )
        self.metrics = {
            "eval/all/avg_score": 0.5,
            "eval/all/pass_at_1": 0.5,
            "eval/gsm8k/avg_score": 0.5,
            "eval/gsm8k/pass_at_1": 0.5,
        }
        self.dump_paths = []
        for step in (0, 2):
            dump = self.root / f"exports/dumped_evals/global_step_{step}_evals"
            self.write(dump / "aggregated_results.jsonl", self.metrics)
            rows = [
                {
                    "uid": str(i),
                    "row_ordinal": i,
                    "token_provenance": "finalized_trajectory",
                    "generator_engine_index": i,
                    "prompt_token_ids": [i, 3],
                    "response_ids": [4, i],
                    "prompt_token_ids_sha256": audit.canonical_sha([i, 3]),
                    "response_ids_sha256": audit.canonical_sha([4, i]),
                    "response_length": 2,
                    "score": [0, i],
                    "stop_reason": "stop",
                    "data_source": "gsm8k",
                }
                for i in range(2)
            ]
            file = dump / "gsm8k.jsonl"
            file.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.dump_paths.append(file)
        self.history = [{"global_step": 0, **self.metrics}]
        for step in (1, 2):
            row = {
                "global_step": step,
                **{key: 0.1 for key in audit.REQUIRED},
                "policy/behavior_drift/finite_fraction": 1,
                "policy/behavior_drift/missing_behavior": 0,
                "consumed/sequences": 2,
                "async/staleness_max": 1,
                "async/performance/core_seconds": 2,
            }
            if step == 2:
                row.update(self.metrics)
            self.history.append(row)
        config = {
            "trainer": {
                "completion": completion,
                "seed": 17,
                "max_steps": 2,
                "ckpt_interval": -1,
                "hf_save_interval": -1,
                "algorithm": {"use_kl_in_reward": False, "use_tis": False},
                "fully_async": {"max_staleness_steps": 1},
            },
            "generator": {"num_inference_engines": 2},
        }
        self.run = types.SimpleNamespace(
            url="https://wandb.ai/actual-entity/project/runs/id",
            name="fixture",
            state="finished",
            config=config,
            scan_history=lambda **kwargs: iter(self.history),
        )

        def get_run(path):
            self.assertEqual(path, "actual-entity/project/id")
            return self.run

        self.api = types.SimpleNamespace(run=get_run)
        self.input = {
            "run_id": "fixture/run",
            "attempt_id": "fixture-attempt",
            "expected_steps": 2,
            "receipt_uri": self.receipt_uri,
            "envelope_uri": str(self.root / "terminal.json"),
            "wandb_url": self.run.url,
            "expected_eval_steps": [0, 2],
            "expected_eval_rows": 2,
        }

    def write(self, path, value):
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_complete_receipt_history_and_dump(self):
        result = audit.audit_run(self.input, self.api)
        self.assertEqual(result["wandb_entity"], "actual-entity")
        self.assertEqual(result["history"]["sums"]["consumed/sequences"], 4)
        self.assertTrue(result["storage"]["receipt_verified"])
        self.assertTrue(all(item["hashes_verified"] == 4 for item in result["eval_dumps"]))
        self.assertNotIn("prompt_token_ids", json.dumps(result))

    def test_paired_snapshot_reuses_verified_dump_after_storage_removed(self):
        self.input["label"] = "reference"
        snapshots = {}
        result = audit.audit_run(self.input, self.api, snapshots, retain_evaluations=True)
        shutil.rmtree(self.root / "exports")
        # These are the audited token hashes and rewards, not a second read or
        # a caller-supplied replacement for a missing durable dump.
        identity, scores = audit.question_scores(snapshots["reference"]["evaluations"][2])
        self.assertEqual(scores, {"0": 0, "1": 1})
        self.assertEqual(identity, [[str(i), audit.canonical_sha([i, 3])] for i in range(2)])
        self.assertNotIn("evaluations", result)

    def test_preview_attempt_is_not_used_for_receipt_sha(self):
        preview = copy.deepcopy(self.envelope)
        preview["request"]["attempt_id"] = "different-preview-attempt"
        self.write(self.root / "preview.json", preview)
        self.input["envelope_uri"] = str(self.root / "preview.json")
        result = audit.audit_run(self.input, self.api)
        self.assertEqual(result["storage"]["actual_attempt_id"], "fixture-attempt")
        self.assertEqual(result["storage"]["request_fingerprint"], self.digest)
        preview["request"]["seed"] = 99
        self.write(self.root / "preview.json", preview)
        with self.assertRaisesRegex(ValueError, "beyond regenerated attempt_id"):
            audit.audit_run(self.input, self.api)

    def test_fsspec_buffered_file_readline_contract(self):
        filesystem = BufferedLocalFileSystem()
        with patch.object(audit, "fs_path", lambda uri: (filesystem, uri)):
            result = audit.audit_run(self.input, self.api)
        self.assertTrue(result["eval_dumps"][0]["aggregate_verified"])
        self.assertEqual(result["eval_dumps"][0]["rows"], 2)

    def failed_launcher(self):
        self.envelope["response"].update(
            state="failed", iris_job_state="worker_failed", failure="Iris job reached worker_failed"
        )
        self.write(self.root / "attempts/fixture-attempt.json", self.envelope)
        self.write(self.root / "preview.json", self.envelope)
        self.input["envelope_uri"] = str(self.root / "preview.json")
        (self.root / "terminal.json").unlink()

    def test_failed_launcher_valid_receipt_passes_training_evidence_only(self):
        self.failed_launcher()
        self.input["require_launcher_success"] = False
        result = audit.audit_run(self.input, self.api)
        self.assertTrue(result["training_evidence_pass"])
        self.assertFalse(result["clean_end_to_end"])
        self.assertFalse(result["storage"]["terminal_manifest_present"])
        self.assertEqual(result["storage"]["launcher"]["state"], "failed")
        self.assertEqual(result["storage"]["launcher"]["failure"], "Iris job reached worker_failed")

    def test_training_evidence_cli_does_not_emit_clean_pass(self):
        self.failed_launcher()
        self.input.update(require_launcher_success=False, label="fixture")
        stdout = io.StringIO()
        with (
            patch.dict(os.environ, {"ASYNC_RL_AUDIT_SPEC": json.dumps({"runs": [self.input]})}),
            patch.object(audit.wandb, "Api", lambda **kwargs: self.api),
            patch.object(sys, "argv", ["audit"]),
            contextlib.redirect_stdout(stdout),
        ):
            with self.assertRaises(SystemExit) as stopped:
                audit.main()
        self.assertEqual(stopped.exception.code, 0)
        self.assertIn("ASYNC_RL_TRAINING_EVIDENCE_PASS", stdout.getvalue())
        self.assertIn("ASYNC_RL_TERMINAL_AUDIT_NOT_CLEAN", stdout.getvalue())
        self.assertNotIn("ASYNC_RL_TERMINAL_AUDIT_PASS", stdout.getvalue())
        result = json.loads(stdout.getvalue().splitlines()[0].split(" ", 1)[1])
        self.assertTrue(result["training_evidence_pass"])
        self.assertFalse(result["clean_end_to_end"])

    def test_failed_launcher_rejected_by_default(self):
        self.failed_launcher()
        with self.assertRaisesRegex(ValueError, "Launcher did not succeed"):
            audit.audit_run(self.input, self.api)

    def test_successful_launcher_requires_recorded_iris_success(self):
        for state in ("failed", "worker_failed", "running", None):
            for require_success in (True, False):
                with self.subTest(iris_job_state=state, require_launcher_success=require_success):
                    self.envelope["response"]["iris_job_state"] = state
                    self.write(self.root / "terminal.json", self.envelope)
                    self.write(self.root / "attempts/fixture-attempt.json", self.envelope)
                    self.input["require_launcher_success"] = require_success
                    with self.assertRaisesRegex(ValueError, "Successful launcher has no recorded Iris success"):
                        audit.audit_run(self.input, self.api)

    def test_failed_launcher_bad_receipt_still_fails_training_evidence(self):
        self.failed_launcher()
        self.input["require_launcher_success"] = False
        self.write(self.receipt_uri, self.receipt | {"request_fingerprint": "b" * 64})
        with self.assertRaisesRegex(ValueError, "Receipt step/identity/SHA"):
            audit.audit_run(self.input, self.api)

    def test_failed_launcher_with_terminal_manifest_is_inconsistent(self):
        self.failed_launcher()
        self.input["require_launcher_success"] = False
        self.write(self.root / "terminal.json", self.envelope)
        with self.assertRaisesRegex(ValueError, "Failed attempt has a terminal manifest"):
            audit.audit_run(self.input, self.api)

    def startup_repeat_fixture(self):
        root = self.dump_paths[0].parent
        for index in range(3):
            target = root / f"startup_pass_{index}"
            target.mkdir()
            for name in ("gsm8k.jsonl", "aggregated_results.jsonl"):
                shutil.copyfile(root / name, target / name)
        for name in ("gsm8k.jsonl", "aggregated_results.jsonl"):
            (root / name).unlink()
        self.history[0] = {
            "global_step": 0,
            **{
                f"eval/startup_pass_{index}/" + key.removeprefix("eval/"): value
                for index in range(3)
                for key, value in self.metrics.items()
            },
        }
        self.run.config["trainer"]["initial_eval_repeat_count"] = 3
        self.input["initial_eval_repeat_count"] = 3
        return root

    def test_startup_repeat_three_namespaces_and_metrics(self):
        self.startup_repeat_fixture()
        result = audit.audit_run(self.input, self.api)
        self.assertEqual(
            [row["dump_namespace"] for row in result["eval_dumps"]],
            ["startup_pass_0", "startup_pass_1", "startup_pass_2", None],
        )
        self.assertTrue(result["startup_repeatability"]["response_hashes_identical"])
        self.assertEqual(len(result["startup_repeatability"]["pairwise"]), 3)

    def test_startup_repeat_missing_metric_pass_fails(self):
        self.startup_repeat_fixture()
        self.history[0] = {
            key: value for key, value in self.history[0].items() if not key.startswith("eval/startup_pass_1/")
        }
        with self.assertRaisesRegex(ValueError, "Startup evaluation namespace coverage"):
            audit.audit_run(self.input, self.api)

    def test_startup_repeat_reports_changed_response_without_rejecting_noise(self):
        root = self.startup_repeat_fixture()
        path = root / "startup_pass_1/gsm8k.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["response_ids"] = [99, 98]
        rows[0]["response_ids_sha256"] = audit.canonical_sha(rows[0]["response_ids"])
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        result = audit.audit_run(self.input, self.api)
        self.assertFalse(result["startup_repeatability"]["response_hashes_identical"])
        self.assertEqual(result["startup_repeatability"]["pairwise"][0]["response_changed_rows"], 1)
        self.assertEqual(result["startup_repeatability"]["pairwise"][0]["response_changed_equal_engine_index_rows"], 1)
        self.assertTrue(result["training_evidence_pass"])

    def test_startup_repeat_changed_prompt_invalidates_frozen_control(self):
        root = self.startup_repeat_fixture()
        path = root / "startup_pass_1/gsm8k.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["prompt_token_ids"] = [99, 98]
        rows[0]["prompt_token_ids_sha256"] = audit.canonical_sha(rows[0]["prompt_token_ids"])
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "Startup evaluation prompt/UID order changed"):
            audit.audit_run(self.input, self.api)

    def test_full_module_async_entrypoint_requires_async_metrics(self):
        for row in self.history[1:]:
            row.pop("async/performance/core_seconds")
        with self.assertRaisesRegex(ValueError, "Missing metrics"):
            audit.summarize_history(self.run, 2, "skyrl_train.entrypoints.fully_async", [])

    def test_background_evaluations_keep_requested_steps_and_training_step(self):
        self.history.append({"global_step": 2, "eval/requested_at_step": 1, "eval/all/avg_score": 0.25})
        history, evaluations = audit.summarize_history(self.run, 2, "fully_async", [])
        self.assertEqual(sorted(evaluations), [0, 1, 2])
        self.assertEqual(evaluations[1]["eval/all/avg_score"], 0.25)
        self.assertEqual(history["steps"], [1, 2])
        self.history.append({"global_step": 2, "eval/requested_at_step": 1, "eval/all/avg_score": 0.5})
        with self.assertRaisesRegex(ValueError, "Duplicate evaluation"):
            audit.summarize_history(self.run, 2, "fully_async", [])

    def test_background_requested_evaluation_step_must_be_valid(self):
        for invalid in (-1, 1.5, 3):
            with self.subTest(invalid=invalid):
                self.history.append({"global_step": 2, "eval/requested_at_step": invalid, "eval/all/avg_score": 0.25})
                with self.assertRaisesRegex(ValueError, "Invalid requested evaluation step"):
                    audit.summarize_history(self.run, 2, "fully_async", [])
                self.history.pop()

    def test_entrypoint_disagreement_is_rejected(self):
        path = self.root / "resolved.json"
        resolved = json.loads(path.read_text())
        resolved["entrypoint"] = "skyrl_train.entrypoints.main_base"
        self.write(path, resolved)
        with self.assertRaisesRegex(ValueError, "Request/resolved entrypoint differs"):
            audit.audit_run(self.input, self.api)

    def test_frozen_gate_requires_actual_engine_indices(self):
        self.input["require_engine_indices"] = True
        result = audit.audit_run(self.input, self.api)
        self.assertEqual(result["eval_dumps"][0]["engine_identity_known_rows"], 2)
        path = self.dump_paths[0]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0].pop("generator_engine_index")
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "Missing actual generator engine index"):
            audit.audit_run(self.input, self.api)

    def test_out_of_range_engine_index_fails(self):
        path = self.dump_paths[0]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["generator_engine_index"] = 2
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "Invalid generator engine index"):
            audit.audit_run(self.input, self.api)

    def test_expanded_ep8_frontends_use_resolved_engine_bound(self):
        # Physical replica templates and logical dispatcher frontends differ.
        for engine_count in (8, 16):
            with self.subTest(engine_count=engine_count):
                self.run.config["generator"].update(
                    num_inference_engines=engine_count // 8,
                    inference_engine_data_parallel_size=8,
                    backend="vllm",
                    run_engines_locally=True,
                )
                path = self.dump_paths[0]
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                rows[0]["generator_engine_index"] = engine_count - 1
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                result = audit.audit_run(self.input, self.api)
                self.assertEqual(result["eval_dumps"][0]["generator_engine_index_counts"][str(engine_count - 1)], 1)
                rows[0]["generator_engine_index"] = engine_count
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                with self.assertRaisesRegex(ValueError, "Invalid generator engine index"):
                    audit.audit_run(self.input, self.api)

    def test_resolved_engine_bound_must_be_positive_integer(self):
        for count in (None, 0, -1, 1.5, True):
            with self.subTest(count=count):
                self.run.config["generator"]["num_inference_engines"] = count
                with self.assertRaisesRegex(ValueError, "Invalid resolved inference replica count"):
                    audit.audit_run(self.input, self.api)
        self.run.config["generator"]["num_inference_engines"] = 2
        for count in (0, -1, 1.5, True):
            with self.subTest(dp=count):
                self.run.config["generator"]["inference_engine_data_parallel_size"] = count
                with self.assertRaisesRegex(ValueError, "Invalid resolved inference data parallel size"):
                    audit.audit_run(self.input, self.api)

    def test_between_run_disagreements_preserve_engine_index_scope(self):
        rows = [["uid0", "prompt0", "response0", 1, "stop", 0], ["uid1", "prompt1", "response1", 0, "stop", 1]]
        left = {
            "request": self.request,
            "source_config": {"trainer": {"weight_change_probe": False}},
            "startup": [copy.deepcopy(rows) for _ in range(3)],
        }
        right = copy.deepcopy(left)
        right["source_config"]["trainer"]["weight_change_probe"] = True
        right["startup"][1][0][2] = "different_response"
        right["startup"][1][0][5] = 1
        result = audit.compare_startup_runs("off", "on", {"off": left, "on": right})
        self.assertEqual(len(result["pairwise"]), 9)
        self.assertEqual(sum(row["response_changed_rows"] for row in result["pairwise"]), 3)
        self.assertEqual(result["pairwise"][1]["response_changed_different_engine_index_rows"], 1)
        self.assertIn("do not identify the same actor/GPU", result["engine_comparison_scope"])
        right["source_config"]["trainer"]["lr"] = 0.1
        with self.assertRaisesRegex(ValueError, "differs beyond weight_change_probe"):
            audit.compare_startup_runs("off", "on", {"off": left, "on": right})

    def test_receipt_sha_tamper_fails(self):
        self.write(self.receipt_uri, self.receipt | {"request_fingerprint": "b" * 64})
        with self.assertRaisesRegex(ValueError, "Receipt step/identity/SHA"):
            audit.audit_run(self.input, self.api)

    def test_hash_tamper_fails(self):
        path = self.dump_paths[0]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["response_ids"][0] = 999
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "token SHA mismatch"):
            audit.audit_run(self.input, self.api)

    def test_aggregate_mismatch_fails(self):
        self.write(self.dump_paths[0].parent / "aggregated_results.jsonl", self.metrics | {"eval/all/avg_score": 0.9})
        with self.assertRaisesRegex(ValueError, "dump aggregate differs"):
            audit.audit_run(self.input, self.api)

    def response_dump(self, reasons, scores, datasets=None, metrics=None):
        rows = []
        for i, (reason, score) in enumerate(zip(reasons, scores, strict=True)):
            rows.append(
                {
                    "uid": str(i),
                    "row_ordinal": i,
                    "token_provenance": "finalized_trajectory",
                    "generator_engine_index": 0,
                    "prompt_token_ids": [i],
                    "response_ids": [7] * i,
                    "prompt_token_ids_sha256": audit.canonical_sha([i]),
                    "response_ids_sha256": audit.canonical_sha([7] * i),
                    "response_length": i,
                    "score": score,
                    "stop_reason": reason,
                    "data_source": "gsm8k" if datasets is None else datasets[i],
                }
            )
        self.dump_paths[0].write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.write(self.dump_paths[0].parent / "aggregated_results.jsonl", metrics)
        return rows

    def test_response_supplements_preserve_legacy_primary_and_use_all_sequence_denominator(self):
        # Scalar and tokenwise scores differ from final-token outcome semantics.
        primary = {
            "eval/all/avg_score": 0.25,
            "eval/all/pass_at_1": 0.25,
            "eval/gsm8k/avg_score": 0.25,
            "eval/gsm8k/pass_at_1": 0.25,
        }
        self.response_dump(["length", "stop", "abort", "unknown"], [0.5, [2, -1], -0.5, []], metrics=primary)
        summary, records = audit.audit_eval_dump(self.request["output"]["export_root"], 0, 4, 1, primary, True)
        values = summary["metrics"]
        self.assertEqual([r[3] for r in records], [0.5, 1, -0.5, 0])
        self.assertEqual([len(r) for r in records], [6] * 4)
        self.assertEqual({key: values[key] for key in primary}, primary)
        self.assertEqual(values["eval/all/response_tokens"], 6)
        self.assertEqual(values["eval/all/response_tokens_mean"], 1.5)
        self.assertEqual(values["eval/all/response_tokens_max"], 3)
        self.assertEqual(values["eval/all/known_stop_count"], 4)
        self.assertEqual(values["eval/all/length_stop_fraction"], 0.25)
        self.assertEqual(values["eval/all/completed_stop_fraction"], 0.25)
        self.assertEqual(values["eval/all/length_stop_score_contribution"], 0.125)
        self.assertEqual(values["eval/all/completed_stop_score_contribution"], 0.25)
        # Abort is known but neither completed nor length; contributions need not sum to raw score.

    def test_unknown_stop_omits_global_contributions_but_keeps_covered_dataset(self):
        primary = {
            "eval/all/avg_score": 0.5,
            "eval/all/pass_at_1": 0.5,
            "eval/a_b/avg_score": 1.0,
            "eval/a_b/pass_at_1": 1.0,
            "eval/other/avg_score": 0.0,
            "eval/other/pass_at_1": 0.0,
        }
        self.response_dump(["length", "eos", None, ""], [1, 1, 0, 0], ["a/b", "a/b", "other", "other"], primary)
        result, _ = audit.audit_eval_dump(self.request["output"]["export_root"], 0, 4, 1, primary, True)
        values = result["metrics"]
        self.assertEqual(values["eval/all/stop_reason_coverage"], 0.5)
        self.assertEqual(values["eval/all/unknown_stop_count"], 2)
        self.assertNotIn("eval/all/length_stop_fraction", values)
        self.assertNotIn("eval/all/completed_stop_score_contribution", values)
        self.assertEqual(values["eval/a_b/completed_stop_score_contribution"], 0.5)
        self.assertEqual(values["eval/other/response_tokens"], 5)

    def test_completed_stop_labels_are_exact_and_not_semantic_answer_checks(self):
        reasons = ["complete", "end_turn", "eos", "stop", "STOP", " stop", "abort"]
        primary = {f"eval/{source}/{name}": 1.0 for source in ("all", "gsm8k") for name in ("avg_score", "pass_at_1")}
        self.response_dump(reasons, [1] * 7, metrics=primary)
        result, _ = audit.audit_eval_dump(self.request["output"]["export_root"], 0, 7, 1, primary, True)
        self.assertEqual(result["metrics"]["eval/all/completed_stop_fraction"], 4 / 7)
        self.assertEqual(result["metrics"]["eval/all/stop_reason_coverage"], 1)

    def test_advertised_supplements_are_verified_in_both_sources(self):
        for source in ("aggregate", "wandb"):
            for value in (5, float("nan")):
                with self.subTest(source=source, value=value):
                    wrong = self.metrics | {"eval/all/response_tokens": value}
                    self.write(
                        self.dump_paths[0].parent / "aggregated_results.jsonl",
                        wrong if source == "aggregate" else self.metrics,
                    )
                    with self.assertRaisesRegex(ValueError, "differs at step 0: eval/all/response_tokens"):
                        audit.audit_eval_dump(
                            self.request["output"]["export_root"],
                            0,
                            2,
                            1,
                            wrong if source == "wandb" else self.metrics,
                            True,
                        )
        valid = self.metrics | {"eval/all/response_tokens": 4, "eval/gsm8k/completed_stop_score_contribution": 0.5}
        self.write(self.dump_paths[0].parent / "aggregated_results.jsonl", valid)
        result, _ = audit.audit_eval_dump(self.request["output"]["export_root"], 0, 2, 1, valid, True)
        self.assertEqual(result["metrics"]["eval/all/response_tokens"], 4)

    def test_advertised_fraction_without_stop_coverage_fails(self):
        self.response_dump([None, "stop"], [0, 1], metrics=self.metrics)
        advertised = self.metrics | {"eval/all/length_stop_fraction": 0}
        with self.assertRaisesRegex(ValueError, "without a covered population"):
            audit.audit_eval_dump(self.request["output"]["export_root"], 0, 2, 1, advertised, True)

    def enable_required_response_metrics(self):
        values = {
            "response_tokens": 4,
            "response_tokens_mean": 2,
            "response_tokens_max": 2,
            "sequences": 2,
            "length_stop_count": 0,
            "known_stop_count": 2,
            "unknown_stop_count": 0,
            "stop_reason_coverage": 1,
            "length_stop_fraction": 0,
            "completed_stop_fraction": 1,
            "length_stop_score_contribution": 0,
            "completed_stop_score_contribution": 0.5,
        }
        metrics = self.metrics | {
            f"eval/{source}/{key}": value for source in ("all", "gsm8k") for key, value in values.items()
        }
        for path in self.dump_paths:
            self.write(path.parent / "aggregated_results.jsonl", metrics)
        self.history[0].update(metrics)
        self.history[-1].update(metrics)
        self.input["require_eval_response_metrics"] = True
        return metrics

    def test_run_spec_requires_response_metrics_in_both_sources(self):
        for source in ("dump aggregate", "W&B"):
            with self.subTest(source=source):
                metrics = self.enable_required_response_metrics()
                missing = "eval/gsm8k/completed_stop_score_contribution"
                if source == "dump aggregate":
                    metrics.pop(missing)
                    self.write(self.dump_paths[0].parent / "aggregated_results.jsonl", metrics)
                else:
                    self.history[0].pop(missing)
                with self.assertRaisesRegex(ValueError, f"{source} missing required response metrics"):
                    audit.audit_run(self.input, self.api)
        self.enable_required_response_metrics()
        result = audit.audit_run(self.input, self.api)
        self.assertTrue(all(row["response_metrics_required"] for row in result["eval_dumps"]))

    def test_required_response_metrics_excludes_unavailable_partial_stop_fractions(self):
        values = {
            "response_tokens": 1,
            "response_tokens_mean": 0.5,
            "response_tokens_max": 1,
            "sequences": 2,
            "length_stop_count": 0,
            "known_stop_count": 1,
            "unknown_stop_count": 1,
            "stop_reason_coverage": 0.5,
        }
        metrics = self.metrics | {
            f"eval/{source}/{key}": value for source in ("all", "gsm8k") for key, value in values.items()
        }
        self.response_dump([None, "stop"], [0, 1], metrics=metrics)
        result, _ = audit.audit_eval_dump(
            self.request["output"]["export_root"], 0, 2, 1, metrics, True, require_eval_response_metrics=True
        )
        self.assertTrue(result["response_metrics_required"])
        self.assertNotIn("eval/all/completed_stop_fraction", result["metrics"])
        self.assertNotIn("eval/all/length_stop_score_contribution", result["metrics"])

    def test_missing_update_fails(self):
        self.history.pop(1)
        with self.assertRaisesRegex(ValueError, "optimizer-step coverage"):
            audit.audit_run(self.input, self.api)

    def test_nonfinite_optional_metric_fails(self):
        self.history[1]["policy/optional"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite metric"):
            audit.audit_run(self.input, self.api)

    def test_checkpoint_object_fails(self):
        self.write(self.root / "checkpoints/latest_ckpt_global_step.txt", {})
        with self.assertRaisesRegex(ValueError, "checkpoint objects"):
            audit.audit_run(self.input, self.api)

    def test_tis_requires_zero_skip_coverage(self):
        self.run.config["trainer"]["algorithm"].update(use_tis=True, require_rollout_logprobs=True)
        for row in self.history[1:]:
            row.update({"tis/batch_skipped_no_logprobs": 0, "tis/skipped_fraction": 0})
        audit.audit_run(self.input, self.api)
        self.history[-1]["tis/skipped_fraction"] = 0.5
        with self.assertRaisesRegex(ValueError, "TIS correction was skipped"):
            audit.audit_run(self.input, self.api)

    def test_flattened_config_is_accepted(self):
        def flatten(value, prefix=""):
            result = {}
            for key, child in value.items():
                path = prefix + key
                if isinstance(child, dict):
                    result.update(flatten(child, path + "/"))
                else:
                    result[path] = child
            return result

        self.run.config = flatten(self.run.config)
        audit.audit_run(self.input, self.api)

    def locked_validation_fixture(self):
        uri = str(self.root / "validation")
        self.request["validation_data"] = [{"uri": uri, "relative_path": "validation.parquet"}]
        old_digest = self.digest
        self.digest = audit.canonical_sha(
            {"schema_version": 2, "request": self.request, "receipt_uri": self.receipt_uri}
        )
        self.receipt["request_fingerprint"] = self.digest
        self.write(self.receipt_uri, self.receipt)
        self.write(self.root / "terminal.json", self.envelope)
        self.write(self.root / "attempts/fixture-attempt.json", self.envelope)
        resolved_path = self.root / "resolved.json"
        resolved_path.write_text(resolved_path.read_text().replace(old_digest, self.digest))
        self.run.config["trainer"]["completion"]["request_fingerprint"] = self.digest
        manifest = {
            "dataset": "openai/gsm8k",
            "revision": "fixed-data-revision",
            "rows": {"train": ["train/0", "train/1"], "test": ["test/128", "test/129"]},
            "validation_window": {
                "purpose": "locked_holdout",
                "split": "test",
                "offset": 128,
                "count": 2,
                "excluded_development_rows": [0, 128],
            },
        }
        path = self.root / "validation/selection.json"
        self.write(path, manifest)
        self.input["locked_validation"] = {
            "manifest_uri": str(path),
            "dataset_revision": "fixed-data-revision",
            "offset": 128,
            "count": 2,
        }
        return path, manifest

    def test_locked_manifest_is_bound_to_request_dataset_and_exact_source_window(self):
        self.locked_validation_fixture()
        result = audit.audit_run(self.input, self.api)
        self.assertTrue(result["locked_validation"]["source_indices_verified"])
        self.assertEqual(result["locked_validation"]["excluded_development_rows"], [0, 128])
        self.input["locked_validation"]["manifest_uri"] = str(self.root / "unrelated/selection.json")
        with self.assertRaisesRegex(ValueError, "request.s validation artifact"):
            audit.audit_run(self.input, self.api)

    def test_locked_manifest_does_not_certify_a_request_for_training_rows(self):
        self.locked_validation_fixture()
        self.request["validation_data"][0]["relative_path"] = "train.parquet"
        with self.assertRaisesRegex(ValueError, "different split file"):
            audit.audit_locked_validation(self.input, self.request)

    def test_locked_manifest_failure_precedes_any_wandb_quality_read(self):
        path, manifest = self.locked_validation_fixture()

        def unexpected_quality_read(_path):
            self.fail("W&B was accessed before validation of the locked source window")

        self.api.run = unexpected_quality_read
        for bad_ids in [["test/0", "test/1"], ["test/129", "test/128"], ["test/128"], ["test/128", "test/128"]]:
            with self.subTest(ids=bad_ids):
                manifest["rows"]["test"] = bad_ids
                self.write(path, manifest)
                with self.assertRaisesRegex(ValueError, "source IDs"):
                    audit.audit_run(self.input, self.api)

    def test_locked_manifest_rejects_wrong_revision_purpose_or_count(self):
        path, manifest = self.locked_validation_fixture()
        self.input["locked_validation"]["dataset_revision"] = "other-revision"
        with self.assertRaisesRegex(ValueError, "dataset/revision"):
            audit.audit_run(self.input, self.api)
        self.input["locked_validation"]["dataset_revision"] = manifest["revision"]
        manifest["validation_window"]["purpose"] = "development"
        self.write(path, manifest)
        with self.assertRaisesRegex(ValueError, "window declaration"):
            audit.audit_run(self.input, self.api)
        manifest["validation_window"]["purpose"] = "locked_holdout"
        self.write(path, manifest)
        self.input["expected_eval_rows"] = 128
        with self.assertRaisesRegex(ValueError, "row count"):
            audit.audit_run(self.input, self.api)

    def test_remote_style_reader_rejects_oversized_line_without_unbounded_readline(self):
        path = self.dump_paths[0]
        path.write_bytes(b" " * (audit.MAX_EVAL_LINE_BYTES + 1))
        filesystem = BufferedLocalFileSystem()
        with patch.object(audit, "fs_path", lambda uri: (filesystem, uri)):
            with self.assertRaisesRegex(ValueError, "audit bounds"):
                audit.audit_run(self.input, self.api)

    def test_resolved_override_types_preserve_launcher_contract(self):
        arguments = [
            "++trainer.hf_hub_repo_id=null",
            "++generator.chat_template_kwargs.enable_thinking=false",
            "++trainer.max_ckpts_to_keep=2",
            "++trainer.ckpt_interval=-1",
            "++terminal_bench_config.trials_dir='s3://fixture-region/tmp/ttl=14d/attempts/trace_jobs'",
            "++data.val_data=['/tmp/validation one.parquet','/tmp/validation-two.parquet']",
            '++trainer.completion.attempt_id="00123"',
        ]
        self.assertEqual(
            audit.parse_hydra_args(arguments),
            {
                "trainer.hf_hub_repo_id": None,
                "generator.chat_template_kwargs.enable_thinking": False,
                "trainer.max_ckpts_to_keep": 2,
                "trainer.ckpt_interval": -1,
                "terminal_bench_config.trials_dir": "s3://fixture-region/tmp/ttl=14d/attempts/trace_jobs",
                "data.val_data": ["/tmp/validation one.parquet", "/tmp/validation-two.parquet"],
                "trainer.completion.attempt_id": "00123",
            },
        )


@pytest.fixture
def paired_study_inputs():
    study = {
        "label": "cadence",
        "pairs": [
            {"seed": seed, "reference": f"reference-{seed}", "candidate": f"candidate-{seed}"} for seed in (17, 29)
        ],
        "initial_step": 0,
        "final_step": 2,
        "differing_config_paths": ["trainer.fully_async.weight_sync_interval"],
        "bootstrap_seed": 2026,
        "bootstrap_repetitions": 500,
    }
    results, snapshots = {}, {}
    # Two questions, two responses per question. The response-level scores
    # intentionally differ, so using only one response gives the wrong mean.
    scores = {
        "reference-17": {0: [[0, 0], [0, 0]], 2: [[0, 1], [0, 1]]},
        "candidate-17": {0: [[0, 1], [0, 1]], 2: [[1, 1], [1, 1]]},
        "reference-29": {0: [[0, 1], [0, 1]], 2: [[0, 1], [0, 1]]},
        "candidate-29": {0: [[0, 0], [0, 0]], 2: [[0, 1], [1, 1]]},
    }
    for pair in study["pairs"]:
        for arm in ("reference", "candidate"):
            label = pair[arm]
            results[label] = {"training_evidence_pass": True, "clean_end_to_end": True}
            snapshots[label] = {
                "request": {
                    "run_id": label,
                    "runtime": {"commit": "a" * 40},
                    "model": {"identity": "frozen-model"},
                    "train_data": ["train-revision"],
                    "validation_data": ["val-revision"],
                    "topology": {"policy_gpus": 8, "inference_gpus": 8},
                    "seed": pair["seed"],
                },
                "source_config": {
                    "entrypoint": "fully_async",
                    "data": {"epoch_seeded_shuffle": True},
                    "trainer": {"fully_async": {"weight_sync_interval": 1 if arm == "reference" else 4}},
                },
                "expected_steps": 2,
                "resolved_epoch_seeded_shuffle": True,
                "expected_eval_steps": [0, 2],
                "initial_eval_repeat_count": 1,
                "evaluations": {
                    step: [
                        [f"question-{uid}", audit.canonical_sha([uid, 3]), "response-hash", reward, "stop", 0]
                        for uid, responses in enumerate(questions)
                        for reward in responses
                    ]
                    for step, questions in scores[label].items()
                },
            }
    return study, results, snapshots


def test_paired_study_point_estimate_and_initial_adjustment(paired_study_inputs):
    result = audit.paired_evaluation_study(*paired_study_inputs)
    first, second = result["seed_results"]
    assert first["final_reward_delta"] == 0.5
    assert first["initial_reward_delta"] == 0.5
    assert first["initial_to_final_change_delta"] == 0
    assert first["reference_initial_to_final_change"] == 0.5
    assert first["candidate_initial_to_final_change"] == 0.5
    assert second["final_reward_delta"] == 0.25
    assert second["initial_reward_delta"] == -0.5
    assert second["initial_to_final_change_delta"] == 0.75
    assert result["mean_final_reward_delta"] == 0.375
    assert result["mean_initial_to_final_change_delta"] == 0.375
    assert result["questions"] == 2
    assert result["seed_final_reward_delta_range"] == [0.25, 0.5]
    assert result["bootstrap"]["percentile_intervals"] == {
        "final_reward_delta": [0.25, 0.5],
        "initial_to_final_change_delta": [0.25, 0.5],
    }
    assert result["source_order"]["actual_consumed_order_verified"] is False
    assert result["evaluation_scope"] == {"classification": "development_or_unspecified"}
    assert len(json.dumps(result)) < 8192
    assert "response-hash" not in json.dumps(result)


def test_pool_bootstrap_reproduces_existing_seed_averaged_auditor(paired_study_inputs):
    study, results, snapshots = paired_study_inputs
    legacy = audit.paired_evaluation_study(study, results, snapshots)
    arms = {"reference": {}, "candidate": {}}
    for pair in study["pairs"]:
        for arm in arms:
            scores = {}
            for _, question_hash, _, value, *_ in snapshots[pair[arm]]["evaluations"][study["final_step"]]:
                scores.setdefault(question_hash, []).append(value)
            arms[arm][pair["seed"]] = scores
    actual = bootstrap(
        arms["reference"],
        arms["candidate"],
        seed=study["bootstrap_seed"],
        repetitions=study["bootstrap_repetitions"],
        inference=SeedInference.FIXED,
    )
    assert actual.delta == pytest.approx(legacy["mean_final_reward_delta"], abs=1e-12)
    assert actual.interval == pytest.approx(legacy["bootstrap"]["percentile_intervals"]["final_reward_delta"], abs=1e-12)
    assert actual.seed_deltas == pytest.approx([row["final_reward_delta"] for row in legacy["seed_results"]], abs=1e-12)


def test_paired_study_duplicate_responses_do_not_inflate_questions_or_precision(paired_study_inputs):
    expected = audit.paired_evaluation_study(*paired_study_inputs)
    for snapshot in paired_study_inputs[2].values():
        for step, records in snapshot["evaluations"].items():
            snapshot["evaluations"][step] = [row for row in records for _ in range(3)]
    actual = audit.paired_evaluation_study(*paired_study_inputs)
    assert actual == expected


def test_paired_study_constant_difference_has_degenerate_interval(paired_study_inputs):
    study, results, snapshots = paired_study_inputs
    for pair in study["pairs"]:
        for step in (0, 2):
            for row in snapshots[pair["reference"]]["evaluations"][step]:
                row[3] = 0.25 if step == 0 else 0.5
            for row in snapshots[pair["candidate"]]["evaluations"][step]:
                row[3] = 0.25 if step == 0 else 0.75
    result = audit.paired_evaluation_study(study, results, snapshots)
    assert result["mean_final_reward_delta"] == 0.25
    assert result["bootstrap"]["percentile_intervals"] == {
        "final_reward_delta": [0.25, 0.25],
        "initial_to_final_change_delta": [0.25, 0.25],
    }


@pytest.mark.parametrize("unverified_arm", [None, "absent", "different_window"])
def test_paired_study_labels_holdout_only_when_every_run_has_same_verified_window(paired_study_inputs, unverified_arm):
    window = {
        "source_indices_verified": True,
        "dataset": "openai/gsm8k",
        "revision": "locked-revision",
        "purpose": "locked_holdout",
        "offset": 128,
        "count": 1191,
    }
    results = paired_study_inputs[1]
    for result in results.values():
        result["locked_validation"] = copy.deepcopy(window)
    if unverified_arm == "absent":
        results["candidate-29"]["locked_validation"] = None
    elif unverified_arm == "different_window":
        results["candidate-29"]["locked_validation"]["offset"] = 129
    result = audit.paired_evaluation_study(*paired_study_inputs)
    expected = (
        {"classification": "verified_locked_holdout", "validation": window}
        if unverified_arm is None
        else {"classification": "development_or_unspecified"}
    )
    assert result["evaluation_scope"] == expected


@pytest.mark.parametrize("corruption", ["permutation", "prompt_hash", "uid", "within_uid_hash", "missing_question"])
def test_paired_study_rejects_question_identity_changes(paired_study_inputs, corruption):
    records = paired_study_inputs[2]["candidate-29"]["evaluations"][2]
    if corruption == "permutation":
        records[:] = records[2:] + records[:2]
    elif corruption == "missing_question":
        del records[2:]
    else:
        records[0][0 if corruption == "uid" else 1] = "different"
        if corruption == "prompt_hash":
            records[1][1] = "different"
    with pytest.raises(ValueError, match=r"(identity|order|different prompts)"):
        audit.paired_evaluation_study(*paired_study_inputs)


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_seed",
        "duplicate_seed",
        "wrong_seed",
        "failed_run",
        "runtime",
        "endpoint",
        "undeclared_config",
        "different_seed_arm",
        "source_order",
        "resolved_source_order",
        "launcher_override",
        "reused_run",
    ],
)
def test_paired_study_rejects_unmatched_or_unaudited_controls(paired_study_inputs, corruption):
    study, results, snapshots = paired_study_inputs
    candidate = snapshots["candidate-29"]
    if corruption == "missing_seed":
        del results["candidate-29"]
    elif corruption == "duplicate_seed":
        study["pairs"][1]["seed"] = 17
    elif corruption == "wrong_seed":
        candidate["request"]["seed"] = 31
    elif corruption == "failed_run":
        results["candidate-29"]["clean_end_to_end"] = False
    elif corruption == "runtime":
        candidate["request"]["runtime"]["commit"] = "b" * 40
    elif corruption == "launcher_override":
        candidate["request"]["overrides"] = ["++trainer.policy.optimizer_config.lr=0.1"]
    elif corruption == "reused_run":
        candidate["request"]["run_id"] = "reference-29"
    elif corruption == "endpoint":
        candidate["expected_steps"] = 3
    elif corruption == "source_order":
        candidate["source_config"]["data"]["epoch_seeded_shuffle"] = False
    elif corruption == "resolved_source_order":
        for snapshot in snapshots.values():
            snapshot["resolved_epoch_seeded_shuffle"] = False
    elif corruption == "different_seed_arm":
        candidate["source_config"]["trainer"]["fully_async"]["weight_sync_interval"] = 5
    else:
        for label in ("candidate-17", "candidate-29"):
            snapshots[label]["source_config"]["trainer"]["learning_rate"] = 0.01
    with pytest.raises(ValueError):
        audit.paired_evaluation_study(study, results, snapshots)


def test_completed_stop_secondary_opposite_ranking_preserves_primary_and_all_response_denominator(paired_study_inputs):
    study, results, snapshots = paired_study_inputs
    for pair in study["pairs"]:
        for arm in ("reference", "candidate"):
            for step, rows in snapshots[pair[arm]]["evaluations"].items():
                for index, row in enumerate(rows):
                    row[3] = 0.25 if step == 0 else (0.5 if arm == "reference" else 0.75)
                    row[4] = "stop" if step == 0 or arm == "reference" or index % 2 == 0 else "length"
    original = copy.deepcopy(snapshots)
    primary = audit.paired_evaluation_study(study, results, snapshots)
    result = audit.paired_evaluation_study(study | {"include_completed_stop_score": True}, results, snapshots)
    secondary = result.pop("secondary_completed_stop_score")
    assert json.dumps(result, sort_keys=True) == json.dumps(primary, sort_keys=True)
    assert json.dumps(snapshots, sort_keys=True) == json.dumps(original, sort_keys=True)
    assert snapshots == original  # Six-column records and their raw scores are never transformed in place.
    assert primary["mean_final_reward_delta"] == 0.25
    # Candidate: one 0.75 completed response + one length-stopped response per UID, divided by TWO.
    # Reference: both 0.5 responses completed. Thus completed score reverses the raw-score ranking.
    assert secondary["mean_final_reward_delta"] == -0.125
    assert secondary["mean_initial_to_final_change_delta"] == -0.125
    assert secondary["questions"] == 2 and secondary["training_seeds"] == [17, 29]
    assert secondary["bootstrap"]["percentile_intervals"] == {
        "final_reward_delta": [-0.125, -0.125],
        "initial_to_final_change_delta": [-0.125, -0.125],
    }
    assert [row["candidate_final_reward"] for row in secondary["seed_results"]] == [0.375, 0.375]
    assert secondary["metric"] == "completed_stop_score"
    assert "ALL responses" in secondary["reward_reduction"]
    assert "not conditional accuracy" in secondary["interpretation"]
    for snapshot in snapshots.values():
        for step, rows in snapshot["evaluations"].items():
            snapshot["evaluations"][step] = [row for row in rows for _ in range(3)]
    repeated = audit.paired_evaluation_study(study | {"include_completed_stop_score": True}, results, snapshots)
    assert repeated["secondary_completed_stop_score"] == secondary


@pytest.mark.parametrize("reason", [None, ""])
@pytest.mark.parametrize("step", [0, 2])
def test_completed_stop_secondary_requires_coverage_at_both_endpoints(paired_study_inputs, reason, step):
    study, results, snapshots = paired_study_inputs
    snapshots["candidate-29"]["evaluations"][step][0][4] = reason
    # The existing primary question score does not depend on stop-label coverage.
    assert "secondary_completed_stop_score" not in audit.paired_evaluation_study(study, results, snapshots)
    with pytest.raises(ValueError, match="requires full evaluation stop coverage"):
        audit.paired_evaluation_study(study | {"include_completed_stop_score": True}, results, snapshots)


def test_completed_stop_secondary_known_noncompleted_labels_contribute_zero(paired_study_inputs):
    study, results, snapshots = paired_study_inputs
    for pair in study["pairs"]:
        for row, reason in zip(
            snapshots[pair["candidate"]]["evaluations"][2], ["abort", "unknown", "STOP", " stop"], strict=True
        ):
            row[4] = reason
    result = audit.paired_evaluation_study(study | {"include_completed_stop_score": True}, results, snapshots)
    assert result["secondary_completed_stop_score"]["mean_final_reward_delta"] == -0.5
    assert result["mean_final_reward_delta"] == 0.375


if __name__ == "__main__":
    unittest.main(verbosity=2)

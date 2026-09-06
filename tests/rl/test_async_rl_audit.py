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

from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractBufferedFile

from experiments.post_training import async_rl_audit as audit


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


if __name__ == "__main__":
    unittest.main(verbosity=2)

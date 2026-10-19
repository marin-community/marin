"""Synchronous frozen-token probe at the initial and post-sync policy weights."""

from __future__ import annotations

import asyncio
import math
import os
import random
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime

import numpy as np
import ray
import torch
from finestore.rl import mismatch_probe as mismatch
from loguru import logger

from skyrl_train.config.mismatch_probe import (
    CACHE_BOTH,
    CACHE_OFF,
    CACHE_ON,
    GENERATION_SCORER,
    RESCORE_SCORER,
    TRAINER_SCORER,
    rescore_label,
    trainer_label,
)
from skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch
from skyrl_train.group_admission import GroupAdvantageInvariant, GroupAdvantageKind
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.inference_engines.vllm_teacher_oracle import tokenizer_vocabulary_fingerprint
from skyrl_train.mismatch_probe.archive import (
    BUILDING_STATUS,
    COMPLETE_STATUS,
    MismatchArchive,
    read_frozen_probe,
)
from skyrl_train.mismatch_probe.protocol import (
    probe_hash,
    request_seed,
    ProbeSample,
)
from skyrl_train.mismatch_probe.modes import NATIVE_MODE, REPEAT_MODE, TRAINER_MODES
from skyrl_train.mismatch_probe.provenance import manifest
from skyrl_train.models.megatron_router_replay import SENTINEL_EXPERT_ID
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.trajectory_runners.trajectory_processing import (
    concatenate_trajectory_batches,
    prepare_trajectory_request,
    scalar_reward_token_credit,
)


@dataclass(frozen=True)
class EncodedRoutes:
    """Router choices encoded with their NumPy shape and dtype."""

    data: bytes | None
    shape: list[int] | None
    dtype: str | None
    replacements: bytes | None = None


@dataclass(frozen=True)
class ProbeSamples:
    """Trajectory rows and stable request identities for one probe."""

    trajectory: dict
    prompt_ids: list[str]
    sample_ids: list[str]
    seeds: list[int]


@dataclass(frozen=True)
class BatchLayout:
    sample_ids: list[str]
    native_order: list[int]
    repeat_order: list[int]
    padded_rows: int
    native_micro_batch_size: int
    repeat_micro_batch_size: int
    collated_bytes: int
    collated_route_bytes: int
    nonzero_advantage_samples: int


def _encode_routes(routes: np.ndarray | None, length: int) -> EncodedRoutes:
    if routes is None:
        return EncodedRoutes(None, None, None)
    value = np.ascontiguousarray(routes[:length])
    return EncodedRoutes(value.tobytes(), list(value.shape), str(value.dtype))


def _reorder_batch(batch: TrainingInputBatch, order: list[int]) -> TrainingInputBatch:
    reordered = TrainingInputBatch.cat([batch[index] for index in order])
    reordered.metadata = dict(batch.metadata)
    if "uids" in reordered.metadata:
        reordered.metadata["uids"] = [batch.metadata["uids"][i] for i in order]
    return reordered


def _candidate_logprobs(output, overrides):
    ids = output.get("student_topk_indices")
    values = output.get("behavior_topk_logprobs")
    if ids is None or values is None or len(ids) != len(overrides) or len(values) != len(overrides):
        raise ValueError("vLLM re-read returned incomplete requested candidates")
    result = []
    for override, candidates, scores in zip(overrides, ids, values, strict=True):
        token = override["logprob_token_ids"][0]
        if len(candidates) != 1 or candidates[0] != [token] or len(scores) != 1 or len(scores[0]) != 1:
            raise ValueError("vLLM re-read omitted the frozen response token")
        result.append(float(scores[0][0]))
    return result


def _prompt_group_contract(trainer, samples_per_prompt: int) -> GroupAdvantageInvariant:
    current = trainer.group_advantage_invariant
    minimum = current.minimum_group_size
    if current.kind is GroupAdvantageKind.MINIMUM_BASELINE_ELIGIBLE:
        if samples_per_prompt < 2:
            raise ValueError("mismatch probe requires at least two samples for a baseline-eligible group")
        minimum = min(minimum, samples_per_prompt)
    return replace(current, physical_group_size=samples_per_prompt, minimum_group_size=minimum)


class ProbeCollector:
    """Frozen probe data and engine collection independent of callback scheduling."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.spec = cfg.trainer.mismatch_probe
        self.updates = int(self.spec.updates)
        self.archive_uri = self.spec.archive_uri
        self.archive: MismatchArchive | None = None
        self.probes = None
        self.training_input: TrainingInputBatch | None = None
        self.probe_hash: str | None = None
        self.source_manifest = None
        self.generation_scores: list[np.ndarray] | None = None
        self.timing: dict[str, float] = {}
        self.batch_layout: BatchLayout | None = None
        self.metrics: dict[str, object] = {}
        self.created_at = datetime.now(UTC).isoformat()
        self.starting_global_step: int | None = None
        self.scored_global_steps: dict[int, int] = {}
        self.tokenizer_fingerprint: str | None = None

    async def _generate(self, trainer):
        if trainer.eval_dataset is None:
            raise ValueError("mismatch probe requires the run's validation dataset")
        count = int(self.spec.prompts.count)
        samples = int(self.spec.prompts.samples_per_prompt)
        if len(trainer.eval_dataset) < count:
            raise ValueError(
                f"mismatch probe requested {count} validation prompts but only {len(trainer.eval_dataset)} exist"
            )
        prompts = trainer.eval_dataset.collate_fn([trainer.eval_dataset[i] for i in range(count)])
        batches = []
        seeds = []
        sample_ids = []
        uids = []
        await trainer.trajectory_runner.start_eval_session(
            run_name=trainer.cfg.trainer.get("run_name") or "mismatch-probe",
            eval_step=trainer.global_step,
            val_set_name="mismatch-probe",
        )
        try:
            for prompt in prompts:
                for repetition in range(samples):
                    seed = request_seed(int(self.spec.seed), prompt["uid"], repetition)
                    sampling = get_sampling_params_for_backend(
                        trainer.cfg.generator.backend, trainer.cfg.generator.sampling_params
                    )
                    sampling["seed"] = seed
                    request, _ = prepare_trajectory_request([prompt], 1, sampling, "eval", 0)
                    request.trajectory_ids[0].repetition_id = repetition
                    batch = await trainer.trajectory_runner.run(request)
                    batches.append(batch)
                    seeds.append(seed)
                    sample_ids.append(f"{prompt['uid']}:{repetition}")
                    uids.append(prompt["uid"])
        finally:
            await trainer.trajectory_runner.stop_eval_session()
        trajectory = concatenate_trajectory_batches(
            batches,
            require_rollout_logprobs=True,
            tis_lcs_alert_threshold=float(trainer.cfg.trainer.algorithm.tis_lcs_alert_threshold),
        )
        return ProbeSamples(trajectory, uids, sample_ids, seeds)

    def _from_source(self):
        source = read_frozen_probe(self.spec.reuse_probe)
        manifest, rows, generations = source.manifest, source.probes, source.generations
        expected = int(self.spec.prompts.count) * int(self.spec.prompts.samples_per_prompt)
        if len(rows) != expected:
            raise ValueError(f"reuse_probe has {len(rows)} samples but this recipe requests {expected}")
        group_counts: dict[str, int] = {}
        for row in rows:
            group_counts[row.prompt_id] = group_counts.get(row.prompt_id, 0) + 1
        if len(group_counts) != int(self.spec.prompts.count) or any(
            count != int(self.spec.prompts.samples_per_prompt) for count in group_counts.values()
        ):
            raise ValueError("reuse_probe prompt groups do not match this recipe")
        self.source_manifest = manifest
        self.probes = rows
        self.probe_hash = manifest.probe_hash
        routes = []
        for row in rows:
            if row.routed_experts is None:
                routes.append(None)
            else:
                decoded = np.frombuffer(row.routed_experts, dtype=row.routed_experts_dtype).reshape(
                    row.routed_experts_shape
                )
                routes.append(decoded.copy())
        if any(route is None for route in routes) and any(route is not None for route in routes):
            raise ValueError("reuse_probe source mixes captured and missing routed-expert rows")
        self.generation_scores = [np.asarray(generations[row.sample_id].logprobs, dtype=np.float32) for row in rows]
        trajectory = {
            "prompt_token_ids": [row.prompt_token_ids for row in rows],
            "response_ids": [row.trainer_input_ids for row in rows],
            "rewards": [scalar_reward_token_credit(row.reward or 0.0, row.trainer_input_ids) for row in rows],
            "loss_masks": [[int(value) for value in row.loss_mask] for row in rows],
            "rollout_logprobs": self.generation_scores,
            "rollout_routed_experts": routes if all(route is not None for route in routes) else None,
            "stop_reasons": ["reused"] * len(rows),
            "rollout_metrics": None,
        }
        return ProbeSamples(
            trajectory,
            [row.prompt_id for row in rows],
            [row.sample_id for row in rows],
            [row.request_seed for row in rows],
        )

    def _collate_and_freeze(self, trainer, trajectory, uids, sample_ids, seeds):
        if self.source_manifest is not None:
            source_tokenizer = self.source_manifest.tokenizer_fingerprint
            self.tokenizer_fingerprint = tokenizer_vocabulary_fingerprint(trainer.inference_engine_client.tokenizer)
            if not source_tokenizer or source_tokenizer != self.tokenizer_fingerprint:
                raise ValueError("reuse_probe tokenizer fingerprint differs from the source archive")
        generated = trajectory["response_ids"]
        if generated is None:
            raise ValueError("mismatch probe requires exact vLLM response IDs from the trajectory runner")
        raw_rewards = trajectory["rewards"]
        rewards = [float(sum(value)) if isinstance(value, list) else float(value) for value in raw_rewards]
        original_contract = trainer.group_advantage_invariant
        original_metrics = dict(trainer.all_metrics)
        try:
            trainer.group_advantage_invariant = _prompt_group_contract(
                trainer, int(self.spec.prompts.samples_per_prompt)
            )
            if self.source_manifest is None:
                trajectory = trainer.postprocess_trajectory_batch(trajectory, uids)
            training_input = trainer.convert_to_training_input(trajectory, uids)
            training_input["values"] = None
            if self.source_manifest is None:
                training_input = trainer.compute_advantages_and_returns(training_input)
        finally:
            trainer.group_advantage_invariant = original_contract
            trainer.all_metrics = original_metrics

        response_width = training_input["response_mask"].shape[1]
        prompt_width = training_input["sequences"].shape[1] - response_width
        padded_count = len(training_input["sequences"]) - len(sample_ids)
        if padded_count < 0 or padded_count != training_input.metadata.get("pad_size", 0):
            raise ValueError("mismatch probe training collation changed the number of frozen samples")
        rows = []
        schema = mismatch
        for position, sample_id in enumerate(sample_ids):
            response_length = len(trajectory["response_ids"][position])
            prompt_tokens = training_input["sequences"][position, :prompt_width]
            prompt_mask = training_input["attention_mask"][position, :prompt_width].bool()
            trainer_prompt = prompt_tokens[prompt_mask].tolist()
            trainer_response = training_input["sequences"][
                position, prompt_width : prompt_width + response_length
            ].tolist()
            route_rows = training_input.routed_expert_rows
            routes = None if route_rows is None else route_rows.rows[position]
            encoded_routes = _encode_routes(routes, response_length)
            route_valid_mask = None
            if encoded_routes.shape is not None:
                if len(encoded_routes.shape) != 3 or encoded_routes.shape[0] != response_length:
                    raise ValueError("captured routes must align with the frozen response tokens")
                route_valid_mask = (routes[:response_length] != SENTINEL_EXPERT_ID).any(axis=-1).tolist()
                if self.source_manifest is not None:
                    route_valid_mask = self.probes[position].route_valid_mask
            advantage = None
            if self.source_manifest is None:
                valid = training_input["loss_mask"][position, :response_length].bool()
                values = training_input["advantages"][position, :response_length][valid]
                advantage = float(values.mean().item()) if values.numel() else 0.0
            else:
                advantage = self.probes[position].advantage
            rows.append(
                schema.ProbeRow(
                    probe_hash="pending",
                    sample_id=sample_id,
                    prompt_id=uids[position],
                    prompt_token_ids=list(trajectory["prompt_token_ids"][position]),
                    trainer_prompt_ids=trainer_prompt,
                    vllm_output_ids=list(generated[position]),
                    trainer_input_ids=trainer_response,
                    response_mask=training_input["response_mask"][position, :response_length].bool().tolist(),
                    loss_mask=training_input["loss_mask"][position, :response_length].bool().tolist(),
                    reward=rewards[position],
                    advantage=advantage,
                    request_seed=seeds[position],
                    batch_position=position,
                    routed_experts=encoded_routes.data,
                    routed_experts_shape=encoded_routes.shape,
                    routed_experts_dtype=encoded_routes.dtype,
                    route_valid_mask=route_valid_mask,
                )
            )
        digest = probe_hash(
            ProbeSample(
                row.sample_id,
                tuple(row.prompt_token_ids),
                tuple(row.vllm_output_ids),
                tuple(row.response_mask),
                tuple(row.loss_mask),
            )
            for row in rows
        )
        if self.source_manifest is not None and digest != self.probe_hash:
            raise ValueError("reuse_probe token hash changed during trainer collation")
        self.probe_hash = digest
        self.probes = [row.model_copy(update={"probe_hash": digest}) for row in rows]
        self.training_input = training_input
        self.generation_scores = trajectory.get("rollout_logprobs")
        if self.generation_scores is None or any(
            len(scores) != len(row.vllm_output_ids) or not all(math.isfinite(score) for score in scores)
            for row, scores in zip(self.probes, self.generation_scores, strict=True)
        ):
            raise ValueError("mismatch probe generation-time logprob vectors are incomplete or nonfinite")
        self.batch_layout = BatchLayout(
            sample_ids=sample_ids,
            native_order=list(range(len(sample_ids))),
            repeat_order=list(reversed(range(len(sample_ids)))),
            padded_rows=padded_count,
            native_micro_batch_size=int(trainer.cfg.trainer.micro_forward_batch_size_per_gpu),
            repeat_micro_batch_size=max(2, 2 * int(trainer.cfg.trainer.micro_forward_batch_size_per_gpu)),
            collated_bytes=sum(
                value.numel() * value.element_size() for value in training_input.values() if value is not None
            ),
            collated_route_bytes=0 if training_input.routed_experts is None else training_input.routed_experts.nbytes,
            nonzero_advantage_samples=(
                0
                if training_input.get("advantages") is None
                else int((training_input["advantages"].abs().sum(dim=1) > 0).sum().item())
            ),
        )

    async def _rescore_vllm(self, trainer, update: int):
        rows = self.probes
        cache_setting = self.spec.rescore_prefix_cache
        cache_modes = (CACHE_OFF, CACHE_ON) if cache_setting == CACHE_BOTH else (cache_setting,)
        result = []
        for cache_mode in cache_modes:
            if cache_mode == CACHE_OFF:
                await trainer.inference_engine_client.reset_prefix_cache()
                prefixes = [row.prompt_token_ids + row.vllm_output_ids[:-1] for row in rows]
                overrides = [{"logprob_token_ids": [row.vllm_output_ids[-1]]} for row in rows]
                sampling = {"prompt_logprobs": 1, "logprobs": 1}
            else:
                # Cached decode scores require one prefix per frozen response token.
                prefixes = [
                    row.prompt_token_ids + row.vllm_output_ids[:index]
                    for row in rows
                    for index in range(len(row.vllm_output_ids))
                ]
                overrides = [{"logprob_token_ids": [token]} for row in rows for token in row.vllm_output_ids]
                sampling = {"logprobs": 1}
            started = time.monotonic()
            engine_input = {
                "prompts": None,
                "prompt_token_ids": prefixes,
                "sampling_params": {
                    "max_tokens": 1,
                    "temperature": 1.0,
                    "skip_reading_prefix_cache": cache_mode == CACHE_OFF,
                    "seed": request_seed(int(self.spec.seed), f"reread:{update}:{cache_mode}", 0),
                    **sampling,
                },
                "session_ids": None,
            }
            engine_input["sampling_params_per_prompt"] = overrides
            output = await trainer.inference_engine_client.generate(engine_input)
            duration = time.monotonic() - started
            chosen = []
            candidates = _candidate_logprobs(output, overrides)
            if cache_mode == CACHE_OFF:
                prompt_scores = output.get("prompt_logprobs")
                if prompt_scores is None or len(prompt_scores) != len(rows):
                    raise ValueError("vLLM re-read returned incomplete prompt logprobs")
                for row, values, final_score in zip(rows, prompt_scores, candidates, strict=True):
                    chosen.append(
                        [
                            float(values[len(row.prompt_token_ids) + index][token])
                            for index, token in enumerate(row.vllm_output_ids[:-1])
                        ]
                        + [final_score]
                    )
            else:
                offset = 0
                for row in rows:
                    chosen.append(candidates[offset : offset + len(row.vllm_output_ids)])
                    offset += len(row.vllm_output_ids)
            label = rescore_label(update, cache_mode)
            self.timing[f"{label}/seconds"] = duration
            for row, values in zip(rows, chosen, strict=True):
                if len(values) != len(row.vllm_output_ids) or not all(math.isfinite(value) for value in values):
                    raise ValueError(f"vLLM re-read returned incomplete or nonfinite scores for {row.sample_id}")
                result.append(
                    mismatch.ScoreRow(
                        probe_hash=self.probe_hash,
                        sample_id=row.sample_id,
                        scorer=RESCORE_SCORER,
                        update=update,
                        global_step=trainer.global_step,
                        cache_mode=cache_mode,
                        logprobs=values,
                        forward_seconds=duration / len(rows),
                    )
                )
        return result

    def _route_observations(self, outputs, mode: str):
        """Gather PP-owned router choices by frozen sample and captured layer."""
        if not self.cfg.trainer.policy.megatron_config.moe_router_replay or all(
            row.routed_experts is None for row in self.probes
        ):
            return [None] * len(self.probes)
        choices = []
        replacements = []
        seen = []
        for row in self.probes:
            if row.routed_experts_shape is None or len(row.routed_experts_shape) != 3:
                raise ValueError("probe route observations require [response, layer, topk] captured routes")
            shape = tuple(row.routed_experts_shape)
            choices.append(np.full(shape, -1, dtype=np.int32))
            replacements.append(np.zeros(shape, dtype=np.bool_))
            seen.append(np.zeros(shape[:2], dtype=np.bool_))
        for output in outputs:
            for observation in output.metadata.get("probe_routes", []):
                sample = int(observation["sample"])
                position = int(observation["position"])
                layer = int(observation["layer"])
                if sample < 0 or sample >= len(self.probes) + self.batch_layout.padded_rows:
                    raise ValueError(f"probe router returned sample index {sample} outside the frozen batch")
                if sample >= len(self.probes):
                    continue
                if position >= choices[sample].shape[0]:
                    if observation["route_valid"]:
                        raise ValueError("probe router marked a padded response position as replayable")
                    continue
                if position < 0 or layer < 0 or layer >= choices[sample].shape[1]:
                    raise ValueError("probe router returned an invalid response position or layer")
                if seen[sample][position, layer]:
                    if not np.array_equal(
                        choices[sample][position, layer], observation["effective"]
                    ) or not np.array_equal(replacements[sample][position, layer], observation["replaced"]):
                        raise ValueError(f"conflicting probe routing observation for {sample}/{position}/{layer}")
                    continue
                seen[sample][position, layer] = True
                choices[sample][position, layer] = observation["effective"]
                replacements[sample][position, layer] = observation["replaced"]
        result = []
        for row, selected, changed, observed in zip(self.probes, choices, replacements, seen, strict=True):
            valid = np.asarray(row.route_valid_mask, dtype=np.bool_)
            if not np.all(observed[valid]):
                missing = int(np.count_nonzero(valid & ~observed))
                raise ValueError(f"{mode} missed {missing} captured routing rows for {row.sample_id}")
            result.append(
                EncodedRoutes(selected.tobytes(), list(selected.shape), str(selected.dtype), changed.tobytes())
            )
        return result

    def _trainer_scores(self, trainer, update: int):
        training_input = self.training_input
        rows = self.probes
        n = len(rows)
        route_rows = training_input.routed_expert_rows
        modes = (NATIVE_MODE, REPEAT_MODE, *self.spec.extra_trainer_modes)
        result = []
        for mode in modes:
            order = self.batch_layout.repeat_order if mode == REPEAT_MODE else self.batch_layout.native_order
            data = training_input.select(
                ["sequences", "attention_mask", *(["rollout_routed_experts"] if route_rows is not None else [])],
                ["response_length"],
            )
            data["probe_row_indices"] = torch.arange(data.batch_size, dtype=torch.long)
            if not TRAINER_MODES[mode].requires_routes and route_rows is not None:
                data.routed_expert_rows = replace(route_rows, rows=tuple(np.zeros_like(row) for row in route_rows.rows))
            if mode == REPEAT_MODE:
                data = _reorder_batch(data, order + list(range(n, data.batch_size)))
            micro_batch_size = self.batch_layout.repeat_micro_batch_size if mode == REPEAT_MODE else None
            fraction = (
                float(self.spec.filtered_replay.keep_fraction) if TRAINER_MODES[mode].requires_keep_fraction else None
            )
            data.metadata.update(
                probe_mode=mode,
                probe_keep_fraction=fraction,
                probe_micro_batch_size=micro_batch_size,
                global_step=update,
            )
            started = time.monotonic()
            outputs = ray.get(trainer.policy_model.async_run_ray_method("mesh", "probe_forward", data=data))
            values = concatenate_outputs_after_mesh_dispatch(trainer.policy_model.actor_infos, outputs)["output"][:n]
            route_observations = self._route_observations(outputs, mode)
            duration = time.monotonic() - started
            label = trainer_label(update, mode)
            self.timing[f"{label}/seconds"] = duration
            for ordered_position, original_position in enumerate(order):
                row = rows[original_position]
                length = len(row.vllm_output_ids)
                logprobs = values[ordered_position, :length].float().tolist()
                if len(logprobs) != length or not all(math.isfinite(value) for value in logprobs):
                    raise ValueError(f"{label} returned incomplete or nonfinite scores for {row.sample_id}")
                result.append(
                    mismatch.ScoreRow(
                        probe_hash=self.probe_hash,
                        sample_id=row.sample_id,
                        scorer=TRAINER_SCORER,
                        mode=mode,
                        update=update,
                        global_step=trainer.global_step,
                        logprobs=logprobs,
                        forward_seconds=duration / n,
                        expert_choices=(
                            None
                            if route_observations[original_position] is None
                            else route_observations[original_position].data
                        ),
                        expert_choices_shape=(
                            None
                            if route_observations[original_position] is None
                            else route_observations[original_position].shape
                        ),
                        expert_choices_dtype=(
                            None
                            if route_observations[original_position] is None
                            else route_observations[original_position].dtype
                        ),
                        replacement_mask=(
                            None
                            if route_observations[original_position] is None
                            else route_observations[original_position].replacements
                        ),
                    )
                )
        return result


async def collect(probe: ProbeCollector, trainer, *, update: int) -> list[mismatch.ScoreRow]:
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        started = time.monotonic()
        probe.scored_global_steps[update] = trainer.global_step
        if update == 0:
            if probe.spec.reuse_probe:
                samples = await asyncio.to_thread(probe._from_source)
            else:
                generation_started = time.monotonic()
                original_metrics = dict(trainer.all_metrics)
                try:
                    samples = await probe._generate(trainer)
                finally:
                    trainer.all_metrics = original_metrics
                probe.timing["probe/generation_seconds"] = time.monotonic() - generation_started
            probe._collate_and_freeze(
                trainer, samples.trajectory, samples.prompt_ids, samples.sample_ids, samples.seeds
            )
            if probe.source_manifest is not None:
                current = manifest(probe, trainer, status=BUILDING_STATUS)
                if (current.checkpoint_path, current.starting_global_step, current.runtime_commit) != (
                    probe.source_manifest.checkpoint_path,
                    probe.source_manifest.starting_global_step,
                    probe.source_manifest.runtime_commit,
                ):
                    raise ValueError("reuse_probe starting checkpoint, step or runtime differs from the source archive")
        elif probe.training_input is None:
            raise RuntimeError("mismatch probe update 0 was not collected")

        # The policy is still offloaded and the inference engine awake after
        # each normal weight sync. Re-read first, then temporarily swap residency.
        rescore_started = time.monotonic()
        rescore_rows = await probe._rescore_vllm(trainer, update)
        probe.timing[f"update@{update}/vllm_total_seconds"] = time.monotonic() - rescore_started
        if trainer.colocate_all:
            await trainer.inference_engine_client.sleep()
        try:
            if trainer.colocate_all:
                trainer.policy_model.backload_to_gpu(backload_optimizer=False, backload_model=True)
            try:
                trainer_rows = probe._trainer_scores(trainer, update)
            finally:
                if trainer.colocate_all:
                    trainer.policy_model.offload_to_cpu(offload_optimizer=False, offload_model=True)
        finally:
            if trainer.colocate_all:
                await trainer.inference_engine_client.wake_up()

        scores = rescore_rows + trainer_rows
        if update == 0:
            for row, values in zip(probe.probes, probe.generation_scores, strict=True):
                scores.append(
                    mismatch.ScoreRow(
                        probe_hash=probe.probe_hash,
                        sample_id=row.sample_id,
                        scorer=GENERATION_SCORER,
                        update=0,
                        global_step=trainer.global_step,
                        logprobs=values,
                    )
                )
        probe.metrics[f"update@{update}"] = {
            "token_count": sum(sum(row.loss_mask) for row in probe.probes),
            "sample_count": len(probe.probes),
            "route_bytes": sum(len(row.routed_experts or b"") for row in probe.probes),
            "optimizer_steps": int(trainer.all_metrics.get("policy/policy_update_steps", 0)),
            "step_timings": {
                key: float(value) for key, value in trainer.all_timings.items() if isinstance(value, (int, float))
            },
        }
        reference = {
            row.sample_id: row for row in scores if row.scorer == RESCORE_SCORER and row.cache_mode == CACHE_OFF
        }
        if not reference:
            reference = {
                row.sample_id: row for row in scores if row.scorer == RESCORE_SCORER and row.cache_mode == CACHE_ON
            }
        if len(reference) == len(probe.probes):
            for mode in (NATIVE_MODE, REPEAT_MODE, *probe.spec.extra_trainer_modes):
                scored = {row.sample_id: row for row in scores if row.scorer == TRAINER_SCORER and row.mode == mode}
                if len(scored) != len(probe.probes):
                    continue
                delta = np.concatenate(
                    [
                        (
                            np.asarray(scored[row.sample_id].logprobs, dtype=np.float64)
                            - np.asarray(reference[row.sample_id].logprobs, dtype=np.float64)
                        )[np.asarray(row.loss_mask, dtype=np.bool_)]
                        for row in probe.probes
                    ]
                )
                summary = {
                    "abs_p99": float(np.percentile(np.abs(delta), 99)),
                    "k3": float(np.mean(np.expm1(delta) - delta)),
                    "share_beyond_2x": float(np.mean(np.abs(delta) > math.log(2))),
                }
                probe.metrics[f"update@{update}"][mode] = summary
                for metric, value in summary.items():
                    trainer.all_metrics[f"mismatch_probe/update_{update}/{mode}/{metric}"] = value
        trainer.all_metrics[f"mismatch_probe/update_{update}/tokens"] = probe.metrics[f"update@{update}"]["token_count"]
        trainer.all_metrics[f"mismatch_probe/update_{update}/route_bytes"] = probe.metrics[f"update@{update}"][
            "route_bytes"
        ]
        trainer.all_metrics[f"mismatch_probe/update_{update}/collated_bytes"] = probe.batch_layout.collated_bytes
        trainer.all_metrics[f"mismatch_probe/update_{update}/collated_route_bytes"] = (
            probe.batch_layout.collated_route_bytes
        )
        status = COMPLETE_STATUS if update == probe.updates else BUILDING_STATUS
        if probe.archive is None:
            probe.archive = await asyncio.to_thread(
                MismatchArchive, probe.archive_uri, writer_id=f"probe-{os.getpid()}"
            )
        archive_started = time.monotonic()
        await asyncio.to_thread(
            probe.archive.write,
            probes=probe.probes if update == 0 else None,
            scores=scores,
        )
        probe.timing[f"update@{update}/archive_seconds"] = time.monotonic() - archive_started
        probe.timing[f"update@{update}/total_seconds"] = time.monotonic() - started
        trainer.all_timings[f"mismatch_probe_update_{update}"] = probe.timing[f"update@{update}/total_seconds"]
        await asyncio.to_thread(probe.archive.write, manifest=manifest(probe, trainer, status=status))
        logger.info(
            "Mismatch probe archived update={} samples={} scores={} status={}",
            update,
            len(probe.probes),
            len(scores),
            status,
        )
        return scores
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)

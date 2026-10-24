"""Collect and grade fixed-size mixed-policy groups before training projection."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, field, replace

from taskcompendium.grading_result import GradeResult, Outcome

from skyrl_gym.identity_repair.detector import detect_attribution_claims
from skyrl_gym.identity_repair.facts import FactManifest
from skyrl_gym.identity_repair.grading import ProductionIdentityJudge
from skyrl_gym.identity_repair.reward import IdentityFact, RewardWeights, TurnContext
from skyrl_train.rollouts.repair_branching import BranchChatClient, TeacherRepairSpec, expand_identity_repair_group
from skyrl_train.rollouts.training_record import TrainingRollout, TrainingTransition
from skyrl_train.trajectory_runners.types import TaskRequestBatch


@dataclass(frozen=True)
class IdentityRepairGroups:
    tokenizer: object
    manifest: FactManifest
    judge: ProductionIdentityJudge
    teacher: BranchChatClient
    student: BranchChatClient
    repair_spec: TeacherRepairSpec
    requested_facts: Mapping[str, tuple[tuple[IdentityFact, ...], ...]]
    weights: RewardWeights
    task_criteria: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    async def _paired(self, original: TrainingRollout, session_id: str, template_kwargs: dict) -> list[TrainingRollout]:
        for index, step in enumerate(original.steps):
            sampled_text = self.tokenizer.decode(list(step.turn.response_token_ids), skip_special_tokens=False)
            detections = await asyncio.to_thread(
                detect_attribution_claims, sampled_text, self.manifest, self.judge
            )
            if not detections:
                continue
            # Fix the turn identity before branching; identical claims can occur on several turns.
            return await expand_identity_repair_group(
                original, [replace(detections[0], turn_index=index, sampled_text=sampled_text)], tokenizer=self.tokenizer,
                teacher_client=self.teacher, student_client=self.student,
                spec=replace(self.repair_spec, chat_template_kwargs={**self.repair_spec.chat_template_kwargs, **template_kwargs}),
                session_id=session_id,
            )
        return [original]

    async def grade(self, rollout: TrainingRollout) -> TrainingRollout:
        facts = self.requested_facts[rollout.task_id]
        if len(facts) != len(rollout.steps):
            raise ValueError("Requested-fact manifest must cover every assistant turn")
        criteria = self.task_criteria[rollout.task_id] if self.task_criteria else ("",) * len(facts)
        if len(criteria) != len(facts):
            raise ValueError("Task criteria must cover every assistant turn")
        records = []
        for step, requested, criterion in zip(rollout.steps, facts, criteria, strict=True):
            user = next(message for message in reversed(step.messages[:-1]) if message.get("role") == "user")
            context = TurnContext(str(user["content"]), bool(requested), requested, None,
                                  tuple(step.messages[:-1]), criterion)
            result = await asyncio.to_thread(
                self.judge.grade_answer, step.turn.message.get("content") or "", context, self.manifest, self.weights
            )
            records.append({"reward": asdict(result.reward), "assessment": result.assessment.model_dump(),
                            "task_criteria": criterion})
        if not records:
            raise ValueError("Cannot grade an empty identity rollout")
        reward = sum(record["reward"]["total"] for record in records) / len(records)
        # Remove inherited per-turn rewards so projection consumes this independent task grade.
        steps = tuple(replace(step, transition=TrainingTransition(done=index == len(rollout.steps)-1))
                      for index, step in enumerate(rollout.steps))
        passed = all(
            record["assessment"]["task_completed"]
            and all(score == 1.0 for score in record["assessment"]["fact_correctness"].values())
            and record["assessment"]["unsupported_claims"] == 0
            and not record["assessment"]["inappropriate_refusal"]
            and not record["assessment"]["unsolicited_identity"]
            for record in records
        )
        grade = GradeResult(Outcome.GRADED, reward, passed=passed,
                            score_max=self.weights.factual_correctness + self.weights.responsiveness)
        return replace(rollout, grade=grade, steps=steps,
                       metrics={**rollout.metrics, "identity_reward_records": records})

    async def collect(
        self, request: TaskRequestBatch, generate: Callable[[int], Awaitable[TrainingRollout]]
    ) -> list[TrainingRollout]:
        """Fill two original and two repair/fallback slots without changing request IDs."""
        if request.trajectory_ids is None:
            raise ValueError("Identity groups require explicit trajectory IDs")
        groups: dict[str, list[int]] = {}
        for index, identity in enumerate(request.trajectory_ids):
            groups.setdefault(identity.instance_id, []).append(index)
        for indices in groups.values():
            if len(indices) != 4 or len({request.trajectory_ids[i].repetition_id for i in indices}) != 4:
                raise ValueError("Identity GRPO requires exactly four distinct repetitions per prompt")
            first = request.rows[indices[0]]
            if any(request.rows[i].spec != first.spec for i in indices):
                raise ValueError("A GRPO group must contain identical task definitions")
            if any(request.rows[i].group_grader is not None for i in indices):
                raise ValueError("Identity semantic grading cannot combine with a second group grader")
            if first.spec.task.id not in self.requested_facts:
                raise ValueError("Missing task in the requested-fact manifest")

        async def collect_group(indices):
            originals = await asyncio.gather(*(generate(index) for index in indices[:2]))
            pairs = await asyncio.gather(*(
                self._paired(original, request.trajectory_ids[index].to_string(),
                             request.rows[index].metadata.chat_template_kwargs)
                for index, original in zip(indices[:2], originals, strict=True)
            ))
            outputs = []
            for slot, pair in enumerate(pairs):
                outputs.append((indices[slot], pair[0]))
                fallback_index = indices[slot+2]
                branch = pair[1] if len(pair) == 2 else await generate(fallback_index)
                outputs.append((fallback_index, branch))
            return [(index, await self.grade(rollout)) for index, rollout in outputs]

        collected = await asyncio.gather(*(collect_group(indices) for indices in groups.values()))
        indexed = {index: rollout for group in collected for index, rollout in group}
        return [indexed[index] for index in range(len(request.rows))]

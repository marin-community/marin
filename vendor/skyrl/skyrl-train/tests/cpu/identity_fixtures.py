from omegaconf import OmegaConf
from rolloutengine.models import ModelTurn
from taskcompendium.grading_result import GradeResult, Outcome
from skyrl_train.rollouts.training_record import TrainingRollout, TrainingStep, TrainingTransition


class Tokenizer:
    eos_token_id = 99


def config():
    return OmegaConf.create({"apply_overlong_filtering": False, "sampling_params": {"logprobs": True}})


def rollout(response_ids, reward, *, metadata=None, metrics=None, grade=None, token_credit=None):
    score = float(sum(reward) if isinstance(reward, list) else reward)
    turn = ModelTurn(
        message={"role": "assistant", "content": "answer"},
        prompt_token_ids=(1, 2),
        response_token_ids=tuple(response_ids),
        logprobs=(-0.1,) * len(response_ids),
        stop_reason="stop",
        metadata=metadata or {},
    )
    grade = grade or GradeResult(Outcome.GRADED, score)
    transition = TrainingTransition(
        done=True,
        reward=score,
        token_rewards=tuple(reward) if isinstance(reward, list) else None,
        token_credit=token_credit,
        grade=grade,
    )
    return TrainingRollout(
        task_id="task",
        messages=(turn.message,),
        prompt_token_ids=turn.prompt_token_ids,
        response_token_ids=turn.response_token_ids,
        loss_mask=(1,) * len(response_ids),
        logprobs=turn.logprobs,
        grade=grade,
        stop_reason="stop",
        steps=(TrainingStep(turn, transition, len(response_ids) - 1, (turn.message,)),),
        metrics=metrics or {},
    )

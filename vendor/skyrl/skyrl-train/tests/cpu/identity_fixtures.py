from skyrl_train.rollouts.repair_branching import TurnContinuation
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


class CharTokenizer:
    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(map(chr, token_ids))


def tokens(text):
    return tuple(map(ord, text))


def original_rollout(texts):
    messages = [{"role": "user", "content": "introduce yourself"}]
    prompt = tokens("USER: introduce yourself\nASSISTANT: ")
    ids, masks, probs, steps = [], [], [], []
    for index, text in enumerate(texts):
        if index:
            messages.append({"role": "user", "content": f"follow up {index}"})
            observation = tokens(f"\nUSER: follow up {index}\nASSISTANT: ")
            ids.extend(observation)
            masks.extend([0] * len(observation))
            probs.extend([0.0] * len(observation))
        response = tokens(text)
        turn = ModelTurn(
            message={"role": "assistant", "content": text},
            prompt_token_ids=prompt + tuple(ids),
            response_token_ids=response,
            logprobs=(-0.1,) * len(response),
            text=text,
            stop_reason="stop",
        )
        messages.append(turn.message)
        ids.extend(response)
        masks.extend([1] * len(response))
        probs.extend([-0.1] * len(response))
        steps.append(
            TrainingStep(turn, TrainingTransition(done=index == len(texts) - 1), len(ids) - 1, tuple(messages))
        )
    return TrainingRollout(
        "identity",
        tuple(messages),
        prompt,
        tuple(ids),
        tuple(masks),
        tuple(probs),
        GradeResult(Outcome.GRADED, 0.0),
        "stop",
        tuple(steps),
    )


class FakeClient:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.histories = []
        self.calls = []

    async def continue_turn(self, messages, *, prefix_token_ids, continue_final_message, **kwargs):
        self.calls.append(kwargs)
        self.histories.append(tuple(dict(message) for message in messages))
        answer = next(self.answers)
        if isinstance(answer, Exception):
            raise answer
        prompt = prefix_token_ids
        if not continue_final_message:
            prompt += tokens("\nUSER: " + messages[-1]["content"] + "\nASSISTANT: ")
        return TurnContinuation(
            prompt, tokens(answer), (-0.5,) * len(answer), answer, "stop", {"role": "assistant", "content": answer}
        )

from marin.post_training.environments.olym_math_env import OlymMathEnv
from marin.post_training.load_environments import load_environment_from_spec


def test_load_environment_from_spec():
    env = load_environment_from_spec("olym_math:difficulty=hard", tokenizer=None)
    assert isinstance(env, OlymMathEnv), "Loaded environment should be an instance of OlymMathEnv"

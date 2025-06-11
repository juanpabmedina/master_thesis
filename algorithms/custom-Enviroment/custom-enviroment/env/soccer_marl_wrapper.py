from ray.rllib.env import PettingZooEnv
from pettingzoo.utils import from_parallel
from ray.tune.registry import register_env
from soccer_environment2 import parallel_env

def env_creator(config):
    raw_parallel_env = parallel_env()
    aec_env = from_parallel(raw_parallel_env)
    return PettingZooEnv(aec_env)

register_env("soccer_marl", env_creator)

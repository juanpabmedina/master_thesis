from ray.rllib.env.multi_agent_env import MultiAgentEnv
from marllib.envs.base_env import ENV_REGISTRY
from gym.spaces import Dict as GymDict, Box
from soccer_environment import parallel_env
import numpy as np

# 1. Scenario registry
REGISTRY = {}
REGISTRY["soccer"] = parallel_env  # Needed for MARLlib to resolve map_name

# 2. Policy mapping dict
policy_mapping_dict = {
    "soccer": {
        "description": "Soccer PettingZoo Env",
        "team_prefix": ("team_0_", "team_1_"),  # Adjust to your naming scheme
        "all_agents_one_policy": True,
        "one_agent_one_policy": False,
    }
}



class SoccerMARLlibEnv(MultiAgentEnv):
    def __init__(self, env_config):
        # Create your env from your parallel_env factory
        self.max_cycles = env_config.get("max_cycles", 500)  # default to 100 steps
        self.env = parallel_env()
        self.agents = self.env.possible_agents
        self.num_agents = len(self.agents)

        sample_obs = self.env.observation_space(self.agents[0])
        self.observation_space = GymDict({
            "obs": Box(low=sample_obs.low, high=sample_obs.high, shape=sample_obs.shape, dtype=np.float32)
        })
        self.action_space = self.env.action_space(self.agents[0])

    def reset(self):
        self.current_step = 0
        obs = self.env.reset()
        return {agent: {"obs": obs[agent]} for agent in self.agents}

    def step(self, action_dict):
        self.current_step += 1
        actions = [action_dict[agent] for agent in self.agents]
        obs, rewards, dones, infos = self.env.step(actions)
        obs = {agent: {"obs": obs[agent]} for agent in self.agents}
        rewards = {agent: rewards[agent] for agent in self.agents}

        # If internal step limit reached, end the episode
        done_flag = all(dones.values()) or self.current_step >= self.max_cycles
        dones = {agent: done_flag for agent in self.agents}
        dones["__all__"] = done_flag
        
        return obs, rewards, dones, infos

    def get_env_info(self):
        return {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": 100,
            "policy_mapping_info": policy_mapping_dict
        }

    def render(self, mode='human'):
        return self.env.render()


# Register the env
ENV_REGISTRY["soccer_marl"] = SoccerMARLlibEnv

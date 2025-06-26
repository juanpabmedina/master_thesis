from ray.rllib.env.multi_agent_env import MultiAgentEnv
from marllib.envs.base_env import ENV_REGISTRY
from gym.spaces import Dict as GymDict, Box
from ma_energy_market_env import parallel_env
import numpy as np

# 1. Scenario registry
REGISTRY = {}
REGISTRY["energy_market"] = parallel_env  # Updated name to be more descriptive

# 2. Policy mapping dict - Updated to handle dynamic agents and different sharing modes
def create_policy_mapping_dict(num_generators, num_consumers, policy_mode="shared_within_type"):
    """
    Create policy mapping dictionary based on agent configuration
    
    Args:
        num_generators: Number of generator agents
        num_consumers: Number of consumer agents  
        policy_mode: One of:
            - "all_shared": All agents share one policy
            - "shared_within_type": Generators share one policy, consumers share another
            - "individual": Each agent has its own policy
    """
    generator_agents = [f"generator_{i}" for i in range(num_generators)]
    consumer_agents = [f"consumer_{i}" for i in range(num_consumers)]
    
    if policy_mode == "all_shared":
        return {
            "energy_market": {
                "description": "Energy Market Multi-Agent Environment",
                "team_prefix": ("agent_",),  # All agents in one team
                "all_agents_one_policy": True,   # All agents share one policy
                "one_agent_one_policy": False,   # Not individual policies
                "generator_agents": generator_agents,
                "consumer_agents": consumer_agents,
            }
        }
    elif policy_mode == "shared_within_type":
        return {
            "energy_market": {
                "description": "Energy Market Multi-Agent Environment", 
                "team_prefix": ("generator_", "consumer_"),  # Two teams: generators and consumers
                "all_agents_one_policy": False,  # Not all agents share one policy
                "one_agent_one_policy": False,   # Not individual policies
                "generator_agents": generator_agents,
                "consumer_agents": consumer_agents,
            }
        }
    elif policy_mode == "individual":
        return {
            "energy_market": {
                "description": "Energy Market Multi-Agent Environment",
                "team_prefix": tuple([f"{agent}_" for agent in generator_agents + consumer_agents]),
                "all_agents_one_policy": False,  # Not all agents share one policy
                "one_agent_one_policy": True,    # Each agent has individual policy
                "generator_agents": generator_agents,
                "consumer_agents": consumer_agents,
            }
        }
    else:
        raise ValueError(f"Unknown policy_mode: {policy_mode}")


# 3. Global policy mapping dict for different configurations
policy_mapping_dict = {
    "energy_market": {
        "description": "Energy Market Multi-Agent Environment",
        "team_prefix": ("agent_",),  # Default: all agents in one team
        "all_agents_one_policy": True,   # Default: all agents share one policy
        "one_agent_one_policy": False,   # Default: not individual policies
    }
}


class EnergyMARLlibEnv(MultiAgentEnv):
    def __init__(self, env_config):
        # Extract environment configuration
        self.max_cycles = env_config.get("max_cycles", 100)
        max_gen_power = env_config.get("max_gen_power", 1.0)
        min_gen_power = env_config.get("min_gen_power", 0.1)
        max_con_price = env_config.get("max_con_price", 1.0)
        min_con_price = env_config.get("min_con_price", 0.1)
        ind_max_gen_power = env_config.get("ind_max_gen_power", 1)
        ind_min_gen_power = env_config.get("ind_min_gen_power", 0.01)
        ind_max_con_price = env_config.get("ind_max_con_price", 1)
        ind_min_con_price = env_config.get("ind_min_con_price", 0.01)
        max_steps = env_config.get("max_steps", 100)  # default to 100 steps
        
        # Extract agent configuration - NEW: Support for dynamic agent numbers
        self.num_generators = env_config.get("num_generators", 1)
        self.num_consumers = env_config.get("num_consumers", 1)
        
        # Policy sharing mode - NEW: Configure how policies are shared
        self.policy_mode = env_config.get("policy_mode", "all_shared")  # Default: all agents share policy

        # Create the parallel environment with agent configuration
        self.env = parallel_env(
            max_gen_power=max_gen_power, 
            min_gen_power=min_gen_power, 
            max_con_price=max_con_price, 
            min_con_price=min_con_price,
            ind_max_gen_power=ind_max_gen_power,
            ind_min_gen_power=ind_min_gen_power,
            ind_max_con_price=ind_max_con_price,
            ind_min_con_price=ind_min_con_price,
            max_steps=max_steps,
            num_generators=self.num_generators,  # NEW: Pass agent numbers
            num_consumers=self.num_consumers     # NEW: Pass agent numbers
        )
        
        # Get agents from the environment after creation
        self.agents = self.env.possible_agents
        self.num_agents = len(self.agents)
        
        # Create policy mapping info based on actual agent configuration
        self.policy_mapping_info = create_policy_mapping_dict(
            self.num_generators, self.num_consumers, self.policy_mode
        )

        # Set up observation and action spaces
        # Get a sample observation from the first agent to determine space
        if self.agents:
            sample_obs_space = self.env.observation_spaces[self.agents[0]]
            self.observation_space = GymDict({
                "obs": Box(
                    low=sample_obs_space.low, 
                    high=sample_obs_space.high, 
                    shape=sample_obs_space.shape, 
                    dtype=np.float32
                )
            })
            self.action_space = self.env.action_spaces[self.agents[0]]
        else:
            # Fallback if no agents (shouldn't happen with valid config)
            self.observation_space = GymDict({"obs": Box(low=-np.inf, high=np.inf, shape=(2,))})
            self.action_space = Box(low=0, high=2, shape=(1,), dtype=int)

    def reset(self):
        """Reset the environment and return initial observations"""
        self.current_step = 0
        obs_dict = self.env.reset()
        
        # Update agents list in case it changed (though it shouldn't in this env)
        self.agents = self.env.agents if hasattr(self.env, 'agents') else self.env.possible_agents
        
        # Format observations for MARLlib
        formatted_obs = {}
        for agent in self.agents:
            if agent in obs_dict:
                formatted_obs[agent] = {"obs": obs_dict[agent]}
            
        return formatted_obs

    def step(self, action_dict):
        """Execute one step in the environment"""
        self.current_step += 1
        
        # Extract actions for active agents only
        actions = {}
        for agent in self.agents:
            if agent in action_dict:
                actions[agent] = action_dict[agent]
        
        # Step the environment
        obs_dict, rewards, dones, infos = self.env.step(actions)
        
        # Format observations
        formatted_obs = {}
        for agent in self.agents:
            if agent in obs_dict:
                formatted_obs[agent] = {"obs": obs_dict[agent]}
        
        # Handle rewards - ensure all active agents have rewards
        formatted_rewards = {}
        for agent in self.agents:
            formatted_rewards[agent] = rewards.get(agent, 0.0)
        
        # Handle termination
        done_flag = all(dones.values()) or self.current_step >= self.max_cycles
        formatted_dones = {agent: done_flag for agent in self.agents}
        formatted_dones["__all__"] = done_flag
        
        # Format infos
        formatted_infos = {}
        for agent in self.agents:
            formatted_infos[agent] = infos.get(agent, {})
        
        return formatted_obs, formatted_rewards, formatted_dones, formatted_infos

    def get_env_info(self):
        """Return environment information for MARLlib"""
        return {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": self.max_cycles,
            "policy_mapping_info": self.policy_mapping_info,
            # Additional info about agent types
            "agent_types": {
                "generators": [f"generator_{i}" for i in range(self.num_generators)],
                "consumers": [f"consumer_{i}" for i in range(self.num_consumers)]
            }
        }

    def render(self, mode='human'):
        """Render the environment"""
        return self.env.render(mode=mode)

    def close(self):
        """Clean up environment resources"""
        if hasattr(self.env, 'close'):
            self.env.close()


# Register the environment with MARLlib
ENV_REGISTRY["energy"] = EnergyMARLlibEnv
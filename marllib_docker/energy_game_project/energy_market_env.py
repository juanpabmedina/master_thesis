from gym.spaces import Discrete, Box
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers, from_parallel
from typing import Optional
import numpy as np
import random
import matplotlib.pyplot as plt


# def env():
#     """Entry point for creating the environment with standard PettingZoo wrappers"""
#     raw = parallel_env()
#     return wrappers.OrderEnforcingWrapper(
#         wrappers.AssertOutOfBoundsWrapper(
#             wrappers.CaptureStdoutWrapper(from_parallel(raw))
#         )
#     )


class parallel_env(ParallelEnv):
    metadata = {'render.modes': ['human'], "name": "energy_market_v1"}

    def __init__(self, max_gen_power, min_gen_power, 
                 max_con_price, min_con_price, 
                 profit_threshold, max_steps):
        """
        Initialize the Energy Market environment.
        
        Args:
            max_gen_power: Maximum generator power output
            min_gen_power: Minimum generator power output
            max_con_price: Maximum consumer price
            min_con_price: Minimum consumer price
            profit_threshold: Profit threshold for winning condition
            max_steps: Maximum number of steps before episode ends
        """
        # Cost function parameters
        self.a = 0.2
        self.b = 2
        self.c = 0

        # Environment bounds
        self.max_gen_power = max_gen_power
        self.min_gen_power = min_gen_power
        self.max_con_price = max_con_price
        self.min_con_price = min_con_price
        self.profit_threshold = profit_threshold
        self.max_steps = max_steps

        # Variable to plot 
        self.cum_gen_power = []
        self.cum_con_price = []

        # Agents
        self.possible_agents = ["generator", "consumer"]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        
        # Action spaces: 3 discrete actions (decrease, stay, increase)
        self.action_spaces = {agent: Discrete(3) for agent in self.possible_agents}
        
        # Observation space: [own_value, opponent_value, own_profit, opponent_profit, step]
        self.observation_spaces = {
            agent: Box(
                low=np.array([-10.0, -10.0,]), 
                high=np.array([10.0, 10.0]), 
                shape=(2,), 
                dtype=np.float32
            ) 
            for agent in self.possible_agents
        }

        # Action mappings
        self.action_step = 0.1
        self.action_to_delta = {
            0: -self.action_step,  # Decrease
            1: 0.0,                # Stay
            2: self.action_step    # Increase
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment to initial state"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.agents = self.possible_agents[:]
        
        # Initialize random positions within bounds
        self._generator_power = np.random.uniform(self.min_gen_power, self.max_gen_power)
        self._consumer_price = np.random.uniform(self.min_con_price, self.max_con_price)
        
        # Initialize step counter
        self.step_count = 0
        
        # Calculate initial profits
        self._update_profits()
        
        # Get initial observations
        observations = self.observe()

        # Variable to plot 
        self.cum_gen_power = []
        self.cum_con_price = []
        
        return observations

    def observe(self):
        """Generate observations for all agents"""
        generator_obs = np.array([
            self._generator_power,
            self._consumer_price
        ], dtype=np.float32)
        
        consumer_obs = np.array([
            self._consumer_price,
            self._generator_power
        ], dtype=np.float32)
        
        observations = {
            "generator": generator_obs,
            "consumer": consumer_obs
        }
        
        return observations

    def step(self, actions):
        """Execute one step of the environment"""
        # Handle empty actions
        if not actions:
            self.agents = []
            return {}, {}, {}, {}

        # Format actions if needed
        formatted_actions = self._format_actions(actions)
        
        # Apply actions simultaneously
        if "generator" in formatted_actions and "generator" in self.agents:
            delta = self.action_to_delta[formatted_actions["generator"]]
            self._generator_power = np.clip(
                self._generator_power + delta, 
                self.min_gen_power, 
                self.max_gen_power
            )
        
        if "consumer" in formatted_actions and "consumer" in self.agents:
            delta = self.action_to_delta[formatted_actions["consumer"]]
            self._consumer_price = np.clip(
                self._consumer_price + delta, 
                self.min_con_price, 
                self.max_con_price
            )
        
        # Update step counter
        self.step_count += 1
        
        # Calculate new profits
        self._update_profits()
        
        # Calculate rewards and check for termination
        rewards = self._calculate_rewards()
        dones = self._check_termination()
        
        # Get new observations
        observations = self.observe()
        
        # Info dictionary
        infos = {
            agent: {
                "generator_power": self._generator_power,
                "consumer_price": self._consumer_price,
                "generator_profit": self.gen_profit,
                "consumer_profit": self.con_profit,
                "step": self.step_count
            } for agent in self.agents
        }
        
        # Remove agents if game is done
        if any(dones.values()):
            self.agents = []
        
        return observations, rewards, dones, infos

    def _update_profits(self):
        """Calculate profits for both agents"""
        # Generator cost function: Hg = a * P^2 + b * P + c
        Hg = self.a * self._generator_power**2 + self.b * self._generator_power + self.c
        
        # Generator profit: P * price - cost
        self.gen_profit = self._generator_power * self._consumer_price - Hg
        
        # Consumer utility: P * (1/ln(1+price))
        # Adding small epsilon to avoid division by zero
        epsilon = 1e-8
        self.con_profit = self._generator_power * (1 / np.log(1 + self._consumer_price + epsilon))

    def _calculate_rewards(self):
        """Calculate rewards for both agents"""
        rewards = {"generator": 0.0, "consumer": 0.0}
        
        # Check if either agent reached profit threshold
        # if self.gen_profit > self.profit_threshold:
        #     rewards["generator"] = 100.0
        #     rewards["consumer"] = -10.0
        # elif self.con_profit > self.profit_threshold:
        #     rewards["generator"] = -10.0
        #     rewards["consumer"] = 100.0
        # else:
        #     # Small negative reward to encourage efficiency
        #     rewards["generator"] = -0.01
        #     rewards["consumer"] = -0.01

        rewards["generator"] = self.gen_profit
        rewards["consumer"] = self.con_profit 
        
        
        return rewards

    def _check_termination(self):
        """Check if the episode should terminate"""
        # Terminate if either agent reaches profit threshold or max steps reached
        profit_achieved = (self.gen_profit > self.profit_threshold or 
                          self.con_profit > self.profit_threshold)
        max_steps_reached = self.step_count >= self.max_steps
        
        terminated = max_steps_reached
        
        return {"generator": terminated, "consumer": terminated}

    def _format_actions(self, actions):
        """Convert different action formats to expected dict format"""
        # If actions is already a dict with agent keys, return as is
        if isinstance(actions, dict) and all(agent in actions for agent in self.agents):
            return actions
        
        # If actions is a list, map to agents by index
        if isinstance(actions, (list, tuple)):
            formatted = {}
            for i, agent in enumerate(self.agents):
                if i < len(actions):
                    formatted[agent] = actions[i]
            return formatted
        
        # If actions is a dict with policy IDs
        if isinstance(actions, dict):
            formatted = {}
            policy_keys = list(actions.keys())
            
            if len(policy_keys) == len(self.agents):
                for i, agent in enumerate(self.agents):
                    if i < len(policy_keys):
                        formatted[agent] = actions[policy_keys[i]]
            return formatted
        
        # Single agent case
        if len(self.agents) == 1:
            return {self.agents[0]: actions}
        
        print(f"Warning: Could not format actions {actions} of type {type(actions)}")
        return {}

    def render(self, mode="human"):
        """Render the environment state"""
        if mode == "human":
            self.print_state()

    def print_state(self):
        self.cum_gen_power.append(self._generator_power)
        self.cum_con_price.append(self._consumer_price)

        print("=========== CURRENT STATE ===========")
        print("Generator: ", self._generator_power)
        print("Consumer: ", self._consumer_price)
        print("Step: ", self.step_count)
        print(f"Min: {self.min_gen_power, self.min_con_price} - Max: {self.max_gen_power, self.max_con_price}")
        print("=====================================")

        if self.step_count == 25:
            """Print current state of the energy market"""
            # Create and save the plot
            plt.figure(figsize=(10, 6))
            plt.plot(self.cum_gen_power, label='Generator Power (P)', marker='o')
            plt.plot(self.cum_con_price, label='Consumer Price ($)', marker='s')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.title('Generator Power and Consumer Price Over Time')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('market_dynamics.png')  # Save figure as PNG
            plt.close()


    def close(self):
        """Clean up environment resources"""
        pass
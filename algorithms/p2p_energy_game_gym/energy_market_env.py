from typing import Optional
import numpy as np
import gymnasium as gym


class EnergyMarketEnv(gym.Env):

    def __init__(self, max_gen_power: float = 5, min_gen_power: float = 0.1, 
                 max_con_price: float = 5, min_con_price: float = 0.1, 
                 threshold: float = 0):

        self.a = 0.2
        self.b = 2
        self.c = 0 

        self.threshold = threshold

        self.max_gen_power = max_gen_power
        self.min_gen_power = min_gen_power
        self.max_con_price = max_con_price
        self.min_con_price = min_con_price

        # Define the agent and target location; randomly chosen in `reset` and updated in `step`
        self._generator_power = -100
        self._consumer_price = -100
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`-1}^2
        self.observation_space = gym.spaces.Dict(
            {
                "generator": gym.spaces.Box(min_gen_power, max_gen_power, shape=(), dtype=float),
                "consumer": gym.spaces.Box(min_con_price, max_con_price, shape=(), dtype=float),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.generator_action_space = gym.spaces.Discrete(3)
        # Dictionary maps the abstract actions to the directions on the grid
        self._generator_action_to_direction = {
            0: -0.1,
            1: 0,
            2: 0.1
        }

        self.consumer_action_space = gym.spaces.Discrete(3)
        # Dictionary maps the abstract actions to the directions on the grid
        self._consuemer_action_to_direction = {
            0: -0.1,
            1: 0,
            2: 0.1
        }

    def _get_obs(self):
        return {"generator": self._generator_power, "consumer": self._consumer_price}
    
    def _get_profit(self):

        Hg = self.a * self._generator_power**2 + self.b * self._generator_power + self.c
        self.gen_profit = self._generator_power * self._consumer_price - Hg
        self.con_profit = self._generator_power * (1/np.log(1+self._consumer_price))
        return {
            "generator": self.gen_profit,
            "consumer": self.con_profit
        }
    
    def _get_directions(self, actions):
        return {
            'generator': self._generator_action_to_direction[actions['generator']],
            'consumer': self._consuemer_action_to_direction[actions['consumer']]
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._generator_power = self.np_random.uniform(low=self.min_gen_power, high=self.max_gen_power)
        self._consumer_price = self.np_random.uniform(low=self.min_con_price, high=self.max_con_price) 

        observation = self._get_obs()
        profit = self._get_profit()

        return observation, profit
    

    def step(self, actions, agent_id):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        directions = self._get_directions(actions)
        # We use `np.clip` to make sure we don't leave the grid bounds
        self._generator_power = np.clip(
            self._generator_power + directions['generator'], self.min_gen_power, self.max_gen_power
        )

        self._consumer_price = np.clip(
            self._consumer_price + directions['consumer'], self.min_con_price, self.max_con_price
        )

        # An environment is completed if and only if the agent has reached the target
        profit = self._get_profit()
        terminated = profit[agent_id] > self.threshold
        if terminated:
            if agent_id == 'generator':
                reward = {
                    'generator': 1,
                    'consumer': -1
                } 
            if agent_id == 'consumer':
                reward = {
                    'generator': -1,
                    'consumer': 1
                } 
        else:
            reward ={
                'generator': 0,
                'consumer': 0
            }

        next_state = self._get_obs()
        profit = self._get_profit()

        return next_state, reward, terminated, profit
    

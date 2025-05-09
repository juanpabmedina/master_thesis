import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class GymEnergyMarketEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, a=0.1, b=2, c=0, 
                 min_power=0.1, max_power=0.5,
                 min_price=1.0, max_price=5.0, 
                 threshold=0.0):
        super(GymEnergyMarketEnv, self).__init__()

        self.a = a
        self.b = b
        self.c = c

        self.min_power = min_power
        self.max_power = max_power
        self.min_price = min_price
        self.max_price = max_price
        self.threshold = threshold

        # Two agents: generator and consumer
        self.agents = ['generator', 'consumer']

        # Action: Generator adjusts power, Consumer adjusts price
        self.action_space = spaces.Dict({
            'generator': spaces.Box(low=-0.1, high=0.1, shape=(1,), dtype=np.float32),
            'consumer': spaces.Box(low=-0.1, high=0.1, shape=(1,), dtype=np.float32)
        })

        # Observation: (gen_power, con_price)
        self.observation_space = spaces.Dict({
            'generator': spaces.Box(low=np.array([min_power, min_price]), 
                                    high=np.array([max_power, max_price]), dtype=np.float32),
            'consumer': spaces.Box(low=np.array([min_power, min_price]), 
                                   high=np.array([max_power, max_price]), dtype=np.float32)
        })

        self.state = None
        self.reset()

    def reset(self):
        self.gen_power = round(random.uniform(self.min_power, self.max_power), 2)
        self.con_price = round(random.uniform(self.min_price, self.max_price), 2)
        self.state = (self.gen_power, self.con_price)
        obs = {
            'generator': np.array(self.state, dtype=np.float32),
            'consumer': np.array(self.state, dtype=np.float32)
        }
        return obs

    def step(self, actions):
        gen_action = float(actions['generator'][0])
        con_action = float(actions['consumer'][0])

        self.gen_power = np.clip(self.gen_power + gen_action, self.min_power, self.max_power)
        self.con_price = np.clip(self.con_price + con_action, self.min_price, self.max_price)

        Hg = self.a * self.gen_power**2 + self.b * self.gen_power + self.c
        gen_profit = self.gen_power * self.con_price - Hg
        con_profit = self.gen_power * (1/np.log(1 + self.con_price))

        # Reward logic: stop if either reaches threshold
        if gen_profit >= self.threshold:
            rewards = {'generator': 1.0, 'consumer': -1.0}
            done = True
        elif con_profit >= self.threshold:
            rewards = {'generator': -1.0, 'consumer': 1.0}
            done = True
        else:
            rewards = {'generator': 0.0, 'consumer': 0.0}
            done = False

        next_state = np.array((self.gen_power, self.con_price), dtype=np.float32)
        obs = {
            'generator': next_state,
            'consumer': next_state
        }

        info = {}

        return obs, rewards, done, info

    def render(self, mode="human"):
        print(f"Power: {self.gen_power}, Price: {self.con_price}")

    def close(self):
        pass

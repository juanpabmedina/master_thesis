import random
import numpy as np
from copy import copy

from pettingzoo import ParallelEnv
from gym.spaces import Discrete, MultiDiscrete


class CustomActionMaskedEnvironment(ParallelEnv):
    metadata = {"name": "custom_environment_v0"}

    def __init__(self):
        self.escape_y = None
        self.escape_x = None
        self.guard_y = None
        self.guard_x = None
        self.prisoner_y = None
        self.prisoner_x = None
        self.timestep = None
        self.possible_agents = ["prisoner", "guard"]
        self.agents = copy(self.possible_agents)

        self.observation_spaces = {
            agent: MultiDiscrete([49, 49, 49]) for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: Discrete(4) for agent in self.possible_agents
        }

    def reset(self):
        self.agents = copy(self.possible_agents)
        self.timestep = 0

        self.prisoner_x = 0
        self.prisoner_y = 0
        self.guard_x = 7
        self.guard_y = 7
        self.escape_x = random.randint(2, 5)
        self.escape_y = random.randint(2, 5)

        observation = (
            self.prisoner_x + 7 * self.prisoner_y,
            self.guard_x + 7 * self.guard_y,
            self.escape_x + 7 * self.escape_y,
        )

        observations = {
            "prisoner": observation,
            "guard": observation,
        }
        infos = {a: {} for a in self.agents}

        return observations, infos

    def step(self, actions):
        prisoner_action = actions["prisoner"]
        guard_action = actions["guard"]

        if prisoner_action == 0 and self.prisoner_x > 0:
            self.prisoner_x -= 1
        elif prisoner_action == 1 and self.prisoner_x < 6:
            self.prisoner_x += 1
        elif prisoner_action == 2 and self.prisoner_y > 0:
            self.prisoner_y -= 1
        elif prisoner_action == 3 and self.prisoner_y < 6:
            self.prisoner_y += 1

        if guard_action == 0 and self.guard_x > 0:
            self.guard_x -= 1
        elif guard_action == 1 and self.guard_x < 6:
            self.guard_x += 1
        elif guard_action == 2 and self.guard_y > 0:
            self.guard_y -= 1
        elif guard_action == 3 and self.guard_y < 6:
            self.guard_y += 1

        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}
        if self.prisoner_x == self.guard_x and self.prisoner_y == self.guard_y:
            rewards = {"prisoner": -1, "guard": 1}
            terminations = {a: True for a in self.agents}
            self.agents = []
        elif self.prisoner_x == self.escape_x and self.prisoner_y == self.escape_y:
            rewards = {"prisoner": 1, "guard": -1}
            terminations = {a: True for a in self.agents}
            self.agents = []

        if self.timestep > 100:
            rewards = {"prisoner": 0, "guard": 0}
            terminations = {"prisoner": True, "guard": True}
            self.agents = []

        self.timestep += 1

        observation = (
            self.prisoner_x + 7 * self.prisoner_y,
            self.guard_x + 7 * self.guard_y,
            self.escape_x + 7 * self.escape_y,
        )
        observations = {
            "prisoner": observation,
            "guard": observation,
        }
        infos = {"prisoner": {}, "guard": {}}

        return observations, rewards, terminations, infos

    def render(self):
        grid = np.full((8, 8), ".", dtype=object)
        grid[self.prisoner_y, self.prisoner_x] = "P"
        grid[self.guard_y, self.guard_x] = "G"
        grid[self.escape_y, self.escape_x] = "E"
        print(grid)

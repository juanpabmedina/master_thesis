from gym.spaces import Discrete, Box
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers, from_parallel
from IPython.display import clear_output

import numpy as np
import random
import time


# Constants
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']

ACTION_TO_DELTA = {
    0: (-1, 0),  # UP
    1: (1, 0),   # DOWN
    2: (0, -1),  # LEFT
    3: (0, 1),   # RIGHT
    4: (0, 0)    # STAY
}


def env():
    raw = parallel_env()
    return wrappers.OrderEnforcingWrapper(
        wrappers.AssertOutOfBoundsWrapper(
            wrappers.CaptureStdoutWrapper(from_parallel(raw))
        )
    )


class parallel_env(ParallelEnv):
    metadata = {'render.modes': ['human'], "name": "soccer_v1"}

    def __init__(self):
        '''
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        '''
        # Environment dimensions
        self.rows = 4
        self.cols = 7  # 5 playable columns + 2 goal columns

        # Agents
        self.possible_agents = ["player_A", "player_B"]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        
        # Action space: 5 discrete actions (UP, DOWN, LEFT, RIGHT, STAY)
        self.action_spaces = {agent: Discrete(5) for agent in self.possible_agents}
        
        # Observation space: positions of both players + ball possession
        # [player_A_row, player_A_col, player_B_row, player_B_col, ball_possession]
        # ball_possession: 0 for player_A, 1 for player_B
        self.observation_spaces = {
            agent: Box(low=0, high=6, shape=(5,), dtype=np.float32) 
            for agent in self.possible_agents
        }

    def render(self, mode="human"):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if mode == "human":
            self.print_state()


    def close(self):
        pass

    def reset(self):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - dones
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        """
        self.agents = self.possible_agents[:]
        
        # Game state
        self.player_positions = {self.agents[0]: (1, 4), self.agents[1]: (2, 2)}
        self.ball_possession = random.choice(self.agents)
        
        # Store actions for simultaneous execution
        self.state = {agent: None for agent in self.agents}
        

        self.observe()  # this sets self.observations

        return self.observations  # now returning real observations

    def observe(self):
        pos_A = self.player_positions[self.agents[0]]
        pos_B = self.player_positions[self.agents[1]]
        ball_poss = 0 if self.ball_possession == self.agents[0] else 1

        self.observations = {
            self.agents[0]: np.array([pos_A[0], pos_A[1], pos_B[0], pos_B[1], ball_poss], dtype=np.float32),
            self.agents[1]: np.array([pos_B[0], pos_B[1], pos_A[0], pos_A[1], ball_poss], dtype=np.float32),
        }
        return self.observations


    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - dones
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}

        # Handle different action formats from MARLlib/RLlib
        formatted_actions = self._format_actions(actions)
        
        # Debug print to see what we're receiving
        # print(f"Original actions: {actions}")
        # print(f"Formatted actions: {formatted_actions}")
        # print(f"Current agents: {self.agents}")

        # Agents positioning aleatory for game
        agent_order = self.possible_agents.copy()
        random.shuffle(agent_order)

        new_positions = self.player_positions.copy()

        for agent in agent_order:
            if agent not in self.agents:  # Skip if agent is not active
                continue
                
            other_agent = self.agents[1] if agent == self.agents[0] else self.agents[0]
            
            # Use formatted actions
            if agent in formatted_actions:
                new_pos = self._move(new_positions[agent], formatted_actions[agent])
                if new_pos == new_positions[other_agent]:
                    self.ball_possession = other_agent
                    continue
                new_positions[agent] = new_pos

        self.player_positions = new_positions

        ball_holder = self.ball_possession
        ball_pos = self.player_positions[ball_holder]

        # rewards for all agents are placed in the rewards dictionary to be returned
        rewards = {agent: 0 for agent in self.agents}

        # Goal condition: ball enters the G positions (column 0 or 6) in rows 1 or 2
        if ball_holder == self.agents[0] and ball_pos[1] == 6 and ball_pos[0] in [1, 2]:
            rewards[self.agents[0]] = 1
            rewards[self.agents[1]] = -1
            game_done = True
        elif ball_holder == self.agents[1] and ball_pos[1] == 0 and ball_pos[0] in [1, 2]:
            rewards[self.agents[0]] = -1
            rewards[self.agents[1]] = 1
            game_done = True
        else:
            rewards[self.agents[0]] = 0
            rewards[self.agents[1]] = 0
            game_done = False

        dones = {agent: game_done for agent in self.agents}
        observations = self.observe()

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent: {} for agent in self.agents}

        self.dones = dones  # Optional but good
        if game_done:
            self.agents = []

        return observations, rewards, dones, infos

    def _format_actions(self, actions):
        """
        Convert different action formats to the expected dict format
        """
        # If actions is already a dict with agent keys, return as is
        if isinstance(actions, dict) and all(agent in actions for agent in self.agents):
            return actions
        
        # If actions is a list, map to agents by index (THIS IS YOUR CASE)
        if isinstance(actions, (list, tuple)):
            formatted = {}
            for i, agent in enumerate(self.agents):
                if i < len(actions):
                    formatted[agent] = actions[i]
            return formatted
        
        # If actions is a dict but with policy IDs instead of agent IDs
        if isinstance(actions, dict):
            # Common case: policy names like "policy_0", "policy_1" or just numbers
            formatted = {}
            policy_keys = list(actions.keys())
            
            # Try to map policy keys to agent names
            if len(policy_keys) == len(self.agents):
                for i, agent in enumerate(self.agents):
                    if i < len(policy_keys):
                        formatted[agent] = actions[policy_keys[i]]
            return formatted
        
        # If actions is a single value (single agent case)
        if len(self.agents) == 1:
            return {self.agents[0]: actions}
        
        # If we can't format, return empty dict
        print(f"Warning: Could not format actions {actions} of type {type(actions)}")
        return {}

    def _move(self, position, action):
        """Move a player based on action"""
        if action is None:
            return position
            
        delta = ACTION_TO_DELTA[action]
        new_row = min(max(position[0] + delta[0], 0), self.rows - 1)
        new_col = position[1] + delta[1]

        # Restrict movement in the x corners (positions [0][0] and [3][0], [0][6] and [3][6])
        if new_col < 0 or new_col > 6:
            new_col = position[1]  # Prevent moving outside grid

        if (new_row == 0 or new_row == 3) and (new_col == 0 or new_col == 6):
            return position  # Prevent entering the x corners

        return (new_row, new_col)

    def print_state(self):
        """Print the current state of the game"""
        grid = [[' . ' for _ in range(self.cols)] for _ in range(self.rows)]

        # Add goals and blocked corners
        for r in [1, 2]:
            grid[r][0] = ' G '
            grid[r][6] = ' G '
        for r in [0, 3]:
            grid[r][0] = ' x '
            grid[r][6] = ' x '

        # Add players
        for player, pos in self.player_positions.items():

            if player == self.possible_agents[0]:
                player_short = 'A'
            else:
                player_short = 'B'
            grid[pos[0]][pos[1]] = ' ' + player_short + ('o' if self.ball_possession == player else ' ')

        clear_output(wait=True)
        for row in grid:
            print(''.join(row))
        # print(f"Scores: {self.scores}")
        # print(f"Ball possession: {self.ball_possession}")
        # print(f"Current agent: {self.agent_selection}")
        # print()
        time.sleep(0.5)
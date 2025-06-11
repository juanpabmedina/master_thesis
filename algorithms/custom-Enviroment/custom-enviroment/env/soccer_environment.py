import random
import time
from typing import Dict, Optional, Any
import numpy as np
from gym.spaces import Discrete, Box
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from IPython.display import clear_output


ACTION_TO_DELTA = {
    0: (-1, 0),  # UP
    1: (1, 0),   # DOWN
    2: (0, -1),  # LEFT
    3: (0, 1),   # RIGHT
    4: (0, 0)    # STAY
}

def env():
    """
    The env function wraps the environment in 3 wrappers by default. These
    wrappers contain logic that is common to many pettingzoo environments.
    We recommend you use at least the OrderEnforcingWrapper on your own environment
    to provide sane error messages. You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env = raw_env()
    # env = wrappers.CaptureStdoutWrapper(env)
    # env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class raw_env(AECEnv):
    """
    The metadata holds environment constants. From gym, we inherit the "render.modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """
    metadata = {'render.modes': ['human'], "name": "soccer_v1"}

    def __init__(self):
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        """
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
            agent: Box(low=0, high=6, shape=(5,), dtype=np.int32) 
            for agent in self.possible_agents
        }

    def render(self, mode="human"):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if mode == "human":
            self.print_state()

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # Observation: [player_A_row, player_A_col, player_B_row, player_B_col, ball_possession]
        pos_A = self.player_positions['A']
        pos_B = self.player_positions['B']
        ball_poss = 0 if self.ball_possession == 'A' else 1
        
        observation = np.array([pos_A[0], pos_A[1], pos_B[0], pos_B[1], ball_poss], dtype=np.int32)
        return observation

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
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
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # Game state
        self.player_positions = {'A': (1, 4), 'B': (2, 2)}
        self.ball_possession = random.choice(['A', 'B'])
        self.game_done = False
        self.scores = {'A': 0, 'B': 0}
        
        # Store actions for simultaneous execution
        self.state = {agent: None for agent in self.agents}
        
        # Agent selection
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - dones
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        if self.dones[self.agent_selection]:
            # handles stepping an agent which is already done
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next done agent,  or if there are no more done agents, to the next live agent
            return self._was_done_step(action)

        agent = self.agent_selection

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0

        # stores action of current agent
        self.state[self.agent_selection] = action

        # collect reward if it is the last agent to act
        # Check if both agents have provided actions
        if all(self.state[agent] is not None for agent in self.agents): # This line make that the game state is only actualized when both agents have
            # Execute the game step with both agents' actions
            self._execute_game_step()
            # Reset state for next round
            self.state = {agent: None for agent in self.agents}
        else:
            # no rewards are allocated until both players give an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

    def _execute_game_step(self):
        """Execute one step of the game with both agents' actions"""
        # Convert agent names to game logic names and actions
        actions_dict = {}
        actions_dict['A'] = self.state['player_A']
        actions_dict['B'] = self.state['player_B']
        
        # Execute game logic
        order = ['A', 'B']
        random.shuffle(order)

        new_positions = self.player_positions.copy()
        for game_agent in order:
            other_agent = 'B' if game_agent == 'A' else 'A'
            new_pos = self._move(new_positions[game_agent], actions_dict[game_agent])
            if new_pos == new_positions[other_agent]:
                self.ball_possession = other_agent
                continue
            new_positions[game_agent] = new_pos

        self.player_positions = new_positions

        # Calculate rewards
        reward = {'A': 0, 'B': 0}
        ball_holder = self.ball_possession
        ball_pos = self.player_positions[ball_holder]

        # Goal condition: ball enters the G positions (column 0 or 6) in rows 1 or 2
        if ball_holder == 'A' and ball_pos[1] == 6 and ball_pos[0] in [1, 2]:
            reward['A'] = 1
            reward['B'] = -1
            self.game_done = True
        elif ball_holder == 'B' and ball_pos[1] == 0 and ball_pos[0] in [1, 2]:
            reward['A'] = -1
            reward['B'] = 1
            self.game_done = True

        self.scores['A'] += reward['A']
        self.scores['B'] += reward['B']
        
        # Update PettingZoo rewards
        self.rewards['player_A'] = reward['A']
        self.rewards['player_B'] = reward['B']
        
        # Update dones
        if self.game_done:
            self.dones = {agent: True for agent in self.agents}

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
            grid[pos[0]][pos[1]] = ' ' + player + ('o' if self.ball_possession == player else ' ')

        clear_output(wait=True)
        for row in grid:
            print(''.join(row))
        print(f"Scores: {self.scores}")
        print(f"Ball possession: {self.ball_possession}")
        print(f"Current agent: {self.agent_selection}")
        print()
        time.sleep(0.5)
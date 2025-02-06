import numpy as np
import random
from scipy.optimize import linprog

class QL_Agent:
    def __init__(self, actions, gamma=0.9, epsilon=0.2, alpha=1, decay=0.9999954):
        self.actions = actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.decay = decay

        # initialize possiible sattes
        self.possible_states = []

        # Grid dimensions
        self.grid_rows = 4
        self.grid_cols = 5

        # Initialize Q, V, and policy π
        self.q_table = {}

        self.getPossibleStates()
        self.getInitLists()

    def getPossibleStates(self):
        # Generate all possible positions for both players
        for ax in range(self.grid_rows):
            for ay in range(self.grid_cols):
                for bx in range(self.grid_rows):
                    for by in range(self.grid_cols):
                        # Ensure players occupy distinct positions
                        if (ax, ay) != (bx, by):
                            # Ball possession can either be with Player A or Player B
                            for ball_possession in ['A', 'B']:
                                # Store the state as a tuple: ((Player A position), (Player B position), Ball possession)
                                state = ((ax, ay), (bx, by), ball_possession)
                                self.possible_states.append(state)

    def getInitLists(self):
        for state in self.possible_states:
            self.q_table[state] = {}
            for action in self.actions:
                self.q_table[state][action] = {}

          
    def choose_action(self, state):
        """Choose an action based on epsilon-greedy strategy."""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            a = np.argmax(list(self.q_table[state].values()))
            return self.actions[a]

    def update(self, state, action, joint_reward, next_state):
        reward = joint_reward[state[2]]
        """
        Update the Q-value based on the reward and next state.

        Args:
            s (int): Current state.
            a (str): Action taken.
            s_prime (int): Next state.
            reward (float): Received reward.
        """
        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        best_next_q_value = self.q_table[next_state][best_next_action]

        # Standard Q-learning update
        self.q_table[state][action] = (1 - self.alpha) * self.q_table[state][action] + \
                       self.alpha * (reward + self.gamma * best_next_q_value)

        self.alpha *= self.decay

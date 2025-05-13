import numpy as np
import random
from scipy.optimize import linprog

class MinimaxQAgent:
    def __init__(self, actions, opponent_actions, alpha=1.0, gamma=0.9, epsilon=0.1, total_steps = 10e6):
        self.actions = actions
        self.opponent_actions = opponent_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = 10 ** (np.log10(0.01) / total_steps)
        self.q_values = {}  # (state, a1, a2) -> value
        self.v_values = {}  # state -> value
        self.pi_values = {}  # Dictionary to store pi[(state, action)] = probability

    def get_q(self, state, a1, a2):
        return self.q_values.get((state, a1, a2), 1)
    
    def get_pi(self, state):
        n_actions = len(self.actions)
        return self.pi_values.get((state), 1/n_actions*np.ones(n_actions))

    def get_state_key(self, state):
        return tuple(state)

    def select_action(self, state, evaluate=False):
        if not evaluate and random.random() < self.epsilon:
            return random.choice(self.actions)
        elif(not evaluate):
            self.pi_values[state] = self.solve_minimax_policy(state)
            return random.choices(self.actions, weights=self.pi_values[state], k=1)[0]
        elif(evaluate):
            return random.choices(self.actions, weights=self.get_pi(state), k=1)[0]


    def solve_minimax_policy(self, state):
        state_key = self.get_state_key(state)
        n = len(self.actions)

        # Create the linear program: max min pi^T Q
        c = [0] * n + [-1]  # We maximize v (the value of the game)
        A = []
        b = []
        for j, a2 in enumerate(self.opponent_actions):
            row = [-self.get_q(state_key, a1, a2) for a1 in self.actions]
            row.append(1)  # coefficient for v
            A.append(row)
            b.append(0)

        A_eq = [[1] * n + [0]]
        b_eq = [1]

        bounds = [(0, 1)] * n + [(None, None)]  # pi_i in [0,1], v free

        result = linprog(c=c, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if result.success:
            pi = result.x[:n]
            self.v_values[state_key] = result.x[-1]
            return pi
        else:
            # Fall back to uniform policy
            print("Fall back to uniform policy")
            self.v_values[state_key] = 1
            return [1.0 / n] * n

    def update(self, state, action, opponent_action, reward, next_state):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        q = self.get_q(state_key, action, opponent_action)
        next_v = self.v_values.get(next_state_key, 1)

        target = reward + self.gamma * next_v
        self.q_values[(state_key, action, opponent_action)] = (1 - self.alpha) * q + self.alpha * target
        self.alpha *= self.decay
